# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, Optional, List
import logging

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse

import ray
from ray import serve
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import OpenAIServingModels, BaseModelPath, LoRAModulePath, PromptAdapterPath
from vllm.executor.mp_distributed_executor import MultiprocessingDistributedExecutor
from vllm.executor.ray_distributed_executor import RayDistributedExecutor
from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.logger import RequestLogger
from recurrent.impls.memory import MemoryAgent

logger = logging.getLogger("ray.serve")

app = FastAPI()

@serve.deployment(
    num_replicas=8,
    max_ongoing_requests=256,
    logging_config=dict(log_level="WARNING"),
)
@serve.ingress(app)
class VLLMDeployment:
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        response_role: str,
        lora_modules: Optional[List[LoRAModulePath]] = None,
        prompt_adapters: Optional[List[PromptAdapterPath]] = None,
        request_logger: Optional[RequestLogger] = None,
        chat_template: Optional[str] = None,
    ):
        logger.info(f"Starting with engine args: {engine_args}")
        self.serving_models = None
        self.openai_serving_chat = None
        self.openai_serving_completion = None
        self.engine_args = engine_args
        self.response_role = response_role
        self.lora_modules = lora_modules
        self.prompt_adapters = prompt_adapters
        self.request_logger = request_logger
        self.chat_template = chat_template

        if engine_args.model.startswith("hdfs:"):
            from hdfs import copy_local_path_from_hdfs
            print(f'start download from {engine_args.model}')
            import os
            local_model_path = copy_local_path_from_hdfs(src=engine_args.model, cache_dir=os.path.expanduser("~/.cache/verl/rlhf"))
            print('finish download')
        else:
            print(f"load from local dir {engine_args.model}")
            local_model_path = engine_args.model
        from pathlib import Path
        if Path(local_model_path).is_dir():
            self.model_name = Path(local_model_path).name
        else:
            self.model_name = local_model_path
        print(self.model_name)
        engine_args.disable_log_requests=True
        engine_args.model = local_model_path
        engine_args.tokenizer = local_model_path
        engine_args.distributed_executor_backend = RayDistributedExecutor
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def get_models(self):
        if not self.serving_models:
            self.serving_models = OpenAIServingModels(
                engine_client=self.engine,
                model_config=await self.engine.get_model_config(),
                base_model_paths=[
                    BaseModelPath(
                        name=self.model_name, model_path=self.engine_args.model
                    )
                ],
                lora_modules=self.lora_modules,
                prompt_adapters=self.prompt_adapters,
            )
        return self.serving_models
    
    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        """OpenAI-compatible HTTP endpoint.

        API reference:
            - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        if not self.openai_serving_chat:
            model_config = await self.engine.get_model_config()
            self.openai_serving_chat = OpenAIServingChat(
                self.engine,
                model_config,
                models=await self.get_models(),
                response_role=self.response_role,
                request_logger=self.request_logger,
                chat_template=self.chat_template,
                chat_template_content_format=None,
                # return_tokens_as_token_ids: bool = False,
                # enable_reasoning: bool = False,
                # reasoning_parser: str | None = None,
                # enable_auto_tools: bool = False,
                # tool_parser: str | None = None,
                # enable_prompt_tokens_details: bool = False
            )
        logger.info(f"Request: {request}")
        generator = await self.openai_serving_chat.create_chat_completion(
            request, raw_request
        )
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.code
            )
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())
        
    @app.get("/v1/models")
    async def show_available_models(self, raw_request: Request):
        model_config= await self.get_models()
        models = await model_config.show_available_models()
        return JSONResponse(content=models.model_dump())

def parse_vllm_args(cli_args: Dict[str, str]):
    """Parses vLLM args based on CLI inputs.

    Currently uses argparse because vLLM doesn't expose Python models for all of the
    config options we want to support.
    """
    arg_parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )

    parser = make_arg_parser(arg_parser)
    arg_strings = []
    for key, value in cli_args.items():
        arg_strings.extend([f"--{key}", str(value)])
    logger.info(arg_strings)
    parsed_args = parser.parse_args(args=arg_strings)
    return parsed_args

def build_app(cli_args: Dict[str, str]) -> serve.Application:
    """Builds the Serve app based on CLI arguments.

    See https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#command-line-arguments-for-the-server
    for the complete set of arguments.

    Supported engine arguments: https://docs.vllm.ai/en/latest/models/engine_args.html.
    """  # noqa: E501
    parsed_args = parse_vllm_args(cli_args)
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
    engine_args.worker_use_ray = True
    tp = engine_args.tensor_parallel_size
    logger.info(f"Tensor parallelism = {tp}")
    pg_resources = []
    pg_resources.append({"CPU": 1})  # for the deployment replica
    # Deployment replica will also use GPU for AsyncLLMEngine.
    for i in range(tp):
        pg_resources.append({"CPU": 1, "GPU": 1})  # for the vLLM actors, 

    print(f"{tp=}, {parsed_args=}, {engine_args=}")
    print("========================================")
    # We use the "STRICT_PACK" strategy below to ensure all vLLM actors are placed on
    # the same Ray node.
    ray.init()
    available_gpus = ray.available_resources()["GPU"]
    return VLLMDeployment.options(
        num_replicas=available_gpus // tp,
        placement_group_bundles=pg_resources,
        placement_group_strategy="STRICT_PACK",
    ).bind(
        engine_args,
        parsed_args.response_role,
        parsed_args.lora_modules,
        parsed_args.prompt_adapters,
        cli_args.get("request_logger"),
        parsed_args.chat_template,
    )

if __name__ == "__main__":
    # a quicker way
    import os
    import argparse
    pwd = os.path.dirname(os.path.abspath(__file__))
    file = os.path.splitext(os.path.basename(__file__))[0]
    parser = argparse.ArgumentParser(description="Ray Serve + vLLM deployment. Usage: python llm070.py --model Qwen/Qwen2.5-7B-Instruct --tp 2")
    parser.add_argument('--model', type=str, required=True,help='model name or path, e.g. Qwen/Qwen2.5-7B-Instruct or /mnt/hdfs/model/MemoryAgent-14B')
    parser.add_argument('--tp', type=int, default=1, help='tensor parallel size')    
    args = parser.parse_args()
    os.chdir(pwd)
    cmd = f"RAY_SERVE_QUEUE_LENGTH_RESPONSE_DEADLINE_S=1 exec serve run --name VLLMMultiDeployment {file}:build_app model={args.model} tensor-parallel-size={args.tp}"
    import subprocess
    p = subprocess.Popen(cmd, shell=True)
    try:
        p.wait()
    except KeyboardInterrupt:
        p.terminate()
        print("interrupted")
