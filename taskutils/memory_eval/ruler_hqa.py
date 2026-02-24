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
import os, csv, json
import argparse
import time
from tqdm import tqdm
from datasets import load_dataset
import re
from transformers import AutoTokenizer
import tiktoken
import torch.multiprocessing as mp
import string
from collections import Counter
from datasets import load_dataset, concatenate_datasets

from utils import extract_solution,update_answer
from utils.envs import DATAROOT


def calc_metrics(predictions, goldens):
    assert len(predictions) == len(goldens)
    metrics = {'f1': 0, 'prec': 0, 'recall': 0, 'em': 0, 'sub_em': 0, 'total_num': 0}
    for pred, gold in zip(predictions, goldens):
        update_answer(metrics, pred, gold)
    for k, _ in metrics.items():
        if k == 'total_num':
            continue
        metrics[k] = round((metrics[k]/metrics['total_num']), 2)
    return metrics


def get_pred(data, args, out_file):
    model = args.model
    if "gpt" in model or "o1" in model or "o3" in model or "o4" in model or "gemini" in model or "claude" in model:
        tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if args.api == "openai":
        from utils.openai_api import async_query_llm
        from utils import extract_answer
    elif args.api == "recurrent":
        from utils.recurrent import async_query_llm
        from utils import extract_answer
    elif args.api == "recurrent-boxed":
        from utils.recurrent_boxed import async_query_llm
        from utils import extract_boxed_answer as extract_answer
    elif args.api == "boxed":
        from utils.boxed import async_query_llm
        from utils import extract_boxed_answer as extract_answer
    else:
        print(f"Invalid API: {args.api}")
        raise ValueError
    coros = []
    for item in data:
        coro = async_query_llm(item, model, tokenizer, temperature=0.7, top_p=0.95)
        coros.append(coro)
    from utils.aio import async_main, close_async_client
    import uvloop
    outputs = uvloop.run(async_main(coros, args.n_proc))
    uvloop.run(async_main([close_async_client()]))
    from collections import defaultdict
    scores = defaultdict(list)
    fout = open(out_file, 'w' if args.force else 'a', encoding='utf-8')
    for i, (output, item) in enumerate(zip(outputs, data)):
        if output == '':
            continue
        response = output.strip()
        pred, _ = extract_solution(response)
        item['response'] = response
        item['answer'] = item["answers"][0]
        item['pred'] = extract_answer(pred) if pred else extract_answer(response)
        item['judge_f1'] = calc_metrics([item["pred"]], [item["answer"]])['f1'] if item["pred"] else 0
        item['judge_em'] = calc_metrics([item["pred"]], [item["answer"]])['em'] if item["pred"] else 0
        item['judge_sub_em'] = calc_metrics([item["pred"]], [item["answer"]])['sub_em'] if item["pred"] else 0
        scores['f1'].append(item['judge_f1'])
        scores['em'].append(item['judge_em'])
        scores['sub_em'].append(item['judge_sub_em'])
        item.pop('context');fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        if i == 0:
            print("="*40 + "New Item Start" + "="*40)
            print(item['response'])
            print("-"*80)
            print(item['pred'])
            print("-"*80)
            print(item['answer'])
            print("-"*80)
            print(item['judge_sub_em'])
            print("="*40 + "New Item End" + "="*40)
    fout.close()
    print(f"ruler_hqa [{args.length}]")
    for k, v in scores.items():
        print(f"{k}: {round(sum(v) * 100 /len(v), 2)}")
    print(f"Total: {len(data)}")


def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)
    out_file = os.path.join(args.save_dir, args.save_file + ".jsonl")

    dataset = concatenate_datasets([
            load_dataset("json", data_files=f"{DATAROOT}/eval_{args.length}.json", split="train"),
        ])
        
    print(f"original data len {len(dataset)}")
    # 通过深拷贝生成新数据集
    import copy
    dataset = [copy.deepcopy(item) for _ in range(args.sampling) for item in dataset]
    print(f"sampling data len {len(dataset)}")

    data_all = []
    for idx, item in enumerate(dataset):
        item["_id"] = idx  # 现在每个 item 是独立对象
        data_all.append(item)

    print(data_all[0]["_id"])
    print(data_all[-1]["_id"])

    # cache
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}
    data = []
    for item in data_all:
        if item["_id"] not in has_data or args.force:
            data.append(item)
        elif args.force:
            data.append(item)

    get_pred(data, args, out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=50, choices=[50, 100, 200, 400, 800, 1600, 3200, 6400])
    parser.add_argument("--save_dir", "-s", type=str, default="results/ruler_hqa")
    parser.add_argument("--save_file", "-f", type=str, default="Qwen2.5-7B-Instruct-recurrent")
    parser.add_argument("--model", "-m", type=str, default="Qwen2.5-7B-Instruct")
    parser.add_argument("--tokenizer", "-t", type=str, default="/mnt/hdfs/hongli/model/Qwen2.5-7B-Instruct")
    parser.add_argument("--n_proc", "-n", type=int, default=64)
    parser.add_argument("--api", "-a", type=str, default="recurrent")
    parser.add_argument("--sampling", "-p", type=int, default=1)
    parser.add_argument('--force', action='store_true', help='force to overrite')
    args = parser.parse_args()
    main()