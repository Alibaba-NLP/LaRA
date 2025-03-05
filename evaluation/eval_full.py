from openai import OpenAI
import tiktoken
from pathlib import Path
from eval_utils import (
    load_data,
    create_msgs,
    dump_jsonl
)
import time
import json
import requests
import concurrent.futures
import dashscope
import time
import json
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--query_type', type=str, help='the type of the query')
parser.add_argument('--context_type', type=str, help='the type of the context')
parser.add_argument('--context_length', type=str, help='the length of the context')
parser.add_argument('--eval_model', type=str, help='model')

args = parser.parse_args()
eval_model = args.eval_model
query_type = args.query_type
context_type = args.context_type
context_length = args.context_length

api_key = ""
org_id = ""

def call_qwen(
    model,
    messages,
    retry_num: int = 10,
    retry_interval: int = 1,
    ):
    dashscope.api_key = api_key
    for _ in range(retry_num):
        response = dashscope.Generation.call(
            model=model,
            messages=messages,
            enable_search=False,
        )
        print(response)
        try:
            text = response.output.text
            if isinstance(text, str) and text:
                return text
        except:
            time.sleep(retry_interval)
    return False

def call_gpt(model, messages, retry_num=5, retry_interval=5):
    client = OpenAI(
        api_key=api_key,
        organization=org_id,
    )
    for _ in range(retry_num):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            text = completion.choices[0].message.content
            if isinstance(text, str) and text:
                return text
        except:
            time.sleep(retry_interval)
            print(f'retry after {retry_interval} seconds')
    return False

def process_example(eg):
    file_name = eg['file']
    eg_file_path = f"../datasets/{context_length}/{context_type}/{file_name}"
    with open(eg_file_path, 'r', encoding='utf-8') as f:
        eg_txt = f.read()
    eg['context'] = eg_txt

    msgs, prompt = create_msgs(
        tokenizer, eg, data_name=context_type, model_name="full"
    )
    # Make prediction
    try:
        if 'qwen' in eval_model:
            response = call_qwen(eval_model, msgs)
        if 'gpt' in eval_model:
            response = call_gpt(eval_model, msgs)

        return response, eg
    except Exception as e:
        print("ERROR:", e)
        time.sleep(10)
        return False, eg
    
if __name__ == "__main__":

    data_path = f'../datasets/query/{context_length}_{context_type}_{query_type}.jsonl'
    examples = load_data(data_path)
    output_path = f'./prediction/{eval_model}/full_preds_{eval_model}_{context_length}_{context_type}_{query_type}.jsonl'
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    preds = []

    print("===================================================")
    print(f"currently processing full_{eval_model}_{context_length}_{context_type}_{query_type}")

    if 'qwen' in eval_model:
        max_workers = 8
    else:
        max_workers = 4

    # for i in range(10):
    #     response, eg = process_example(examples[i])
    #     breakpoint()

    if not os.path.exists(output_path):
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_example, example) for example in examples]
            for future in futures:
                result, eg = future.result()
                if query_type in ['location', 'reasoning', 'hallu']:
                    preds.append(
                        {
                            "type": eg['type'],
                            "level": eg['level'],
                            "file": eg['file'],
                            "context_order": eg['context_order'],
                            "question": eg['question'],
                            "prediction": result,
                            "ground_truth": eg['answer'],
                        }
                    )
                if query_type in ['comp']:
                    preds.append(
                        {
                            "type": eg['type'],
                            "level": eg['level'],
                            "file": eg['file'],
                            "comp_parts": eg['comp_parts'],
                            "question": eg['question'],
                            "prediction": result,
                            "ground_truth": eg['answer'],
                        }
                    )
        dump_jsonl(preds, output_path)