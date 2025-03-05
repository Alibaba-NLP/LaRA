import time
import json
import requests
import concurrent.futures
import os

from openai import OpenAI
import tiktoken
from pathlib import Path
from eval_utils import (
    load_data,
    create_msgs,
    dump_jsonl
)

import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
import transformers
import dashscope


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--query_type', type=str, help='the type of the query')
parser.add_argument('--context_type', type=str, help='the type of the context')
parser.add_argument('--context_length', type=str, help='the length of the context')
parser.add_argument('--eval_model', type=str, help='model')

MODEL_PATH = {
    "llama-3.1-8b": "../models/Meta-Llama-3___1-8B-Instruct",
    "llama-3.2-3b": "../models/Llama-3___2-3B-Instruct",
}

args = parser.parse_args()
eval_model = args.eval_model
query_type = args.query_type
context_type = args.context_type
context_length = args.context_length
model_path = MODEL_PATH[eval_model]


def init_model(model_path):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        device_map="auto",
    )
    return pipeline

def call_model(pipeline, messages):
    outputs = pipeline(
        messages,
        max_new_tokens=1024,
    )
    response = outputs[0]["generated_text"][-1]['content']
    return response

def process_example(eg, pipeline):
    file_name = eg['file']
    eg_file_path = f"../datasets/{context_length}/{context_type}/{file_name}"
    with open(eg_file_path, 'r', encoding='utf-8') as f:
        eg_txt = f.read()
    eg['context'] = eg_txt

    msgs, prompt = create_msgs(
        g4_tokenizer, eg, data_name=context_type, model_name="full"
    )
    # Make prediction
    try:
        response = call_model(pipeline, msgs)
        if type(response) == str:
            print('success')
        return response, eg
    except Exception as e:
        print("ERROR:", e)
        time.sleep(10)
        return False, eg
    

if __name__ == "__main__":
    # set the model
    print("========loading the model=======")
    # model, tokenizer = init_model(model_path)
    pipeline = init_model(model_path)
    print("=======loading done!=======")

    data_path = f'../datasets/query/{context_length}_{context_type}_{query_type}.jsonl'
    examples = load_data(data_path)
    output_path = f'./prediction/{eval_model}/full_preds_{eval_model}_{context_length}_{context_type}_{query_type}.jsonl'
    g4_tokenizer = tiktoken.encoding_for_model("gpt-4")
    preds = []

    print("===================================================")
    print(f"currently processing full_{eval_model}_{context_length}_{context_type}_{query_type}")

    if not os.path.exists(output_path):
        for i in range(len(examples)):
            response, eg = process_example(examples[i], pipeline)
            if query_type in ['location', 'reasoning', 'hallu']:
                preds.append(
                    {
                        "type": eg['type'],
                        "level": eg['level'],
                        "file": eg['file'],
                        "context_order": eg['context_order'],
                        "question": eg['question'],
                        "prediction": response,
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
                        "prediction": response,
                        "ground_truth": eg['answer'],
                    }
                )
        dump_jsonl(preds, output_path)
