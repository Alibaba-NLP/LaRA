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
import os

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
    DashScopeTextEmbeddingType,
)
from search.simpleHybridSearcher import SimpleHybridSearcher
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
import transformers

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
    outputs = pipeline(messages, max_new_tokens=1024)
    response = outputs[0]["generated_text"][-1]['content']
    return response

def process_example(eg, pipeline):
    file_name = eg['file']
    eg_file_path = f"../datasets/{context_length}/{context_type}/{file_name}"
    with open(eg_file_path, 'r', encoding='utf-8') as f:
        eg_txt = f.read()

    # set the retriever
    transformations = []
    splitter = SentenceSplitter(
        include_metadata=True, include_prev_next_rel=True,
        chunk_size=600,
        chunk_overlap=100,
        separator=' ',       
        paragraph_separator='\n\n\n', secondary_chunking_regex='[^,.;。？！]+[,.;。？！]?')
    transformations.append(splitter)
    embed_model = HuggingFaceEmbedding(model_name="../embedding_models/gte-enzh-emb-large-v1.5")
    transformations.append(embed_model)
    rag_pipeline = IngestionPipeline(
        transformations=transformations
    )

    # RAG模块检索chunk
    eg_doc = Document(text=eg_txt)
    documents = [eg_doc]
    nodes = rag_pipeline.run(documents=documents, show_progress=False)
    config = {
        "class_name": "SimpleHybridSearcher",
        "class_file": "simpleHybridSearcher",
        "remove_if_exists": False,
        "thread_num": 1,
        "rerank_size": 5,
        "vector_ratio": 0.5,
        "embed_model_name": "../embedding_models/gte-enzh-emb-large-v1.5",
        "rerank_model": "../embedding_models/gte-rerank-large-v1.5"
    }

    searcher = SimpleHybridSearcher(config, nodes)
    search_nodes = searcher.process(eg['question'])
    eg["context"] = ''
    for i, node in enumerate(search_nodes):
        node_txt = node.text
        eg["context"] += f'chunk {i}: {node_txt}.' + '\n\n'

    msgs, prompt = create_msgs(
        g4_tokenizer, eg, data_name=context_type, model_name="rag"
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
    # set the gpt model
    print("========loading the model=======")
    # model, tokenizer = init_model(model_path)
    pipeline = init_model(model_path)
    print("=======loading done!=======")

    data_path = f'../datasets/query/{context_length}_{context_type}_{query_type}.jsonl'
    examples = load_data(data_path)
    output_path = f'./prediction/{eval_model}/rag_preds_{eval_model}_{context_length}_{context_type}_{query_type}.jsonl'
    g4_tokenizer = tiktoken.encoding_for_model("gpt-4")

    preds = []

    print("===================================================")
    print(f"currently processing rag_{eval_model}_{context_length}_{context_type}_{query_type} ")


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
