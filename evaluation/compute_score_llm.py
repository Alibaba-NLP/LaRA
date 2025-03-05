from eval_utils import (
    load_data,
    dump_jsonl
)
import re
import string
from collections import Counter
import json
from tqdm import tqdm
import requests
import time
import os
import concurrent.futures
import csv
import dashscope
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--eval_model', default='qwen2.5-7b-instruct', type=str, help='')

args = parser.parse_args()

eval_model = args.eval_model

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

def get_score_one_llm(pred, label, query, query_type, model='qwen-max-allinone') -> float:
    if query_type == 'hallu':
        prompt = f'''You need to help me determine whether an AI model is hallucinating. I will provide you with a question, a ground truth answer, and a prediction from the model. If the prediction from the model and the ground truth answer are basically consistent, and it is determined that this question is not mentioned in the text, then the model is deemed not to be hallucinating and the answer is considered correct. If it is correct, you should only output True; if it is incorrect, only output False.

[Query] {query}

[Groundtruth Answer] {label}

[AI Assistant's Answer] {pred}

Now, start your judgement:'''
    else:
        prompt = f'''I will provide you with a question and its groundtruth answer, as well as an answer from an AI assistant. You need to judge whether the AI assistant's answer is correct based on the groundtruth answer. If it is correct, you should only output True; if it is incorrect, only output False.

[Query] {query}

[Groundtruth Answer] {label}

[AI Assistant's Answer] {pred}

Now, start your judgment:'''
    msg = [
        {
            "role": "system",
            "content": "You are a discriminator that judges whether the predictions to questions are correct.",
        },
        {"role": "user", "content": prompt},
    ]
    for _ in range(20):
        try:
            # response = call_qwen(model=model, messages=msg)
            response = call_gpt(eval_model, msg)
            if not response:
                return False
            else:
                if 'true' in response.lower():
                    return 1.0
                else:
                    return 0.0
        except:
            time.sleep(5)

def process_example(sample, query_type):
    pred = sample['prediction']
    query = sample['question']
    label = sample['ground_truth']
    score = get_score_one_llm(pred=pred, label=label, query=query, query_type=query_type)
    return score, sample     

query_type_list = ['location', 'reasoning', 'comp', 'hallu']
context_type_list = ['book', 'financial', 'paper']
length_list = ['32k', '128k']

save_all_path = f'./prediction/result/{eval_model}_all.jsonl'
save_order_path = f'./prediction/result/{eval_model}_order.jsonl'

for rag_or_full in ['rag', 'full']:
    for context_length in length_list:
        for query_type in query_type_list:
            for context_type in context_type_list:      
                print("\n============================================================")
                check = f'{rag_or_full}_{eval_model}_{context_length}_{context_type}_{query_type}'
                if os.path.exists(save_all_path):
                    with open(save_all_path, 'r') as f:
                        check_str = f.read()
                    if check in check_str:
                        print(f"{rag_or_full}_{eval_model}_{context_length}_{context_type}_{query_type} is already evaluated.")
                        continue

                print(f"current data is {rag_or_full}_{eval_model}_{context_length}_{context_type}_{query_type}")
                  
                data_path = f'./prediction/{eval_model}/{rag_or_full}_preds_{eval_model}_{context_length}_{context_type}_{query_type}.jsonl'

                score_all = 0.0
                cnt_all = 0
                score_location = {}
                cnt_location = {}

                data = load_data(data_path)
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [executor.submit(process_example, sample, query_type) for sample in data]
                    for future in futures:
                        score, sample = future.result()
                        if score is not False:
                            score_all += score
                            cnt_all += 1
                            if query_type in ['location', 'reasoning']:
                                if sample['context_order'] in score_location:
                                    score_location[sample['context_order']] += score
                                    cnt_location[sample['context_order']] += 1
                                else:
                                    score_location[sample['context_order']] = score
                                    cnt_location[sample['context_order']] = 1                         
                score_all_avg = score_all / cnt_all
                with open(save_all_path, 'a') as f:
                    f.write(f'{rag_or_full}_{eval_model}_{context_length}_{context_type}_{query_type}: {score_all_avg}, \cnt:{cnt_all}\n')
                if query_type in ['location', 'reasoning']:
                    with open(save_order_path, 'a') as f:
                        for loc in score_location:
                            acc = score_location[loc] / cnt_location[loc]
                            f.write(f'{rag_or_full}_{eval_model}_{context_length}_{context_type}_{query_type}:\n')
                            f.write(f'\tlocation {loc}: accuracy: {acc}, cnt: {cnt_location[loc]}\n')
                print(f"score all: {score_all / cnt_all}")

# Save evaluation results into csv format

input_file = save_all_path
output_csv = f'./prediction/result/{eval_model}_all.csv'

with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Task Name', 'Accuracy'])

    for line in lines:
        line = line.strip()
        parts = line.split(':')
        task_name = parts[0].strip()
        accuracy_info = parts[1].split(',')[0].strip()
        accuracy = float(accuracy_info)
        csvwriter.writerow([task_name, accuracy])

df = pd.read_csv(output_csv)
sub_tasks = ["location", "reasoning", "comp", "hallu"]
results = {'full32k': {}, 'rag32k': {}, 'full128k': {}, 'rag128k': {}}

for sub_task in sub_tasks:
    for length in ['32k', '128k']:
        for rf in ['full', 'rag']:
            filtered_df = df[df['Task Name'].str.contains(f"{rf}_") & df['Task Name'].str.contains(sub_task)
            & df['Task Name'].str.contains(length)]
            avg_score = round(filtered_df['Accuracy'].mean().item() * 100, 2)
            results[rf+length][sub_task] = avg_score

results['rag32k']['overall'] = round((results['rag32k']['location'] + results['rag32k']['reasoning'] + results['rag32k']['comp'] + results['rag32k']['hallu']) / 4, 2)
results['rag128k']['overall'] = round((results['rag128k']['location'] + results['rag128k']['reasoning'] + results['rag128k']['comp'] + results['rag128k']['hallu']) / 4, 2)
results['full32k']['overall'] = round((results['full32k']['location'] + results['full32k']['reasoning'] + results['full32k']['comp'] + results['full32k']['hallu']) / 4, 2)
results['full128k']['overall'] = round((results['full128k']['location'] + results['full128k']['reasoning'] + results['full128k']['comp'] + results['full128k']['hallu']) / 4, 2)


print(json.dumps(results, ensure_ascii=False, indent=4))
