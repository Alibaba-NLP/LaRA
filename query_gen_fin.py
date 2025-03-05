import time
import json
import os
import requests
import re
import random

from openai import OpenAI
import tiktoken

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--query_type', type=str, choices=['location', 'comp', 'reasoning', 'hallu'],
                    help='the type of the query')
parser.add_argument('--context_length', type=str, choices=['32k', '128k'], 
                    help='the length of the context')
parser.add_argument('--num_parts', type=int, help='num_parts')

args = parser.parse_args()
query_type = args.query_type
context_type = 'financial'
context_length = args.context_length
num_parts = args.num_parts

examples = {
    "location": '''Here are some examples of responses.\n <examples>
{"Q": "How many reportable business segments does 2U, Inc. have?", "A": "2U, Inc. has two reportable business segments."}
{"Q": "What was the amount of impairment charges recorded to goodwill during the three months ended December 31, 2023?", "A": "$62.8 million."}
{"Q": "What was the net cash used in financing activities during the year ended December 31, 2023?", "A": "$55.0 million."}
</examples>''',

    "reasoning": '''Here are some examples of responses.\n <examples>
{"Q": "What is the difference between the total lease payments and the present value of future minimum lease payments as of March 31, 2024, and what does this difference represent?", "A": "The total lease payments as of March 31, 2024, amount to $131,817, while the present value of future minimum lease payments is $114,807. The difference of $17,010 represents the amount of lease payments that accounts for interest, reflecting the amortization of future lease payments to their present value using the discount rate."}
{"Q": "What percentage of the total fair value of the liabilities assumed during the merger with DHC were warrant liabilities?", "A": "To calculate the percentage of warrant liabilities in the merger with DHC, sum the warrant liability\'s value, $1,913,737, and divide it by the total net liabilities assumed, $9,863,196. The percentage is found by (1,913,737 / 9,863,196) * 100, which equals approximately 19.4%."}
{"Q": "What was the main reason for the increase in general and administrative expenses for the three months ended March 31, 2024, compared to the same period in 2023?", "A": "The main reason for the increase in general and administrative expenses was due to transaction costs of $3.2 million incurred in connection with the Closing."}
</examples>''',

    "comp": '''Here are some examples of responses.\n <examples>
{"Q": "How did the average revenue per Full Course Equivalent (FCE) enrollment change in the Degree Program Segment compared to the Alternative Credential Segment from 2022 to 2023?", "A": "The average revenue per FCE enrollment increased by 17.8% in the Degree Program Segment from $2,447 in 2022 to $2,883 in 2023, while it decreased by 6% in the Alternative Credential Segment from $3,897 in 2022 to $3,662 in 2023."}
{"Q": "What was the primary reason for the decrease in Healthcare Solutions revenue for the year ended December 31, 2022, compared to 2021?", "A": "The primary reason for the decrease in Healthcare Solutions revenue for the year ended December 31, 2022, compared to 2021, was due to divestitures, negative impacts of foreign exchange, and lower sales volumes in the dental market as mentioned"}
{"Q": "What was the total bitcoin revenue for Block, Inc. in 2023, and how does it compare to the bitcoin cost?", "A": "The total bitcoin revenue for Block, Inc. in 2023 was $9,498,302,000, and the bitcoin costs were identical, as bitcoin costs fluctuate in line with bitcoin revenue."}
</examples>''',

    "hallu": '''Here are some examples of responses.\n <examples>
{"Q": "What measures are being taken to mitigate foreign currency risk?", "A": "The document does not specify any measures being taken to mitigate foreign currency risk." }
{"Q": "What was the impact of currency fluctuations on the company's revenue?", "A": "The financial statements do not provide information on the impact of currency fluctuations on the company's revenue."}
{"Q": "What is the company's market share in the non-combustible nicotine-related products sector?", "A": "The document does not provide a specific percentage for the company's market share in the non-combustible nicotine-related products sector."}
</examples>'''
}

api_key = ""
org_id = ""


def call_gpt(messages, retry_num=5, retry_interval=10):
    client = OpenAI(
        api_key=api_key,
        organization=org_id,
    )
    for _ in range(retry_num):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
            )
            text = completion.choices[0].message.content
            if isinstance(text, str) and text:
                return text
        except:
            time.sleep(retry_interval)
            print(f'retry after {retry_interval} seconds')
    return False

def truncate_input(input, max_length, manner="middle"):
    if len(input) <= max_length:
        return input
    if manner == "sc":
        num_parts = math.ceil(len(input) / max_length)
        all_parts = []
        for i in range(num_parts):
            if i < (num_parts - 1):
                all_parts.append(input[i * max_length : (i + 1) * max_length])
            else:
                all_parts.append(input[i * max_length :])
        return all_parts
    if manner == "middle":
        return input[0 : max_length // 2] + input[-max_length // 2 :]
    else:
        return None

prompt_easy = '''You are a financial question designer. Based on the provided financial statement, create a factual question and its corresponding answer that strictly satisfies these criteria:

Design Requirements:

1. Direct Extraction:
 - The answer must be explicitly stated in a single contiguous segment of the document
 - Require pinpoint localization (e.g., specific value, exact date, named section)

2. Answerability Constraints:
 - Unanswerable without the provided context
 - Must reference concrete elements (numerical values, named metrics, verbatim terms)

3. Response Specifications:
 - Maximum answer length: 20 words
 - Prohibit any interpretive phrasing'''

prompt_reasoning = '''You are a financial analyst. Design a question requiring mathematical/logical derivation from the financial statement, adhering to:

Design Requirements:

1. Require multi-step processing of:
 - Numerical calculations (e.g., ratios, growth rates)
 - Temporal comparisons
 - Reasonable inferences

2. Answerability Constraints:
 - Unanswerable without the provided context
 - Impossible to answer through simple lookup

3. Response Specifications:
 - Maximum answer length: avoid too long
 - Forbid speculative or probabilistic responses'''


prompt_comp = '''You are a financial analyst. Create a comparative question requiring synthesis of two distinct sections from the financial statement, following these principles:

Design Requirements:

1. Must reference:
 - Disparate metrics (e.g., departmental budgets vs regional sales)
 - Chronological differences (quarterly/annual comparisons)
 - Contrasting categories (actual vs projected figures)

2. Dependency Rules:
 - Each segment provides unique essential information
 - No overlapping data between required sections

3. Answerability Constraints:
 - Unanswerable without the provided context
 - Answer must demonstrate relational understanding
 - Require explicit mention of both referenced sections

4. Response Specifications:
 - Maximum answer length: avoid too long
 - Comparisons must use contextually appropriate units
'''

prompt_hallu = '''You are a financial analyst. Design a pseudo-relevant question that appears answerable but actually lacks sufficient basis in the financial statement, following these principles:

Design Requirements:

1. Surface-level Relevance:
 - Use document-specific terminology
 - Reference actual sections/metrics as distractors

2. Unanswerability Guarantees:
 - Absolutely not mentioned in the context
 - Missing critical data points required for resolution
 - No inferential path from provided information

3. Confirm absence of: 
 - Direct mentions
 - Implied values
 - Comparable proxies'''

prompt_template = {
    "location": prompt_easy,
    "reasoning": prompt_reasoning,
    'comp': prompt_comp,
    'hallu': prompt_hallu
}

def get_qa(sys_prompt, context, examples):
    prompt = '''<financial statement>\n{financial_statement}\n</financial statement>
You should respond with a Python dict. The keys of the dict should be Q and A, and the values should be the content of the question and answer, respectively. Strings in the dict should be enclosed in double quotes.
'''
    prompt = prompt.format(financial_statement=context)    
    prompt += examples    
    msg = [
            {
                "role": "system",
                "content": sys_prompt,  
            },  # noqa
            {"role": "user", "content": prompt},
        ]


    response = call_gpt(model, gpt_url, msg, retry_num=5, retry_interval=10)
    return response


folder_path = f'./datasets/{context_length}/{context_type}'


file_list = os.listdir(folder_path)

queries = []
egs = examples[query_type]
sys_prompt = prompt_template[query_type]


for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    if file_name.endswith('.txt'):
        print(f'start processing {file_name}')
        with open(file_path, 'r', encoding='utf-8') as file:
            file_text = file.read()
        if query_type in ['location', 'reasoning', 'hallu']:
            for i in range(num_parts):
                len_part = len(file_text) // num_parts
                query = {}
                query["type"] = context_type
                query["level"] = query_type
                query["length"] = context_length
                query["file"] = file_name
                if i != (len_part - 1):
                    context = file_text[i * (len_part): (i+1) * (len_part)]
                else:
                    context = file_text[i * (len_part):]
                # query["context"] = context
                query["context_order"] = i
                try:
                    response = get_qa(sys_prompt, context, egs)
                    start_index = response.find('{')
                    end_index = response.rfind('}')
                    qa_dict = response[start_index: end_index + 1]
                    qa_dict = eval(qa_dict)
                    query['question'] = qa_dict['Q']
                    query['answer'] = qa_dict['A']
                    queries.append(query)
                except:
                    print(response)
                    continue

        if query_type in ["reasoning", "hallu"]:
            query = {}
            query["type"] = context_type
            query["level"] = query_type
            query["length"] = context_length
            query["file"] = file_name
            query["context_order"] = "full"
            
            tokenizer = tiktoken.encoding_for_model("gpt-4")
            tokens = tokenizer.encode(file_text)
            tokens = truncate_input(tokens, 128_000 - 2000, manner="middle")
            context = tokenizer.decode(tokens)

            try:
                response = get_qa(sys_prompt, context, egs)
                start_index = response.find('{')
                end_index = response.rfind('}')
                qa_dict = response[start_index: end_index + 1]
                qa_dict = eval(qa_dict)
                query['question'] = qa_dict['Q']
                query['answer'] = qa_dict['A']
                queries.append(query)
            except:
                print(response)
                continue
        
        if query_type == 'comp':
            len_part = len(file_text) // num_parts
            comp_list = []
            while len(comp_list) < 3:
                query = {}
                query["type"] = context_type
                query["level"] = query_type
                query["length"] = context_length
                query["file"] = file_name
                numbers = list(range(num_parts))
                selected_numbers = random.sample(numbers, 2)
                sorted_numbers = sorted(selected_numbers)
                if sorted_numbers not in comp_list:
                    comp_list.append(sorted_numbers)
                    p1_i = sorted_numbers[0]
                    p2_i = sorted_numbers[1]
                    seg_1 = file_text[p1_i * (len_part): (p1_i+1) * (len_part)]
                    seg_2 = file_text[p2_i * (len_part): (p2_i+1) * (len_part)]
                    context = f"<segment 1>\n{seg_1}<\end of segment 1>\n\n\n <segment 2>\n{seg_2}<\end of segment 2>"
                    query['comp_parts'] = sorted_numbers
                    try:
                        response = get_qa(sys_prompt, context, egs)
                        start_index = response.find('{')
                        end_index = response.rfind('}')
                        qa_dict = response[start_index: end_index + 1]
                        qa_dict = eval(qa_dict)
                        query['question'] = qa_dict['Q']
                        query['answer'] = qa_dict['A']
                        queries.append(query)
                    except:
                        print(response)
                        continue        

save_file_path = os.path.join('./datasets/query', f'{context_length}_{context_type}_{query_type}.jsonl')
with open(save_file_path, 'w') as file:
    first_line = True
    for line in queries:
        if first_line:
            file.write(json.dumps(line, ensure_ascii=False))
            first_line = False
            continue
        else:
            file.write('\n')
            file.write(json.dumps(line, ensure_ascii=False))