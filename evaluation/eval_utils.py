import json
import re
from typing import Optional

from prompt import (
    full_templates,
    rag_templates
)

MODEL_TO_PROMPT_TEMPLATE = {
    "full": full_templates,
    "rag": rag_templates
}

def load_data(data_path):
    data_list = []
    with open(data_path, "r") as f:
        for line in f:
            data_list.append(json.loads(line))
    return data_list

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


def create_msgs(tokenizer, eg: dict, data_name: str, model_name:str) -> tuple[list[dict], str]:
    """
    Create messages for a given example.
    """
    prompt = create_prompt(eg, data_name, model_name)

    # Check if tokenizer is provided and initialized
    if tokenizer and model_name == 'full':
        tokens = tokenizer.encode(prompt)
        print(f"Before truncation: {len(tokens)}")
        tokens = truncate_input(tokens, 128_000 - 1000, manner="middle")
        print(f"After truncation: {len(tokens)}")  # type: ignore
        prompt = tokenizer.decode(tokens)

    return [
        {
            "role": "system",
            "content": "You are a helpful assistant",  # noqa
        },  # noqa
        {"role": "user", "content": prompt},
    ], prompt

def create_prompt(eg: dict, data_name: str, model_name: Optional[str]) -> str:
    """
    Create prompt for a given example.

    Args:
        eg: example dict
        data_name: name of the dataset/task
        model_name: optional, used to fetch model-specific templates.
    """

    # Directly use the appropriate template if the model_name is provided.
    if model_name and model_name in MODEL_TO_PROMPT_TEMPLATE:
        templates = MODEL_TO_PROMPT_TEMPLATE[model_name]
        template = templates[data_name]
    else:
        # If no model-specific template, return a basic prompt or handle differently.
        return eg["context"]

    # Now create the prompt based on the template and task data

    if data_name in [
        "longbook_choice_eng",
        "longbook_qa_eng",
        "longbook_sum_eng",
        "longbook_qa_chn",
    ]:
        book = eg["context"]
        if data_name == "longbook_choice_eng":
            return template.format(
                question=eg["input"],
                context=book,
                OPTION_A=eg["options"][0],
                OPTION_B=eg["options"][1],
                OPTION_C=eg["options"][2],
                OPTION_D=eg["options"][3],
            )
        elif data_name == "longbook_qa_eng":
            return template.format(
                question=eg["question"],
                context=book,
            )
        else:
            raise ValueError


    format_dict = {
        "context": eg["context"],
        "question": eg["question"],
    }
    prompt = template.format(**format_dict)
    return prompt

def dump_jsonl(data, fname):
    with open(fname, "w", encoding="utf8") as fout:
        for line in data:
            fout.write(json.dumps(line, ensure_ascii=False) + "\n")
