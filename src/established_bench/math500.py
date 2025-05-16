import json,os
from tqdm import tqdm
from vllm import LLM, SamplingParams
import argparse
from datasets import load_dataset
import re, os
import copy
import random
from collections import defaultdict
from transformers import AutoTokenizer
from mathruler.grader import extract_boxed_content, grade_answer

import torch
tensor_parallel_size = torch.cuda.device_count()

seed = 1


# Chain-of-Thought Prompt
def format_prompt(example):
    q = example["problem"]

    prompt = (
        f"What is the correct answer to this question: {q}\n"
        f"Please reason step by step, and put your final answer within \\boxed{{}}."
    )
    return prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model with main.py")
    parser.add_argument("--model", type=str, default='Qwen/Qwen2.5-72B-Instruct', help="Model name or path to use")
    args = parser.parse_args()
    model = args.model
    model_name = model.split('/')[-1]

    dataset = load_dataset("simplescaling/openaimath", split="test")

    tokenizer = AutoTokenizer.from_pretrained(model)

    prompts_text = [format_prompt(item) for item in dataset]
    # prompts_text = [ele[0] for ele in prompts_text_gold]
    chat_prompts = []
    for prompt_text in prompts_text:
        chat = [
                {"role": "system", "content": "You are a helpful assistant. Please reason step by step."},
                {"role": "user", "content": prompt_text},
            ]
        chat_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        chat_prompts.append(chat_prompt)

    llm = LLM(model=model, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=0.9)
    sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=16384, seed=seed)
    outputs = llm.generate(chat_prompts, sampling_params)

    save_results = defaultdict(dict)

    correct = 0
    total = len(dataset)
    for i, (item, output) in enumerate(zip(dataset,outputs)):
        
        response = output.outputs[0].text.strip()

        gold = item['answer']
        save_results[i]['prompt'] = prompts_text[i]
        save_results[i]['ans'] = response
        save_results[i]['gold'] = gold
        

        pred = extract_boxed_content(response)
        acc = grade_answer(pred, gold)

        save_results[i]['pred'] = pred
        save_results[i]['acc'] = acc
        correct += acc

    accuracy = correct / total
    print(f"Model: {model_name}, MATH500 Accuracy: {accuracy:.2%}")

    os.makedirs(f"results/math500", exist_ok=True)
    output_file = f"results/math500/{model_name}_cot_chat.json"

    with open(output_file, 'w') as json_file:
        json.dump(save_results, json_file, indent=1)
