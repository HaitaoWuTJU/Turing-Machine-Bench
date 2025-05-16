import json
from tqdm import tqdm
from vllm import LLM, SamplingParams
import argparse
from datasets import load_dataset
import re, os
import copy
import random
from collections import defaultdict
from transformers import AutoTokenizer
import torch
tensor_parallel_size = torch.cuda.device_count()

seed = 1
def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

# Chain-of-Thought Prompt
def format_prompt(example):
    q = example["Question"]

    choices = [
            preprocess(example["Incorrect Answer 1"]),
            preprocess(example["Incorrect Answer 2"]),
            preprocess(example["Incorrect Answer 3"]),
            preprocess(example["Correct Answer"]),
        ]
    random.shuffle(choices)
    correct_answer_index = choices.index(preprocess(example["Correct Answer"]))
    gold = chr(65 + correct_answer_index)

    prompt = (
        f"What is the correct answer to this question:{q}\n"
        f"Choices:\n"
        f"(A) {choices[0]}\n(B) {choices[1]}\n(C) {choices[2]}\n(D) {choices[3]}\n"
        f"Please reason step by step, and put your final answer (A, B, C, or D) within \\boxed{{}}."
    )
    return prompt,gold

def postprocess_strict(text):
    # match = re.search(r"(?<=The answer is )([A-D])(?=\.)", text)
    match = re.search(r"\\boxed\{.*?([A-D]).*?\}", text) 
    return match.group(1) if match else None

def postprocess_flexible(text):
    match = re.findall(r"\(([A-D])\)", text, re.IGNORECASE)
    return match[-1].upper() if match else None

def exact_match(pred, gold):
    return pred is not None and pred.strip().upper() == gold.strip().upper()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model with main.py")
    parser.add_argument("--model", type=str, default='Qwen/Qwen2.5-72B-Instruct', help="Model name or path to use")
    args = parser.parse_args()
    model = args.model
    model_name = model.split('/')[-1]

    
    tokenizer = AutoTokenizer.from_pretrained(model)

    dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")


    prompts_text_gold = [format_prompt(item) for item in dataset]
    prompts_text = [ele[0] for ele in prompts_text_gold]

    chat_prompts = []
    for prompt_text in prompts_text:
        chat = [
                {"role": "system", "content": "You are a helpful assistant. Please reason step by step."},
                {"role": "user", "content": prompt_text},
            ]
        chat_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        chat_prompts.append(chat_prompt)

    
    gold_list = [ele[1] for ele in prompts_text_gold]

    llm = LLM(model=model, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=0.9)

    sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=16384, seed=seed)
    outputs = llm.generate(chat_prompts, sampling_params)

    save_results = defaultdict(dict)

    correct = 0
    total = len(dataset)
    for i, (item, output) in enumerate(zip(dataset,outputs)):
        gold = gold_list[i]

        response = output.outputs[0].text.strip()
        pred = postprocess_strict(response) or postprocess_flexible(response)
    
        acc = exact_match(pred, gold)

        save_results[i]['prompt'] = prompts_text[i]
        save_results[i]['ans'] = response
        save_results[i]['gold'] = gold
        save_results[i]['pred'] = pred
        save_results[i]['acc']=acc

        correct += acc

    accuracy = correct / total
    print(f"Model: {model_name}, GPQA Diamond Accuracy: {accuracy:.2%}")

    os.makedirs(f"results/gpqa_diamond", exist_ok=True)
    output_file = f"results/gpqa_diamond/{model_name}_cot_chat.json"

    with open(output_file, 'w') as json_file:
        json.dump(save_results, json_file, indent=1)
