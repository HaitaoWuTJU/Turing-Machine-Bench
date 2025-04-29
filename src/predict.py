from vllm import LLM, SamplingParams
import argparse
import random
import numpy as np
import json
import os
from transformers import AutoTokenizer
from prompt import generate_prompt
cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')
tensor_parallel_size = len(cuda_visible_devices.split(','))

seed = 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model with main.py")
        
    parser.add_argument("--model", type=str, default='Qwen/Qwen2.5-14B-Instruct', help="Model name or path to use")
    parser.add_argument("--bench_name", type=str,default="TMBench")
    parser.add_argument("--max_token", type=int, default=16384, choices=[8192,16384,32768], help="max_token: 8192,16384,32768")
    parser.add_argument("--temperature", type=float, default=0.0, help="temperature")
    parser.add_argument("--gpu_memory_utilization",type=float,default=0.92)
    parser.add_argument("--chat", action="store_true", default=False)

    args = parser.parse_args()
    temperature = args.temperature
    model = args.model
    model_name = model.split('/')[-1]
    max_token = args.max_token
    gpu_memory_utilization=args.gpu_memory_utilization
    bench_name = args.bench_name
    chat = args.chat

    
    tokenizer = AutoTokenizer.from_pretrained(model)

    sampling_params = SamplingParams(temperature=temperature, top_p=0.8, max_tokens=max_token, seed=seed) #top p  not work when temp=0.0

    input_json_path = f'data/{bench_name}.json'
    if chat:
        output_file = f'results/{bench_name}_{model_name}_{temperature}_chat.json'
    else:
        output_file = f'results/{bench_name}_{model_name}_completions.json'
    data, prompts_text = generate_prompt(input_json_path)
    chat_prompts_text = []
    for prompt in prompts_text:
        chat = [
                {"role": "system", "content": "You are a helpful assistant. Please reason step by step."},
                {"role": "user", "content": prompt},
            ]
        chat_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        chat_prompts_text.append(chat_prompt)
    # params = BeamSearchParams(beam_width=5, max_tokens=50)
    # outputs = llm.beam_search([{"prompt": "Hello, my name is "}], params)
    llm = LLM(model=model, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=gpu_memory_utilization)
    if chat:
        outputs = llm.generate(chat_prompts_text, sampling_params) #prompts_text, chat_prompts_text
    else:
        outputs = llm.generate(prompts_text, sampling_params)

    for i, entry, output in zip(range(len(outputs)), prompts_text, outputs):
        generated_text = output.outputs[0].text
        generated_token_count = len(output.outputs[0].token_ids)

        data["samples"][i]['ans'] =  generated_text
        data["samples"][i]['num_token'] =  generated_token_count

    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=1,ensure_ascii=False)
        