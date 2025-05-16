import argparse
import random, os
import numpy as np
import json
import time
import sys
from openai import OpenAI, RateLimitError
import re
# from prompt import generate_prompt
from prompt import generate_prompt
# from prompt_long import generate_prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model with main.py")
        
    parser.add_argument("--model", type=str, default='chatgpt-4o-latest', help="Model name or path to use")
    parser.add_argument("--bench_name", type=str)
    parser.add_argument("--max_token", type=int, default=65536, choices=[100,8192,16384,32768,65536,131072], help="max_token: 8192,16384,32768")
    parser.add_argument("--gpu_memory_utilization",type=float,default=0.92)
    parser.add_argument("--api",type=str,default='apiyi')
    parser.add_argument("--stream", action="store_true", default=False)

    args = parser.parse_args()
    model = args.model
    max_token = args.max_token
    gpu_memory_utilization=args.gpu_memory_utilization
    bench_name = args.bench_name
    api = args.api
    stream = args.stream

    input_json_path = f'data/{bench_name}.json'
    output_file = f'results/{bench_name}_{model}_chat.json'


    if api == 'ali':
        key = 'xxxxxx'
        url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        if model in ['llama-4-scout-17b-16e-instruct','llama-4-maverick-17b-128e-instruct','qwen-max','qwen3-235b-a22b','qwen3-30b-a3b']:
            max_token = min(max_token,8192)
    elif api=='google':
        key = 'xxxxxx'
        url = 'https://generativelanguage.googleapis.com/v1beta/openai'
    elif api == 'deepseek':
        key = 'xxxxxx'
        max_token = min(max_token,8192)
        url = 'https://api.deepseek.com/v1'
    elif api =='doubao':
        key = 'xxxxxx'
        max_token = min(max_token,12288)
        url = 'https://ark.cn-beijing.volces.com/api/v3'
    else:
        print('exit, no api')
        exit(1)
    index = None
    data, prompts_text = generate_prompt(input_json_path,index)

    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            save_data = json.load(f)
    else:
        save_data = data
        
    _api_call=0
    for i, prompt in enumerate(prompts_text):
        if isinstance(key, list):
            _key = random.choice(key)
        else:
            _key = key
        client = OpenAI(api_key=_key, base_url=url)
        print(f"\n{_api_call=} {i+1}/{len(prompts_text)}")
        # if _api_call>2:
        #     break
        
        if 'ans' in save_data["samples"][i] and save_data["samples"][i]['ans']: #and 'num_token' in save_data["samples"][i] and save_data["samples"][i]['num_token']>5
            continue
        else:
            if model in ['o3-mini','openai/o3-mini-high','o3-mini-2025-01-31']:
                chat = [
                        {"role": "user", "content": "You are a helpful assistant. Please reason step by step." + prompt},
                    ]
                response = client.chat.completions.create(
                            model = model,
                            messages = chat,
                            max_tokens=max_token, temperature=0.0,reasoning_effort="medium", timeout=1200) #reasoning_effort="high", low medium
                
            elif model=='o1-mini':
                chat = [
                        {"role": "user", "content": "You are a helpful assistant. Please reason step by step." + prompt},
                    ]
                response = client.chat.completions.create(
                            model = model,
                            messages = chat,
                            max_tokens=max_token, temperature=0.0, timeout=900)
            else:
                chat = [
                        {"role": 'system', "content": "You are a helpful assistant. Please reason step by step."},
                        {"role": "user", "content": prompt},
                    ]
                if stream:
                    response = client.chat.completions.create(
                                model = model,
                                messages = chat,
                                max_tokens=max_token, temperature=0.0, top_p=0.8, timeout=900, stream=True)
                    streamed_text = ""
                    # generated_token_count = 0
                    for chunk in response:
                        content = chunk.choices[0].delta.content
                        if content is not None:
                            print(content, end="", flush=True)
                            streamed_text += content
                else:
                
                    try:
                        response = client.chat.completions.create(
                                model = model,
                                messages = chat,
                                max_tokens=max_token, temperature=0.0, top_p=0.8, timeout=900)
                    except RateLimitError as e:

                        error_message = str(e)
                        if "GenerateRequestsPerDayPerProjectPerModel-FreeTier" in error_message:
                            print("Free tier quota exceeded, trying next key...")
                            time.sleep(10)
                            continue 
                        else:
                            print("Non-free-tier rate limit error or other issue.")
                            sys.exit(1)

                    except Exception as e:
                        print(f"Other exception occurred: {e}")
                        sys.exit(1)
            
            if stream:
                generated_text = streamed_text
                generated_token_count = -1
            else:
                print(response)
                generated_text = response.choices[0].message.content
                generated_token_count = response.usage.completion_tokens
            
            _api_call+=1
            
            match = re.match(r"<think>(.*?)</think>(.*)", generated_text, re.DOTALL)
            if match:
                think = match.group(1).strip()
                ans = match.group(2).strip()
            else:
                think = ''
                ans = generated_text

            print(f"{ans=} \n{generated_token_count=}")
            save_data["samples"][i]['think'] =  think
            save_data["samples"][i]['ans'] =  ans
            save_data["samples"][i]['num_token'] =  generated_token_count

            with open(output_file, 'w') as json_file:
                json.dump(save_data, json_file, indent=1)

            time.sleep(5)