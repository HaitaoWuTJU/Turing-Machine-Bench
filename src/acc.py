import re, json
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model with main.py")
    parser.add_argument("--model", type=str, default='Qwen2.5-14B-Instruct', help="Model name or path to use")
    parser.add_argument("--bench_name", type=str,default="0407")
    parser.add_argument("--chat", action="store_true", default=False)

    args = parser.parse_args()
    model = args.model
    chat = args.chat
    bench_name = args.bench_name

    if chat:
        path = f'results/{bench_name}_{model}_chat.json'
    else:
        path = f'results/{bench_name}_{model}_completions.json'

    with open(path, 'r') as json_file:
        data = json.load(json_file)

    i=0
    samples = data.get("samples", [])
    # count_with_ans = sum(1 for sample in samples if "ans" in sample)
    # samples = samples[:count_with_ans]
    # print(f"{count_with_ans=}")
    idx = [i for i, sample in enumerate(samples) if "ans" in sample]
    samples = [samples[i] for i in idx]


    def default_count():
        return {'correct': 0, 'incorrect': 0}
    cnt_results = defaultdict(default_count)

    pass_sample=0
    for id,sample in enumerate(samples):
        step_results = sample['step_results']
        ans = sample['ans']
        num_token = sample["num_token"]
        # print(f"{ans=}")

        pattern = r"(?i)step (\d+):.*?Queue State:.*?\[([ABCDE12345αβγδε@#$%&* ,]*)\]"

        matches = re.findall(pattern, ans, re.DOTALL)
        
        steps_ans = {}
        _max = -1
        for step, content in matches:
            step = int(step)
            _max = max(_max,step)
            # if step not in steps_ans:
            content = content.replace(" ", "").replace(",","").replace(" ","")
            steps_ans[step] = content
                # print(content)
        # print(f"{_max=}")
        step_maxs = len(step_results)

        max_correct = -1

        pass_flag = True

        for i in range(step_maxs):
            # if i==1:
            #     print(f"{step_gt=} {step_ans=}")
            step_gt = step_results[i]
            step_ans = steps_ans.get(i,None)

            if step_gt == step_ans:
                cnt_results[i]["correct"] += 1
                max_correct=i
                # if i==_max:
                #     print(f"step {i} is right.")
            else:
                if pass_flag:
                    print(f"step {i} is wrong. {step_gt=} {step_ans=}")

                pass_flag=False
                cnt_results[i]["incorrect"] += 1
                
                # if i==_max:
                #     print(f"sample {id} num_token {num_token} step {i} is wrong. {step_gt=} {step_ans=}")
        if pass_flag:
                pass_sample+=1
        # print(f"{id=} {max_correct=}, gt={step_results[max_correct+1]}, ans={steps_ans.get(max_correct+1,None)}")
        print(f"{id=} {_max=} {max_correct=}\n")

    max_steps = 31
    x_values = range(max_steps)
    # y_values = [cnt_results[i]['correct'] / (cnt_results[i]['incorrect']+cnt_results[i]['correct']) if (cnt_results[i]['incorrect']+cnt_results[i]['correct']) != 0 else 0 for i in range(max_steps)]
    y_values = [cnt_results[step]["correct"]/(cnt_results[step]["correct"]+cnt_results[step]["incorrect"]) for step in range(max_steps)]
    uni_acc = sum([y_values[step]*(1) for step in range(max_steps)]) *100/(sum([(1) for step in range(max_steps)]))
    lin_acc = sum([y_values[step]*(step+1) for step in range(max_steps)]) *100/(sum([(step+1) for step in range(max_steps)]))
    # min_index = y_values.index(0)

    # print(y_values[0], y_values[1],y_values[2], y_values[-2],y_values[-1])
    
    print([cnt_results[step]['correct'] for step in range(max_steps)])
    print([cnt_results[step]['correct']+cnt_results[step]["incorrect"] for step in range(max_steps)])
    # print([cnt_results[i]['incorrect']+cnt_results[i]['correct'] for i in range(max_steps)])
    print(len(y_values))
    
    print(f"{model}: {uni_acc=:.1f}, {lin_acc=:.1f} {pass_sample=}")
    print(f"{len(samples)}")
    # print(y_values)