import json,re
from tqdm import tqdm
import argparse
import glob

seed = 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model with main.py")
    parser.add_argument("--task", type=str, default='math500', help="Benchmark")
    args = parser.parse_args()

    task = args.task
    files= glob.glob(f"./results/{task}/*_cot_chat.json")
    
    def extract_model_size(model_name):
        match = re.search(r'(\d+)(B|b)', model_name)
        if match:
            return int(match.group(1))
        else:
            return float('inf')
        
    models = [file.rsplit('/',1)[-1].split('_cot_chat.json',1)[0]  for file in files]

    models.sort(key=extract_model_size, reverse=True)

    print(f"\n------task: {task}------\n")
    for model in models:
        file = f'./results/{task}/{model}_cot_chat.json'
        with open(file, 'r') as f:
            save_data = json.load(f)

        correct = 0
        total = len(save_data)
        for i in range(len(save_data)):
            i = str(i)
            sample = save_data[i]
            
            response =  sample['ans']
            gold = sample['gold']
            pred = sample['pred']
            acc = sample['acc']
            correct += acc

        accuracy = correct/len(save_data)
        print(f"Model: {model}, Accuracy: {accuracy:.1%} {correct=} {total=}")
