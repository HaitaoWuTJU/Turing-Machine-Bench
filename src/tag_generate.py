
import json
import hashlib
import random
import argparse

random.seed(1)

class mTagSystem:
    def __init__(self, initial_string, rules, delete_count=2):
        self.string = initial_string  
        self.rules = rules

        self.delete_count = delete_count
        
        self.step_num = 0
        self.step_results = []
    def step(self):
        if not self.string:
            return False

        current_symbol = self.string[0]
        
        if len(self.string)<self.delete_count:
            return False

        if current_symbol in self.rules:
            add_string = self.rules[current_symbol]
            
            self.string = self.string[self.delete_count:]
            
            self.string += add_string
            self.step_num += 1
            return True
        else:
            return False
    
    def run(self, steps):
        for _ in range(steps):
            self.step_results.append(self.string)
            if not self.step():
                # self.step_results.append("<HALT>")
                break

def generate_id(init_str, rule):
        rule_str = json.dumps(rule, sort_keys=True)
        combined_str = init_str + rule_str
        unique_id = hashlib.sha256(combined_str.encode('utf-8')).hexdigest()
        return unique_id

def random_str(min_length,max_length,symbol_set):
    init_str = ''.join(random.choices(symbol_set, k=random.randint(min_length, max_length)))
    return init_str

def random_rule(min_length,max_length,symbol_set):
    rule = {symbol: ''.join(random.choices(symbol_set, k=random.randint(min_length, max_length))) for symbol in symbol_set}
    return rule

def reject_sampling(sample):
    step_results = sample['step_results']
    for i in range(len(step_results)):
        if i!=0:
            if step_results[i]==step_results[i-1]:
                return False
    return True

def cnt_halt(data,m):
    cnt = 0
    for sample in data:
        step_results = sample['step_results']
        if len(step_results[-1]) < m:
            cnt+=1
    return cnt
    
if __name__=="__main__":

    parser = argparse.ArgumentParser(description='args')
    parser.add_argument('--output_path', type=str, default=f'data/0426.json',help='json output path')
    args = parser.parse_args()

    json_path = args.output_path

    delete_count = 2
    max_step = 31
    num_samples = 100
    
    symbol_set=['A','B','C', 'D', 'E']
    rule_min_length = 1
    rule_max_length = 5
    str_min_length = delete_count
    str_max_length = delete_count+7

    data = {
        'symbol_set': symbol_set,
        'rule_min_length': rule_min_length,
        'rule_max_length': rule_max_length,
        'str_min_length': str_min_length,
        'str_max_length': str_max_length,
        'max_step': max_step,
        'delete_count': delete_count,
        'samples':[]
    }

    seen_ids = []
    while len(data['samples']) < num_samples:
        rule = random_rule(rule_min_length, rule_max_length, symbol_set)
        init_str = random_str(str_min_length, str_max_length,symbol_set)
        
        sample_id = generate_id(init_str, rule)
        if sample_id in seen_ids:
            continue
        seen_ids.append(sample_id)
        
        sample = {
            'id': sample_id,
            'init_str': init_str,
            'rule': rule,
            'delete_count':delete_count
        }
        
        tm = mTagSystem(init_str, rule, delete_count)
        tm.run(max_step)
        sample['step_results'] = tm.step_results

        flag = reject_sampling(sample)
        if flag:
            data['samples'].append(sample)

    print(len(data['samples']))
    print(cnt_halt(data['samples'],delete_count))
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=1)
