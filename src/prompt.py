import json
def generate_prompt(json_path,index=None):

   with open(json_path, 'r') as json_file:
      data = json.load(json_file)

   max_step = data['max_step']-1

   if index:
      samples = data['samples']
      data['samples'] = [samples[i] for i in index]
   
   samples = data['samples']

   prompts_text = []

   for i,sample in enumerate(samples):
      init_str = sample["init_str"]
      rule = sample["rule"]
      step_results = sample["step_results"]
      id = sample["id"]
      m = sample['delete_count']
      
      ALPHABET = '{' + ', '.join(sorted(rule.keys())) + '}'
      RULES = '\n'.join([f"{k} : {' '.join(v)}" for k, v in rule.items()])
      INIT = ' '.join(init_str)

      example = f"""\
Simulate a m-tag system. Your task is to simulate each transition step-by-step and provide the queue's state at each step. Follow the rules and examples closely, and stop upon reaching the halt condition or {max_step} steps. Do not generate additional examples or new problems. No code.

## Rules for Simulation:
1. In each transition, the machine performs the following steps:
   - If the queue length is less than m, halt
   - Read the head symbol of queue
   - Append symbols to the tail based on the head symbol and the corresponding transition rule
   - Delete m symbols from the head of the queue

2. The machine halt if:
   - The queue's length is less than m.

## Example:
m: 2
Alphabet: {{A, B, C}}
Init: [B C A]
Transition rules:
A : C A C
B : A
C : B
Simulation steps:
### step 0:
   - Action: Init
   - Queue State: [B C A]

### step 1:
   - Head Symbol: B
   - Action: Append A to the end of the queue. Remove B C from the head.
   - Queue State: [A A]

### step 2:
   - Head Symbol: A
   - Action: Append C A C to the end of the queue. Remove A A from the head.
   - Queue State: [C A C]

### step 3:
   - Head Symbol: C
   - Action: Append B to the end of the queue. Remove C A from the head.
   - Queue State: [C B]

### step 4:
   - Head Symbol: C
   - Action: Append B to the end of the queue. Remove C B from the head.
   - Queue State: [B] <halt>

---

## The Only Problem to Solve:
m: {m}
Alphabet: {ALPHABET}
Init: [{INIT}]
Transition Rules:
{RULES}
Simulation steps:
"""
      prompts_text.append(example)

      data["samples"][i]["prompts"] = example
    
   return data, prompts_text