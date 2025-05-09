# Turing Machine Bench (TMBench)
This is the official repository for [Turing Machine Evaluation for Large Language Model](https://arxiv.org/pdf/2504.20771?). This repository includes the necessary scripts and instructions for generating data, running LLM simulations, and evaluating model performance.

## Data generate
To generate the data required for Turing Machine evaluation, run the following command:
```
python tag_generate.py --output_path OUTPUT_PATH
```

## LLM simulation
```
python predict.py --model Qwen/Qwen2.5-14B-Instruct --bench_name TMBench --chat
```
- `--model`: The model to use for prediction (e.g., `Qwen/Qwen2.5-14B-Instruct`).
- `--bench_name`: The name of the benchmark (e.g., `TMBench`).
- `--chat`: Option to use chat-based interaction.
## Eval
Evaluate the model's performance with the following command:
```
python acc.py --model Qwen/Qwen2.5-14B-Instruct --bench_name TMBench --chat
```
## Cite
```
@article{wu2025turing,
  title={Turing Machine Evaluation for Large Language Model},
  author={Wu, Haitao and Han, Zongbo and Huang, Huaxi and Zhang, Changqing},
  journal={arXiv preprint arXiv:2504.20771},
  year={2025}
}
```
