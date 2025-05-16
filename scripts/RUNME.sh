#!/bin/bash

#Data Generation
python tag_generate.py --output_path OUTPUT_PATH

#LLM Simulation
bench_name='TMBench'
temperature=0
models=("Qwen/Qwen2.5-Math-72B-Instruct" "Qwen/QVQ-72B-Preview" "meta-llama/Llama-3.3-70B-Instruct"
        "Qwen/Qwen2.5-32B-Instruct" "Qwen/QwQ-32B-Preview" "NovaSky-AI/Sky-T1-32B-Preview" "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        "Qwen/Qwen2.5-14B-Instruct"
        "meta-llama/Llama-3.1-8B-Instruct" "meta-llama/Llama-3.2-3B-Instruct" "meta-llama/Llama-3.2-1B-Instruct" "Qwen/QwQ-32B-Preview")
for model in "${models[@]}"
do
    echo "Running predict.py with model: $model, task: $bench_name, temperature: $temperature"
    python src/predict.py  --model $model --bench_name $bench_name --temperature $temperature --chat
done

#For close-source model:
python src/predict_close_ai.py --model gemini-2.5-pro-exp-03-25 --bench_name TMBench --api google


#LLM Evaluation
for model in "${models[@]}"
do
    echo "Running acc.py with model: $model, task: $bench_name"
    python src/acc.py --model $model --bench_name TMBench --chat
done

#LLM Evaluation for gpqa_diamond, math500, aime2024
for model in "${models[@]}"
do  
    echo "Running gpqa_diamond with model: $model}"
    python src/established_bench/gpqa_diamond.py --model "$model"
    echo "Running math500 with model: $model}"
    python src/established_bench/math500.py --model "$model"

    echo "Running aime2024 with model: $model}"
    python src/established_bench/aime2024.py --model "$model"
done
#Print Results
python src/established_bench/print_result.py --task aime2024
python src/established_bench/print_result.py --task math500
python src/established_bench/print_result.py --task gpqa_diamond