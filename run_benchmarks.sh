#!/bin/bash

# Start the API server for different quantization methods
function start_server {
  model_name=$1
  quant_method=$2
  extra_args=$3

  echo "Starting server for model: $model_name, quantization: $quant_method"

  python -m vllm.entrypoints.openai.api_server \
    --model $model_name \
    $extra_args \
    --max-model-len 9500 \
    --max-num-seqs=100 \
    --quantization=$quant_method &

  # Wait for the server to be up
  echo "Hey, I am waiting for 5 minutes for the server to start..."
  sleep 300  # 5 minutes wait time
}

# Benchmark function to run the tests based on given input and output lengths
function run_benchmark {
  input_len=$1
  output_len=$2
  model_name=$3
  extra_args=$4

  echo "Running benchmark for model: $model_name, input length: $input_len, output length: $output_len"

  python benchmark_serving.py \
    --backend openai \
    --base-url http://127.0.0.1:8000 \
    --model $model_name \
    $extra_args \
    --max-concurrency 42 \
    --num-prompts 100 \
    --dataset-name random \
    --random-input-len=$input_len \
    --random-output-len=$output_len \
    --seed=12 \
    --save-result
}

# Define the models and quantization setups
declare -a quant_models=(
  "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit:bitsandbytes"
  "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4:gptq_marlin"
  "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4:awq_int4"
  "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8:w8a8"
  "meta-llama/Llama-3.1-8B-Instruct:fp8"
  "meta-llama/Llama-3.1-8B-Instruct:full"
)

# Define scenarios
declare -a scenarios=(
  "300,1500"
  "3000,300"
  "8000,300"
)

# Loop through each model and scenario
for model_quant in "${quant_models[@]}"; do
  IFS=":" read -r model quant <<< "$model_quant"

  start_server $model $quant

  for scenario in "${scenarios[@]}"; do
    IFS="," read -r input_len output_len <<< "$scenario"
    run_benchmark $input_len $output_len $model
  done

  # Stop any running server after benchmarks
  pkill -f "vllm.entrypoints.openai.api_server"
done
