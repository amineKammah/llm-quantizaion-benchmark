# Quantization Benchmarking

This project provides a setup to benchmark various quantization techniques using the Llama 3.1 8b Instruct model on an NVIDIA A10G GPU. The benchmarks measure the impact of different quantization techniques on the total throughput across scenarios with varying input and output lengths.

## Requirements

- Python 3.x
- NVIDIA A10G GPU (or compatible hardware)
- Installed NVIDIA CUDA and cuDNN

## Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/amineKammah/llm-quantizaion-benchmark.git
   cd llm-quantizaion-benchmark
   ```

2. **Create a Virtual Environment (Optional)**
   ```bash
   python -m venv quantization-benchmark-env
   source quantization-benchmark-env/bin/activate
   ```

3. **Install Required Packages**
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

Execute the benchmark by running the provided script:

```bash
./run_benchmarks.sh
```

This script will:

- Start the server for each quantization method.
- Wait for 5 minutes (`300 seconds`) to ensure the server is ready.
- Run benchmarks for each defined scenario.
- Output the results of the benchmark.

## Customization

- Modify the `run_benchmarks.sh` script to add new models or change scenarios.

## Results

The benchmark results will be saved and can be used to analyze throughput improvements achieved by various quantization techniques.
