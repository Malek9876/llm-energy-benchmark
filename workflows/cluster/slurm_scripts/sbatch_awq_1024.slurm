#!/bin/bash
#SBATCH --job-name=awq_1024
#SBATCH --output=logs/awq_1024_output_%x_%j.txt
#SBATCH --ntasks=1
#SBATCH --time=01:30:00
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# --- Config ---
MODEL="TheBloke/Mistral-7B-Instruct-v0.1-AWQ"
OUT_DIR="experiment_outputs/awq_1024"
CSV_FILE="results_awq_1024.csv"
BATCH_SIZES=(1 2 4 8 16 32)

# --- Setup & Run ---
mkdir -p logs "$OUT_DIR"
export HUGGING_FACE_HUB_TOKEN="YOUR_HUGGING_FACE_TOKEN_HERE"
export HF_HOME="/home/2024004/PARTAGE/huggingface_cache"
source /soft/AIDL/conda/2410/etc/profile.d/conda.sh
conda activate vllm_env

echo "--- Benchmarking AWQ+Marlin for 1024 tokens across multiple batch sizes ---"

for BATCH in "${BATCH_SIZES[@]}"; do
    echo -e "\n--- Testing Batch Size: $BATCH ---"
    python run_benchmark_vllm_new.py \
        --model_id_or_path "$MODEL" \
        --quant_method "awq_marlin" \
        --batch_size $BATCH \
        --max_new_tokens 1024 \
        --output_dir "$OUT_DIR" \
        --output_csv "$CSV_FILE" \
        --measure_energy
done

echo "Benchmark finished for AWQ - 1024 tokens."
