#!/bin/bash
# ==============================================================================
# --- Slurm Job Directives for COMPREHENSIVE Baseline Accuracy Benchmark ---
# ==============================================================================
#SBATCH --job-name=baseline_accuracy_full
#SBATCH --output=logs/baseline_accuracy_full_output_%x_%j.txt
#SBATCH --error=logs/baseline_accuracy_full_error_%x_%j.txt
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# ==============================================================================
# --- JOB CONFIGURATION ---
# ==============================================================================
BASELINE_MODEL="mistralai/Mistral-Small-24B-Instruct-2501"

# --- The Comprehensive Evaluation Suite ---
# This suite provides a robust, paper-worthy evaluation of the model's capabilities.
# - arc_challenge: Science & Reasoning
# - hellaswag: Common-sense Inference
# - truthfulqa_mc2: Tendency to avoid generating false information
# - winogrande: Subtle bias and pronoun resolution
# - mmlu: Massive multi-task knowledge across 57 subjects (very slow)
EVAL_TASKS="arc_challenge,hellaswag,truthfulqa_mc2,winogrande,mmlu"

# --- Script and Environment Configuration ---
PYTHON_SCRIPT="run_baseline_accuracy.py"
CONDA_ENV_NAME="baseline_accuracy_env"
OUTPUT_DIR="experiment_outputs/baseline_accuracy_full_${SLURM_JOB_ID}"
RESULTS_CSV="baseline_accuracy_full_summary.csv"
PYTHON_EXECUTABLE="$HOME/.conda/envs/${CONDA_ENV_NAME}/bin/python"

# ==============================================================================
# --- Job Setup & Execution ---
# ==============================================================================
echo "========================================================"
echo "Starting COMPREHENSIVE Baseline Accuracy Benchmark"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Model: $BASELINE_MODEL"
echo "Tasks: $EVAL_TASKS"
echo "========================================================"

# --- 1. Setup ---
mkdir -p logs "$OUTPUT_DIR"
export HF_HOME="/home/2024004/PARTAGE/huggingface_cache"
export HUGGING_FACE_HUB_TOKEN="input"
export HF_DATASETS_OFFLINE=1
# Activate the correct conda environment
echo -e "\n--> Activating conda environment: $CONDA_ENV_NAME"
source /soft/AIDL/conda/2410/etc/profile.d/conda.sh
conda activate "$CONDA_ENV_NAME"
echo "--> Environment activated."

# --- 2. Run the Full Evaluation ---
echo -e "\n--- Evaluating BASELINE model on the full suite... ---"
# We call the python script with the expanded list of tasks
"$PYTHON_EXECUTABLE" "$PYTHON_SCRIPT" \
    --model_id_or_path "$BASELINE_MODEL" \
    --tasks "$EVAL_TASKS" \
    --output_dir "$OUTPUT_DIR" \
    --output_csv "$RESULTS_CSV"

echo -e "\n========================================================"
echo "✅ Comprehensive baseline accuracy evaluation finished."
echo "Results saved to: ${OUTPUT_DIR}/${RESULTS_CSV}"
echo "========================================================"
