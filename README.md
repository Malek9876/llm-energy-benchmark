# LLM Energy Benchmark

This repository contains the official code and analysis scripts for the paper, **"Measurements and optimizations of the energy consumption of AI systems."**

The scripts are organized to reflect the two primary environments used for the experiments: a local workstation for quantization and initial testing, and an HPC cluster for large-scale benchmarking.

---


## Environment and Dependencies

The specific versions of the libraries used in this project (such as PyTorch, vLLM, `autoawq`, `pynvml`, etc.) are detailed in the **Experimental Setup** section of our paper. We recommend setting up a similar environment for reproducing the results.

---

## How These Scripts Were Used to Generate the Paper's Results

The measurements presented in the paper were generated using two distinct workflows.

### 1. Workstation (Hilbert) Workflow

The local workstation was primarily used for model preparation (quantization) and supplementary benchmarking. The workflow consisted of three main steps:

**Step A: Model Quantization**
The `run_autoawq_quantize.py` script was used to create the 4-bit AWQ quantized versions of the baseline models.

*   **Example Usage:**
    ```bash
    python workflows/workstation_hilbert/run_autoawq_quantize.py --model-path "mistralai/Mistral-7B-Instruct-v0.1" --quant-path "./models/Mistral-7B-Instruct-v0.1-AWQ"
    ```

**Step B: Power and Performance Benchmarking**
The `run_power_benchmark.py` script was executed to measure the performance and energy consumption of the quantized models on the workstation's GPU (NVIDIA RTX 5000 Ada). The output was saved as a CSV file in the `analysis/` directory.

**Step C: Model Accuracy Evaluation**
The `run_accuracy.py` script was used to evaluate the accuracy of both the original baseline models and the newly quantized models on the ARC, HellaSwag, and MMLU benchmarks.

### 2. HPC Cluster Workflow

The HPC cluster was used for the main large-scale benchmarks on the datacenter-class GPUs (NVIDIA A100 and H200). The execution was managed by the Slurm workload manager.

The core benchmarking logic is contained in `run_benchmark_vllm_new.py`, while the Slurm scripts in the `slurm_scripts/` directory were used to submit jobs to the cluster.

#### **Configuring and Running Cluster Experiments**

The `.slurm` scripts are templates that were modified for each specific experiment. To reproduce a specific result from the paper, the user would need to edit the script to specify the target model and GPU architecture.

**Example: Modifying `sbatch_base_128.slurm`**

1.  **GPU Resource Request:** The `#SBATCH --partition` and/or `#SBATCH --gres` lines were edited to select the target GPU (e.g., `gpu_a100` or `gpu_h200`).
2.  **Model Argument:** The `--model` argument in the `python` command was set to the Hugging Face identifier for the model being tested.

*   **Example `sbatch_base_128.slurm` configuration:**
    ```slurm
    #!/bin/bash
    #SBATCH --job-name=base_128_benchmark
    #SBATCH --partition=gpu_a100  # <-- This was edited for A100 or H200 runs
    #SBATCH --gres=gpu:1
    #SBATCH --output=base_128_%j.out

    # Activate the project's Conda environment
    source activate your_env_name

    # Run the benchmark script with specific parameters
    python ../run_benchmark_vllm_new.py \
      --model "mistralai/Mistral-7B-Instruct-v0.1" \ # <-- This was edited for different models
      --max-new-tokens 128 \
      --output-file "../../analysis/7b_base_128_a100.csv"
    ```

---

## Results and Analysis

All raw data from the benchmarks was saved as `.csv` files in the `analysis/` directory.

The Jupyter Notebook, `plot_results.ipynb`, located in the same directory, was used to load this data, perform the EDP calculations, and generate the final plots and figures presented in the paper.

---

## Citation

If you use this code in your research, please cite our work:
[Your BibTeX citation will go here once the paper is published]
code
Code
## License

This project is licensed under the MIT License. See the `LICENSE` file for details.