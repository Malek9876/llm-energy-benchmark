import argparse
import time
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from vllm import LLM, SamplingParams

# =======================================================
# ========== HELPER FUNCTIONS (DEFINED FIRST) ===========
# =======================================================
PYNVML_INITIALIZED_SUCCESSFULLY = False
def init_pynvml_safe():
    global PYNVML_INITIALIZED_SUCCESSFULLY
    try:
        import pynvml
        if PYNVML_INITIALIZED_SUCCESSFULLY: return True
        pynvml.nvmlInit()
        PYNVML_INITIALIZED_SUCCESSFULLY = True
        return True
    except Exception:
        print("pynvml not found or failed to initialize. Energy measurement disabled.")
        return False

def shutdown_pynvml_safe():
    global PYNVML_INITIALIZED_SUCCESSFULLY
    if PYNVML_INITIALIZED_SUCCESSFULLY:
        try:
            import pynvml; pynvml.nvmlShutdown()
            PYNVML_INITIALIZED_SUCCESSFULLY = False
        except Exception: pass

def get_gpu_total_energy_mj(gpu_logical_idx):
    if not PYNVML_INITIALIZED_SUCCESSFULLY: return None
    try:
        import pynvml; handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_logical_idx)
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    except Exception: return None

# =======================================================
# ========== SIMPLIFIED WORKLOAD FUNCTION ==============
# =======================================================
def llm_inference_workload(llm: LLM, sampling_params: SamplingParams, prompts: list[str]) -> dict:
    """
    Runs the vLLM workload and returns the core performance metrics.
    """
    print("  Workload: Warm-up...")
    _ = llm.generate(prompts[:1], sampling_params, use_tqdm=False)
    torch.cuda.synchronize()
    print("  Workload: Warmup complete.")

    start_time = time.monotonic()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    torch.cuda.synchronize()
    duration = time.monotonic() - start_time
    
    total_new_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput_tokens_per_sec = total_new_tokens / duration if duration > 0 else 0
    avg_latency_per_request_sec = duration / len(prompts) if prompts else 0

    return {
        "throughput_tokens_per_sec": throughput_tokens_per_sec,
        "avg_latency_per_request_sec": avg_latency_per_request_sec,
        "duration_sec": duration,
    }

# =======================================================
# ========== MAIN FUNCTION (REVISED FOR FINAL SUMMARY) ==
# =======================================================
def main(args):
    """
    Runs a vLLM benchmark multiple times, calculates the mean and standard
    deviation, and saves a single summary row to a CSV file.
    """
    if args.measure_energy and not init_pynvml_safe():
        args.measure_energy = False
    
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    csv_output_path = output_dir / args.output_csv
    
    num_gpus_to_use = args.tensor_parallel_size
    if torch.cuda.device_count() < num_gpus_to_use:
        print(f"Error: TP={num_gpus_to_use} but only {torch.cuda.device_count()} GPUs available.")
        return

    print(f"\n--- Loading Model: {args.model_id_or_path} ---")
    llm = LLM(
        model=args.model_id_or_path, trust_remote_code=True, gpu_memory_utilization=0.95,
        quantization=args.quant_method, tokenizer_mode="auto", tensor_parallel_size=args.tensor_parallel_size
    )
    print("--- vLLM Engine Initialized ---")

    sampling_params = SamplingParams(n=1, temperature=0.0, max_tokens=args.max_new_tokens_per_run)
    prompts = [args.prompt] * args.batch_size

    # This list will store the results of each individual run
    all_run_metrics = []

    for run_num in range(1, args.num_runs + 1):
        print(f"\n--- Starting Benchmark Run {run_num}/{args.num_runs}: Batch Size={args.batch_size} ---")

        energy_before_mj = {i: get_gpu_total_energy_mj(i) for i in range(num_gpus_to_use)} if args.measure_energy else {}
        perf_metrics = llm_inference_workload(llm, sampling_params, prompts)
        energy_after_mj = {i: get_gpu_total_energy_mj(i) for i in range(num_gpus_to_use)} if args.measure_energy else {}
        
        # Calculate metrics for THIS run
        total_duration = perf_metrics["duration_sec"]
        energy_j_total = sum((energy_after_mj[i] - energy_before_mj.get(i, 0)) / 1000.0 for i in range(num_gpus_to_use)) if args.measure_energy and all(v is not None for v in energy_before_mj.values()) else None
        power_w_total = energy_j_total / total_duration if total_duration > 0 and energy_j_total is not None else None
        
        # Add this run's results to our list
        run_data = {**perf_metrics, "total_energy_j": energy_j_total, "avg_power_w": power_w_total}
        all_run_metrics.append(run_data)
        
        print(f"  > Run {run_num} finished. Throughput: {perf_metrics['throughput_tokens_per_sec']:.2f} tokens/sec")
        time.sleep(2)

    print("\n--- All runs complete. Aggregating results... ---")

    # Convert the list of results into a Pandas DataFrame
    results_df = pd.DataFrame(all_run_metrics)
    
    # Calculate mean and standard deviation for all numeric columns
    mean_stats = results_df.mean()
    std_stats = results_df.std()

    # Prepare the final, single-row dictionary for the summary CSV
    technique = f"vLLM-{args.quant_method or 'Baseline'}-TP{args.tensor_parallel_size}"
    final_summary = {
        "technique": technique,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens_per_run,
        "num_runs": args.num_runs
    }
    
    # Add the mean and std of each metric to the summary dictionary
    for metric in results_df.columns:
        final_summary[f"{metric}_mean"] = mean_stats[metric]
        final_summary[f"{metric}_std"] = std_stats[metric]
    
    # Create the final single-row DataFrame and save it
    summary_df = pd.DataFrame([final_summary])
    summary_df.to_csv(csv_output_path, index=False, header=not os.path.exists(csv_output_path), mode='a')
    
    print("\n--- Final Aggregated Results ---")
    # Pretty-print the results to the console
    for key, value in final_summary.items():
        if isinstance(value, float):
            print(f"{key:<30}: {value:.3f}")
        else:
            print(f"{key:<30}: {value}")

    print(f"\n✅✅✅ Final summary saved to {csv_output_path}")
    shutdown_pynvml_safe()

# =======================================================
# ========== SCRIPT ENTRYPOINT ==========================
# =======================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run and summarize a rigorous performance benchmark with vLLM.")
    
    parser.add_argument("--model_id_or_path", type=str, required=True)
    parser.add_argument("--quant_method", type=str, default=None, choices=['awq', 'gptq', 'awq_marlin'])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens_per_run", type=int, default=512)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--prompt", type=str, default="What is the story of the Three-Body Problem?")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of times to repeat each benchmark for statistical significance.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True, help="Filename for the FINAL SUMMARY CSV file.")
    parser.add_argument("--measure_energy", action="store_true", help="Enable GPU energy measurement.")
    
    args = parser.parse_args()
    main(args)
