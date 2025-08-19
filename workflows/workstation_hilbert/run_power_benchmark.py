import argparse
import time
import os
import pandas as pd
import torch
from pathlib import Path
from vllm import LLM, SamplingParams
import pynvml

def init_pynvml_safe():
    """Initializes pynvml and returns True on success, False on failure."""
    try:
        pynvml.nvmlInit()
        return True
    except Exception as e:
        print(f"WARNING: pynvml failed to initialize: {e}. Power measurement will be disabled.")
        return False

def llm_inference_workload(llm, sampling_params, prompts):
    """Runs a standard inference workload and returns performance metrics."""
    print("  Workload: Warm-up...")
    _ = llm.generate(prompts[:1], sampling_params, use_tqdm=False)
    torch.cuda.synchronize()
    print("  Workload: Warmup complete.")

    start_time = time.monotonic()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    torch.cuda.synchronize()
    duration = time.monotonic() - start_time
    
    total_new_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_new_tokens / duration if duration > 0 else 0
    return {"throughput_tokens_per_sec": throughput, "duration_sec": duration}

def main(args):
    pynvml_ok = init_pynvml_safe()
    
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    csv_output_path = output_dir / args.output_csv
    
    print(f"\n--- Loading Model: {args.model_id_or_path} ---")
    print(f"--- Quantization Method: {args.quant_method} ---")
    llm = LLM(model=args.model_id_or_path, quantization=args.quant_method, trust_remote_code=True)
    print("--- vLLM Engine Initialized ---")

    sampling_params = SamplingParams(n=1, temperature=0.0, max_tokens=args.max_new_tokens)
    prompts = [args.prompt] * args.batch_size
    all_run_metrics = []

    handle = pynvml.nvmlDeviceGetHandleByIndex(0) if pynvml_ok else None

    for run_num in range(1, args.num_runs + 1):
        print(f"\n--- Starting Benchmark Run {run_num}/{args.num_runs} ---")
        energy_before = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) if pynvml_ok else 0
        perf_metrics = llm_inference_workload(llm, sampling_params, prompts)
        energy_after = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) if pynvml_ok else 0
        
        energy_j = (energy_after - energy_before) / 1000.0 if pynvml_ok else None
        power_w = energy_j / perf_metrics["duration_sec"] if pynvml_ok and perf_metrics["duration_sec"] > 0 else None
        
        run_data = {**perf_metrics, "total_energy_j": energy_j, "avg_power_w": power_w}
        all_run_metrics.append(run_data)
        print(f"  > Run {run_num} finished. Throughput: {perf_metrics['throughput_tokens_per_sec']:.2f} tokens/sec")
        time.sleep(2)

    print("\n--- Aggregating results... ---")
    results_df = pd.DataFrame(all_run_metrics)
    mean_stats, std_stats = results_df.mean(), results_df.std()
    
    final_summary = {"technique": f"vLLM-{args.quant_method}", "batch_size": args.batch_size}
    for metric in results_df.columns:
        final_summary[f"{metric}_mean"] = mean_stats[metric]
        final_summary[f"{metric}_std"] = std_stats[metric]
    
    summary_df = pd.DataFrame([final_summary])
    summary_df.to_csv(csv_output_path, index=False, header=not os.path.exists(csv_output_path), mode='a')
    
    print("\n--- Final Aggregated Results ---")
    print(summary_df.to_string())
    print(f"\n✅ Final summary saved to {csv_output_path}")
    if pynvml_ok:
        pynvml.nvmlShutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Power and Performance Benchmark for Quantized Models")
    parser.add_argument("--model_id_or_path", type=str, required=True)
    parser.add_argument("--quant_method", type=str, required=True, choices=['awq', 'gptq', 'awq_marlin'])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_runs", type=int, default=5, help="Number of runs for statistical averaging.")
    parser.add_argument("--prompt", type=str, default="A detailed story about a robot who discovers music.")
    parser.add_argument("--output_dir", type=str, default="results/power_benchmarks")
    parser.add_argument("--output_csv", type=str, default="power_summary.csv")
    args = parser.parse_args()
    main(args)
