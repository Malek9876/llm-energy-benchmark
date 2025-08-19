# filename: run_final_accuracy.py
import argparse
import os
import sys
import json
import pandas as pd

# This uses the standard Hugging Face model loader from lm-eval.
# It is capable of automatically using libraries like vLLM if they are installed.
try:
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
except ImportError:
    print("FATAL: lm-eval or transformers library is not installed correctly.")
    sys.exit(1)


def main(args):
    """
    Runs lm-eval accuracy benchmarks using the standard HFLM adapter.
    This works for both baseline and quantized models when the correct
    backend libraries (like vllm) are installed in the environment.
    """
    print(f"--- Starting Standard Accuracy Evaluation ---")
    print(f"Model ID/Path: {args.model_id_or_path}")
    print(f"Tasks: {args.tasks}")

    # The HFLM adapter is smart. It will pass quantization arguments
    # to the underlying AutoModelForCausalLM.from_pretrained call.
    eval_adapter = HFLM(
        pretrained=args.model_id_or_path,
        trust_remote_code=True,
        dtype="auto",
        device="cuda",
        # We pass the quantization argument here for vLLM to pick up.
        quantization=args.quant_method
    )

    eval_results = evaluator.simple_evaluate(
        model=eval_adapter,
        tasks=args.tasks.split(','),
        batch_size="auto"
    )

    print("\n--- Accuracy Results ---")
    print(json.dumps(eval_results['results'], indent=2))

    technique = "Baseline" if not args.quant_method else args.quant_method.upper()
    accuracy_scores = {"model_id": args.model_id_or_path, "technique": technique}

    for task, result in eval_results['results'].items():
        for metric, value in result.items():
            if isinstance(value, float):
                accuracy_scores[f"{task}_{metric.split(',')[0]}"] = value

    df = pd.DataFrame([accuracy_scores])
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, args.output_csv)
    df.to_csv(csv_path, index=False, header=not os.path.exists(csv_path), mode='a')
    print(f"\nAccuracy scores appended to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run lm-eval using the standard HFLM adapter.")
    parser.add_argument("--model_id_or_path", type=str, required=True)
    # This argument is now passed to the HFLM adapter, which is a new feature
    parser.add_argument("--quant_method", type=str, default=None, choices=['awq', 'gptq'])
    parser.add_argument("--tasks", type=str, default="arc_challenge,hellaswag,truthfulqa_mc2")
    parser.add_argument("--output_dir", type=str, default="results/accuracy")
    parser.add_argument("--output_csv", type=str, default="accuracy_summary.csv")
    args = parser.parse_args()
    main(args)
