# filename: run_baseline_accuracy.py
import argparse
import os
import sys
import json
import pandas as pd

# This uses the standard Hugging Face model loader from lm-eval
try:
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
except ImportError:
    print("FATAL: lm-eval or transformers library is not installed correctly.")
    sys.exit(1)


def main(args):
    """
    Runs lm-eval accuracy benchmarks for standard (baseline) Hugging Face models.
    """
    print(f"--- Starting Standard Accuracy Evaluation ---")
    print(f"Model ID/Path: {args.model_id_or_path}")
    print(f"Tasks: {args.tasks}")

    # Use the standard Hugging Face adapter from lm-eval.
    # It is very reliable for baseline models.
    eval_adapter = HFLM(
        pretrained=args.model_id_or_path,
        trust_remote_code=True,
        dtype="auto", # Automatically selects float16 or bfloat16
        device="cuda" # Ensure it runs on the GPU
    )

    # Run the evaluation
    eval_results = evaluator.simple_evaluate(
        model=eval_adapter,
        tasks=args.tasks.split(','),
        batch_size="auto"
    )

    print("\n--- Accuracy Results ---")
    print(json.dumps(eval_results['results'], indent=2))

    # Save results to a CSV file
    accuracy_scores = {
        "model_id": args.model_id_or_path,
        "technique": "Baseline"
    }

    for task, result in eval_results['results'].items():
        for metric, value in result.items():
            if isinstance(value, float):
                clean_metric = metric.split(',')[0]
                accuracy_scores[f"{task}_{clean_metric}"] = value

    df = pd.DataFrame([accuracy_scores])
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, args.output_csv)
    df.to_csv(csv_path, index=False, header=not os.path.exists(csv_path), mode='a')
    print(f"\nAccuracy scores appended to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run lm-eval accuracy for baseline models.")
    parser.add_argument("--model_id_or_path", type=str, required=True)
    parser.add_argument("--tasks", type=str, default="arc_challenge,hellaswag,truthfulqa_mc2")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="baseline_accuracy_results.csv")
    args = parser.parse_args()
    main(args)
