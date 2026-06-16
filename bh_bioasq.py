# bh_bioasq.py
import argparse
import json
import random
import sys
from src.data.dataset import BioASQDataset
from src.retrieval.retriever import Retriever
from src.generation.llm_client import OllamaClient
from src.generation.gemini_client import GeminiClient
from src.agents.pipeline import QAPipeline
from src.evaluation.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(
        description="BioASQ QA System — evaluation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python bh_bioasq.py --model phi3:mini --sample-size 50
  uv run python bh_bioasq.py --model mistral:7b-instruct-q4_0 --seed 42 --output experiments/results/mistral.json
  uv run python bh_bioasq.py --model SpeakLeash/bielik-11b-v2.3-instruct:Q4_K_M --sample-size 50
  uv run python bh_bioasq.py --model gemini-2.0-flash --sample-size 50 --output experiments/results/gemini.json
  uv run python bh_bioasq.py --model phi3:mini --question-type yesno --sample-size 20
        """
    )
    parser.add_argument("--model", type=str, help="Model name (Ollama) or 'gemini-X' for Gemini")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--sample-size", type=int, default=50, help="Number of questions (default: 50)")
    parser.add_argument("--output", type=str, default=None, help="Path to save results JSON")
    parser.add_argument("--question-type", type=str, default=None,
                        choices=["yesno", "factoid", "list", "summary"],
                        help="Evaluate only specific question type")
    parser.add_argument("--data", type=str,
                        default="data/raw/BioASQ-training13b/training13b.json",
                        help="Path to BioASQ JSON file")

    # Wyświetl help jeśli brak argumentów
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    return parser.parse_args()


def build_client(model_name: str):
    """Builds LLM client based on model name."""
    if "gemini" in model_name.lower():
        return GeminiClient(model=model_name)
    return OllamaClient(model=model_name)


def main():
    args = parse_args()

    print(f"\n{'='*50}")
    print(f"BioASQ Evaluation Run")
    print(f"{'='*50}")
    print(f"Model:       {args.model}")
    print(f"Seed:        {args.seed}")
    print(f"Sample size: {args.sample_size}")
    print(f"Type filter: {args.question_type or 'all'}")
    print(f"{'='*50}\n")

    # Load data
    random.seed(args.seed)
    ds = BioASQDataset(args.data)
    sample = ds.get_sample(args.sample_size, question_type=args.question_type)

    # Load retriever
    retriever = Retriever()
    retriever.load("data/index")

    # Build pipeline
    llm = build_client(args.model)
    pipeline = QAPipeline(retriever, llm)
    evaluator = Evaluator(pipeline)

    # Run evaluation
    metrics = evaluator.evaluate(sample)

    # Print results
    print(f"\n{'='*50}")
    print("Results:")
    print(f"{'='*50}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "config": {
                    "model": args.model,
                    "seed": args.seed,
                    "sample_size": args.sample_size,
                    "question_type": args.question_type,
                },
                "metrics": metrics,
            }, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()