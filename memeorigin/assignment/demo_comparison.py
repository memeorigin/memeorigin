"""
Interactive Gen Z Slang Explainer - Comparison Mode
Shows BOTH model output AND dataset truth for educational purposes
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "services", "slang-explainer", "src"))

from inference import generate
from postprocess import parse_definition_example
from datasets import load_dataset
import time

# Load the full dataset
dataset = load_dataset("MLBtrio/genz-slang-dataset")
dataset_dict = {item['Slang'].lower(): item for item in dataset['train']}

def print_header():
    print("\n" + "=" * 70)
    print("       GEN Z SLANG EXPLAINER - Autoregressive Model Demo")
    print("              Comparison Mode: Model vs Dataset")
    print("=" * 70)
    print("\nThis demo shows:")
    print("1. Autoregressive model generation (TinyLlama + LoRA)")
    print("2. Ground truth from dataset (for comparison)")
    print("3. Demonstrates both capabilities and limitations\n")

def explain_term(term):
    """Generate explanation and compare with dataset"""
    print(f"\n{'='*70}")
    print(f"  TERM: {term.upper()}")
    print(f"{'='*70}")

    # Get dataset truth first
    term_lower = term.lower()
    has_dataset = term_lower in dataset_dict

    if has_dataset:
        dataset_entry = dataset_dict[term_lower]
        ground_truth_def = dataset_entry['Description']
        ground_truth_ex = dataset_entry['Example']
    else:
        ground_truth_def = "Not in dataset"
        ground_truth_ex = "N/A"

    # Generate with model
    print("\n AUTOREGRESSIVE MODEL GENERATION...")
    print("    (Generating token by token)\n")

    start_time = time.time()
    raw_output = generate(term, max_new_tokens=100)
    parsed = parse_definition_example(raw_output)
    elapsed = time.time() - start_time

    # Display model output
    print(f"{'─'*70}")
    print("  MODEL OUTPUT (TinyLlama-1.1B + LoRA):")
    print(f"{'─'*70}")

    if parsed["format_ok"] and parsed["definition"]:
        print(f"\n  Definition: {parsed['definition']}")
        print(f"\n  Example: {parsed['example']}")
    else:
        print("\n  [Format Error - Model needs more training]")
        print(f"  Raw: {raw_output[:150]}...")

    # Display dataset truth
    print(f"\n{'─'*70}")
    print("  GROUND TRUTH (Dataset):")
    print(f"{'─'*70}")
    print(f"\n  Definition: {ground_truth_def}")
    print(f"\n  Example: {ground_truth_ex}")

    # Quick accuracy check
    print(f"\n{'─'*70}")
    print("  ANALYSIS:")
    print(f"{'─'*70}")

    if not has_dataset:
        print("  Status: Term not in training dataset")
    elif parsed["format_ok"] and parsed["definition"]:
        # Simple similarity check (case-insensitive substring)
        model_def_lower = parsed['definition'].lower()
        truth_def_lower = ground_truth_def.lower()

        # Check for key word overlap
        model_words = set(model_def_lower.split())
        truth_words = set(truth_def_lower.split())
        common_words = model_words & truth_words
        overlap = len(common_words) / max(len(truth_words), 1)

        if overlap > 0.4:
            print(f"  Accuracy: GOOD - Model captures key concepts")
        elif overlap > 0.1:
            print(f"  Accuracy: PARTIAL - Model has some understanding")
        else:
            print(f"  Accuracy: NEEDS IMPROVEMENT - Different definition")
    else:
        print("  Status: Model output format error")

    print(f"\n  Generation time: {elapsed:.2f} seconds")
    print(f"{'─'*70}\n")

def main():
    print_header()

    print(f"Enter a slang term!\n")

    while True:
        print("─" * 70)
        user_input = input("\nEnter a slang term (or 'quit' to exit): ").strip()

        if not user_input:
            print("Please enter a term!")
            continue

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n" + "=" * 70)
            print("  Thanks for using the Gen Z Slang Explainer!")
            print("  Demonstrating Autoregressive Language Models")
            print("=" * 70 + "\n")
            break

        try:
            explain_term(user_input)
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try another term.")

if __name__ == "__main__":
    print("\nLoading TinyLlama model, LoRA adapters, and dataset...")
    print("This may take a minute...\n")

    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted. Goodbye!\n")
    except Exception as e:
        print(f"\nFatal error: {e}\n")
