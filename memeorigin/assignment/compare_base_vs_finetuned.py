"""
Base Model vs Fine-Tuned Model Comparison
Shows if LoRA training actually improved the model or if TinyLlama already knew the answers
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "services", "slang-explainer", "src"))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from postprocess import parse_definition_example
import time

# Configuration
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "..", "services", "slang-explainer", "models", "adapters", "tinyllama-lora@2025-10-29")

INSTRUCT_TEMPLATE = (
    "Task: Explain the internet slang.\n"
    "Term: {term}\n\n"
    "Definition:"
)

print("\n" + "="*80)
print("  BASE MODEL vs FINE-TUNED MODEL COMPARISON")
print("  Testing if LoRA training actually worked")
print("="*80)

print("\nLoading models (this may take a minute)...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load base model (NO adapters)
print("  [1/2] Loading base TinyLlama model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float32,
    device_map={"": "cpu"},
    low_cpu_mem_usage=True,
)
base_model.config.pad_token_id = tokenizer.pad_token_id
base_model.eval()

# Load fine-tuned model (WITH adapters)
print("  [2/2] Loading fine-tuned model with LoRA adapters...")
finetuned_base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float32,
    device_map={"": "cpu"},
    low_cpu_mem_usage=True,
)
finetuned_base.config.pad_token_id = tokenizer.pad_token_id
finetuned_model = PeftModel.from_pretrained(finetuned_base, ADAPTER_DIR, device_map={"": "cpu"})
finetuned_model.eval()

print("\n✓ Both models loaded!\n")


def generate_with_model(model, term: str, max_new_tokens: int = 100) -> str:
    """Generate explanation using a specific model"""
    prompt = INSTRUCT_TEMPLATE.format(term=term.strip())
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding for deterministic results
            num_beams=1,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)


def compare_term(term: str):
    """Compare base model vs fine-tuned model on a term"""
    print(f"\n{'='*80}")
    print(f"  TERM: {term.upper()}")
    print(f"{'='*80}")

    # Generate with BASE model
    print("\n  [BASE MODEL] Generating...")
    start_time = time.time()
    base_output = generate_with_model(base_model, term)
    base_time = time.time() - start_time
    base_parsed = parse_definition_example(base_output)

    # Generate with FINE-TUNED model
    print("  [FINE-TUNED MODEL] Generating...")
    start_time = time.time()
    ft_output = generate_with_model(finetuned_model, term)
    ft_time = time.time() - start_time
    ft_parsed = parse_definition_example(ft_output)

    # Display results side by side
    print(f"\n{'─'*80}")
    print("  BASE MODEL OUTPUT (No training):")
    print(f"{'─'*80}")

    if base_parsed["format_ok"] and base_parsed["definition"]:
        print(f"\n  Definition: {base_parsed['definition']}")
        print(f"  Example: {base_parsed['example']}")
    else:
        print("\n  [Format Error or Empty Response]")
        print(f"  Raw output: {base_output[:200]}")

    print(f"\n  Generation time: {base_time:.2f}s")

    print(f"\n{'─'*80}")
    print("  FINE-TUNED MODEL OUTPUT (With LoRA adapters):")
    print(f"{'─'*80}")

    if ft_parsed["format_ok"] and ft_parsed["definition"]:
        print(f"\n  Definition: {ft_parsed['definition']}")
        print(f"  Example: {ft_parsed['example']}")
    else:
        print("\n  [Format Error or Empty Response]")
        print(f"  Raw output: {ft_output[:200]}")

    print(f"\n  Generation time: {ft_time:.2f}s")

    # Analysis
    print(f"\n{'─'*80}")
    print("  ANALYSIS:")
    print(f"{'─'*80}")

    base_ok = base_parsed["format_ok"] and base_parsed["definition"]
    ft_ok = ft_parsed["format_ok"] and ft_parsed["definition"]

    if not base_ok and ft_ok:
        print("  ✓ TRAINING WORKED! Fine-tuned model outputs correct format.")
        print("    Base model failed, fine-tuned model succeeded.")
    elif base_ok and ft_ok:
        # Compare quality
        if base_parsed["definition"] == ft_parsed["definition"]:
            print("   IDENTICAL OUTPUT - Models produce same result.")
            print("    This term may have been in pre-training data.")
        else:
            print("  ✓ DIFFERENT OUTPUTS - Fine-tuning changed the model!")
            print("    Compare definitions above to see which is better.")
    elif base_ok and not ft_ok:
        print("  ✗ REGRESSION - Base model worked but fine-tuned didn't.")
        print("    Training may have hurt performance on this term.")
    else:
        print("  ✗ BOTH FAILED - Neither model produced valid output.")
        print("    This term may be very obscure or formatted oddly.")

    print(f"{'─'*80}\n")


def main():
    print(f"{'='*80}")
    print("  TEST SUGGESTED TERMS")
    print(f"{'='*80}")
    print("\n  Good test terms:")
    # print("  • Recent slang (2023-2024): rizz, mid, bussin, no cap, slaps")
    # print("  • Common slang: bet, fr, w, L, lowkey")
    print("  • Obscure terms: gyatt, based, ratio, cheugy")
    print(f"\n{'='*80}\n")

    while True:
        user_input = input("\nEnter a slang term to compare (or 'quit' to exit): ").strip()

        if not user_input:
            print("Please enter a term!")
            continue

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n" + "="*80)
            print("  Comparison complete!")
            print("="*80 + "\n")
            break

        try:
            compare_term(user_input)
        except Exception as e:
            print(f"\n✗ Error: {e}")
            print("Please try another term.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nComparison interrupted. Goodbye!\n")
    except Exception as e:
        print(f"\n✗ Fatal error: {e}\n")
