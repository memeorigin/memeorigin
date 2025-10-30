import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "adapters", "tinyllama-lora@2025-10-29")

INSTRUCT_TEMPLATE = (
    "Task: Explain the internet slang.\n"
    "Term: {term}\n\n"
    "Definition:"
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float32,             # CPU-friendly
    device_map={"": "cpu"},
    low_cpu_mem_usage=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
base.config.pad_token_id = tokenizer.pad_token_id

model = PeftModel.from_pretrained(base, ADAPTER_DIR, device_map={"": "cpu"})
model.eval()

def generate(term: str, max_new_tokens: int = 100) -> str:
    prompt = INSTRUCT_TEMPLATE.format(term=term.strip())
    inputs = tokenizer(prompt, return_tensors="pt")  # already on CPU
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Use greedy decoding for more deterministic output
            num_beams=1,
            repetition_penalty=1.2,  # Reduce repetition
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)
