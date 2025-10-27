from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .inference import generate
from .postprocess import parse_definition_example
from .retrieval import lookup

app = FastAPI(title="Slang Explainer")

@app.get("/")
def index():
    return {"service": "slang-explainer", "status": "ok"}

@app.get("/health")
def health():
    return {"ok": True}

class ExplainInput(BaseModel):
    term: str

@app.post("/v1/explain")
def explain(payload: ExplainInput):
    term = payload.term.strip().lower()

    # 1) try LoRA
    raw = generate(term)
    parsed = parse_definition_example(raw)

    # 2) fallback to baseline if needed
    source = "lora"
    if not parsed["format_ok"]:
        base = lookup(term)
        if base:
            parsed["definition"] = parsed["definition"] or base["definition"]
            parsed["example"] = parsed["example"] or base["example"]
            source = "lora+baseline"
        else:
            source = "lora_raw"

    return {
        "term": term,
        "definition": parsed["definition"],
        "example": parsed["example"],
        "raw": raw,     # keep for debugging
        "source": source
    }