from fastapi import FastAPI
from pydantic import BaseModel

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
    return {
        "term": payload.term,
        "definition": "Temporary placeholder definition.",
        "example": "This is where your generated example will go."
    }