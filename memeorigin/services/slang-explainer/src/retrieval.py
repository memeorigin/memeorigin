import json, os
DATA = os.getenv("SLANG_DATA", os.path.join(os.path.dirname(__file__), "..", "data", "slang_pairs.jsonl"))

def _load_lexicon():
    lex = {}
    try:
        with open(DATA, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): 
                    continue
                r = json.loads(line)
                term = r["term"].strip().lower()
                lex[term] = {"definition": r["definition"].strip(), "example": r["example"].strip()}
    except FileNotFoundError:
        pass
    return lex

_LEX = _load_lexicon()

def lookup(term: str):
    return _LEX.get(term.strip().lower())
