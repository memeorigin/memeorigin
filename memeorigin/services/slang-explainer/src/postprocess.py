import re

def parse_definition_example(text: str):
    defn = re.search(r"Definition:\s*(.+)", text, re.I)
    ex = re.search(r"Example:\s*(.+)", text, re.I)
    return {
        "definition": defn.group(1).strip() if defn else None,
        "example": ex.group(1).strip() if ex else None,
        "format_ok": bool(defn and ex),
    }
