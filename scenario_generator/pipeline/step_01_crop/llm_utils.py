import json
from typing import Any, Dict, Optional


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Balanced-brace scan to extract a top-level JSON object from model output."""
    # Fast path
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = None
    depth = 0
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    snippet = text[start: i + 1]
                    try:
                        obj = json.loads(snippet)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        start = None
                        continue
    return None
