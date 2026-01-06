import json
import re
from typing import Any, Dict, List, Optional


def _extract_all_json_objects(text: str) -> List[Any]:
    """
    Extract JSON values (dict OR list) from arbitrary text.

    We collect multiple candidates because LLMs often wrap JSON in prose,
    code-fences, or output multiple JSON blobs. We later pick the most
    plausible one (usually the last one).
    """
    vals: List[Any] = []

    # 1) Direct parse (whole text is JSON)
    try:
        obj = json.loads(text)
        if isinstance(obj, (dict, list)):
            vals.append(obj)
    except Exception:
        pass

    # 2) Code-fenced blocks ```json ... ``` or ``` ... ```
    for m in re.finditer(r"```(?:json)?\s*\n([\s\S]*?)```", text, flags=re.IGNORECASE):
        block = m.group(1).strip()
        if not block:
            continue
        try:
            obj = json.loads(block)
            if isinstance(obj, (dict, list)):
                vals.append(obj)
        except Exception:
            continue

    # Helper: balanced scan for a given opener/closer
    def _balanced_scan(opener: str, closer: str) -> None:
        n = len(text)
        start_search = 0
        while True:
            start = text.find(opener, start_search)
            if start < 0:
                break
            depth = 0
            in_str = False
            esc = False
            i = start
            while i < n:
                ch = text[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    i += 1
                    continue
                else:
                    if ch == '"':
                        in_str = True
                        i += 1
                        continue
                    if ch == opener:
                        depth += 1
                    elif ch == closer:
                        depth -= 1
                        if depth == 0:
                            snippet = text[start:i + 1]
                            try:
                                obj = json.loads(snippet)
                                if isinstance(obj, (dict, list)):
                                    vals.append(obj)
                            except Exception:
                                pass
                            break
                    i += 1
            start_search = start + 1

    # 3) Balanced dict scan {...}
    _balanced_scan("{", "}")
    # 4) Balanced list scan [...]
    _balanced_scan("[", "]")

    return vals


def _pick_last_matching(objs: List[Dict[str, Any]], required_top_keys: List[str]) -> Optional[Dict[str, Any]]:
    for obj in reversed(objs):
        ok = True
        for k in required_top_keys:
            if k not in obj:
                ok = False
                break
        if ok:
            return obj
    return None


def _find_key_recursive(obj: Any, key: str) -> Optional[Any]:
    """Depth-first search for the first occurrence of a key in nested dict/list structures."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            got = _find_key_recursive(v, key)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for it in obj:
            got = _find_key_recursive(it, key)
            if got is not None:
                return got
    return None


def parse_llm_json(text: str, required_top_keys: List[str]) -> Dict[str, Any]:
    """
    Best-effort parse of LLM output into a JSON dict.

    Handles common failure modes:
    - top-level list (e.g., the model outputs just an array of actors)
    - required key nested (e.g., {"output": {"actors": [...]}})
    - JSON inside ```json fences
    - multiple JSON blobs; we usually want the LAST valid one
    """
    candidates = _extract_all_json_objects(text)

    # 1) Prefer exact top-level match (dict with required keys)
    for obj in reversed(candidates):
        if isinstance(obj, dict) and all(k in obj for k in required_top_keys):
            return obj

    # 2) If a single key is required and the model returned a top-level list, wrap it.
    if len(required_top_keys) == 1:
        k = required_top_keys[0]
        for obj in reversed(candidates):
            if isinstance(obj, list):
                return {k: obj}

    # 3) If key is nested somewhere, lift it to top-level.
    lifted: Dict[str, Any] = {}
    for k in required_top_keys:
        found = None
        for obj in reversed(candidates):
            found = _find_key_recursive(obj, k)
            if found is not None:
                break
        if found is None:
            lifted = {}
            break
        lifted[k] = found

    if lifted:
        return lifted

    # 4) Last resort: salvage by regex for the first required key (common case: actors)
    if len(required_top_keys) == 1:
        k = required_top_keys[0]
        # Try to grab an array following the key, even if surrounded by prose
        m = re.search(rf'"{re.escape(k)}"\s*:\s*(\[[\s\S]*?\])', text)
        if m:
            try:
                arr = json.loads(m.group(1))
                if isinstance(arr, list):
                    return {k: arr}
            except Exception:
                pass
        # Try to grab an object containing the key
        m2 = re.search(rf'\{{[\s\S]*?"{re.escape(k)}"\s*:\s*(\[[\s\S]*?\])[\s\S]*?\}}', text)
        if m2:
            snippet = m2.group(0)
            try:
                obj = json.loads(snippet)
                if isinstance(obj, dict) and k in obj:
                    return obj
            except Exception:
                pass

    # 5) Give up
    raise ValueError(
        f"Could not find JSON containing required top-level keys: {required_top_keys}.\n"
        f"Raw model output (truncated): {text[:800]!r}"
    )


__all__ = [
    "_extract_all_json_objects",
    "_find_key_recursive",
    "_pick_last_matching",
    "parse_llm_json",
]
