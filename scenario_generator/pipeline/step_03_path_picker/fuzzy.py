import re
from difflib import SequenceMatcher
from typing import List, Optional, Tuple


def _normalize(s: str) -> str:
    return re.sub(r"\s+", "", s.strip().lower())


def _best_fuzzy_match(requested: str, choices: List[str]) -> Optional[str]:
    """Very conservative fuzzy match for slightly mangled names."""
    if not choices:
        return None
    rn = _normalize(requested)
    best: Tuple[float, str] = (0.0, "")
    for c in choices:
        score = SequenceMatcher(None, rn, _normalize(c)).ratio()
        if score > best[0]:
            best = (score, c)
    return best[1] if best[0] >= 0.92 else None


__all__ = [
    "_best_fuzzy_match",
    "_normalize",
]
