import re
from typing import Any, Dict, List, Tuple


_GEOM_WORDS = {
    "intersection", "junction", "tjunction", "roundabout", "road", "street", "highway", "freeway",
    "lane", "lanes", "shoulder", "median", "curb", "sidewalk", "crosswalk", "ramp", "exit", "entry",
    "approach", "connector", "merge", "turn", "turning", "intersectional", "roadway",
}
_STOP_WORDS = {
    "the", "a", "an", "of", "to", "in", "on", "at", "into", "through", "from", "for", "with",
    "and", "or", "near", "by", "around", "across", "along", "before", "after", "during",
}
_DIR_WORDS = {
    "left", "right", "center", "middle", "same", "opposite", "oncoming", "main", "side",
    "straight", "forward", "behind", "ahead", "front", "back",
}
_EGO_VEHICLE_RE = re.compile(r"\bvehicle\s+\d+\b", re.IGNORECASE)
_NON_EGO_QUALIFIERS = {
    "npc", "non-ego", "non ego", "other", "another", "oncoming", "incoming", "approaching",
    "opposing", "opposite", "traffic",
}
_SPECIFIC_VEHICLE_NOUNS = {
    "truck", "bus", "van", "car", "pickup", "motorcycle", "bike", "bicycle", "cyclist",
}


def _norm_ws_lower(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def _contains_exact_quote(description: str, quote: str) -> bool:
    """Best-effort check that quote appears in description (case-insensitive, whitespace-normalized)."""
    d = _norm_ws_lower(description)
    q = _norm_ws_lower(quote)
    if not q:
        return False
    return q in d


def _is_ego_vehicle_mention(mention: str) -> bool:
    """True iff mention is exactly 'Vehicle N' (ego naming)."""
    m = _norm_ws_lower(mention)
    # Allow users to explicitly refer to NPCs like "NPC Vehicle 1"
    if "npc" in m or "non-ego" in m or "non ego" in m:
        return False
    return re.fullmatch(r"vehicle\s+\d+", m) is not None


def _ego_vehicle_mentions(text: str) -> List[str]:
    return _EGO_VEHICLE_RE.findall(_norm_ws_lower(text))


def _has_non_ego_qualifier(text: str) -> bool:
    t = _norm_ws_lower(text)
    return any(q in t for q in _NON_EGO_QUALIFIERS)


def _extract_ego_vehicle_numbers(text: str) -> set:
    """Extract Vehicle N references (ego naming) from a string."""
    s = _norm_ws_lower(text)
    if not s:
        return set()
    if "npc" in s or "non-ego" in s or "non ego" in s:
        return set()
    return {int(m.group(1)) for m in re.finditer(r"\bvehicle\s+(\d+)\b", s)}


def _is_pure_map_geometry_phrase(phrase: str) -> bool:
    """Heuristic: phrase is basically just a map/location noun like 'the intersection' or 'left lane'."""
    s = _norm_ws_lower(phrase)
    toks = re.findall(r"[a-z0-9]+", s)
    toks = [t for t in toks if t not in _STOP_WORDS and t not in _DIR_WORDS]
    if not toks:
        return False
    return all(t in _GEOM_WORDS for t in toks)


def _should_drop_stage1_entity(e: Dict[str, Any], description: str) -> Tuple[bool, str]:
    mention = str(e.get("mention") or "").strip()
    kind = str(e.get("actor_kind") or "").strip()
    evidence = str(e.get("evidence") or "").strip()

    if not mention:
        return True, "empty mention"

    # Hard guardrail: never spawn ego vehicles as entities.
    if _is_ego_vehicle_mention(mention):
        return True, "ego vehicle mention (Vehicle N)"

    # Drop NPC vehicle entities that are just a list of ego vehicles (Vehicle N, Vehicle M, ...).
    if kind == "npc_vehicle":
        mention_norm = _norm_ws_lower(mention)
        if "vehicle" in mention_norm and not _has_non_ego_qualifier(mention_norm):
            if any(noun in mention_norm for noun in _SPECIFIC_VEHICLE_NOUNS):
                pass
            else:
                ego_refs = set(_ego_vehicle_mentions(f"{mention} {evidence}"))
                if len(ego_refs) >= 2:
                    return True, "ego vehicle group mention (Vehicle N list)"
    if kind in ("npc_vehicle", "parked_vehicle"):
        ego_refs = _extract_ego_vehicle_numbers(mention) | _extract_ego_vehicle_numbers(evidence)
        # Drop if the actor is actually described as multiple ego vehicles.
        if len(ego_refs) >= 2:
            return True, "ego vehicle references in NPC/parked entity"

    # Evidence requirement (Fix D): require a quote that actually appears in the description.
    # If evidence is missing or mismatched, fall back to mention ONLY if mention appears.
    if evidence:
        if not _contains_exact_quote(description, evidence):
            if _contains_exact_quote(description, mention):
                e["evidence"] = mention
            else:
                return True, "evidence not found in description"
    else:
        if _contains_exact_quote(description, mention):
            e["evidence"] = mention
        else:
            return True, "no evidence/mention match in description"

    # Drop obvious map-geometry pseudo-entities (Fix A), but avoid over-filtering.
    # - Only apply aggressively to static props / parked vehicles where geometry confusions are common.
    # - Keep pedestrians/cyclists/NPC vehicles even if mention contains location words.
    if kind in ("static_prop", "parked_vehicle"):
        if _is_pure_map_geometry_phrase(mention):
            return True, "map-geometry phrase (not a spawnable actor)"

    return False, ""


__all__ = [
    "_contains_exact_quote",
    "_extract_ego_vehicle_numbers",
    "_is_ego_vehicle_mention",
    "_is_pure_map_geometry_phrase",
    "_norm_ws_lower",
    "_should_drop_stage1_entity",
]
