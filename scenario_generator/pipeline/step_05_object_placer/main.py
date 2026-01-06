import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import numpy as np
except Exception as e:
    raise RuntimeError("This script requires numpy") from e

from .assets import get_asset_bbox, keyword_filter_assets, load_assets
from .csp import (
    _compute_merge_min_s_by_vehicle,
    build_stage2_constraints_prompt,
    expand_group_to_actors,
    solve_weighted_csp_with_extension,
    validate_actor_specs,
)
from .filters import _contains_exact_quote, _should_drop_stage1_entity
from .guardrails import (
    apply_after_turn_segment_corrections,
    apply_in_intersection_segment_corrections,
    build_repair_prompt,
    validate_stage2_output,
)
from .model import generate_with_model
from .nodes import _override_seg_points_with_picked, build_segments_from_nodes, load_nodes
from .parsing import parse_llm_json
from .prompts import build_stage1_prompt, build_stage2_prompt, build_vehicle_segment_summaries
from .spawn import build_motion_waypoints, compute_spawn_from_anchor, resolve_nodes_path
from .viz import visualize
import math
import numpy as np


def run_object_placer(args, model=None, tokenizer=None):
    """
    Main pipeline body, optionally reusing a provided model/tokenizer.
    """
    t_obj_start = time.time()
    # Set default values for optional args that may not be provided by SimpleNamespace
    if not hasattr(args, 'placement_mode'):
        args.placement_mode = "csp"  # default to CSP-based placement
    if not hasattr(args, 'do_sample'):
        args.do_sample = False
    if not hasattr(args, 'temperature'):
        args.temperature = 0.2
    if not hasattr(args, 'top_p'):
        args.top_p = 0.95

    t0 = time.time()
    with open(args.picked_paths, "r", encoding="utf-8") as f:
        picked_payload = json.load(f)

    picked = picked_payload.get("picked", [])
    if not isinstance(picked, list) or not picked:
        raise SystemExit("[ERROR] picked_paths_detailed.json has no 'picked' list.")

    crop_region = picked_payload.get("crop_region")
    nodes_field = picked_payload.get("nodes")
    if not nodes_field:
        raise SystemExit("[ERROR] picked_paths_detailed.json missing 'nodes' field")

    all_assets = load_assets(args.carla_assets)

    # Build vehicle segment summaries for LLM
    vehicle_segments = build_vehicle_segment_summaries(picked)
    print(f"[TIMING] object_placer setup (load paths, assets, summaries): {time.time() - t0:.2f}s", flush=True)

    # Load HF model if not provided
    if tokenizer is None or model is None:
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model.eval()
        print(f"[TIMING] object_placer model load: {time.time() - t0:.2f}s", flush=True)

    # --------------------------
    # Stage 1: extract entities
    # --------------------------
    t_stage1_start = time.time()
    stage1_prompt = build_stage1_prompt(args.description)
    t0 = time.time()
    stage1_text = generate_with_model(
        model=model,
        tokenizer=tokenizer,
        prompt=stage1_prompt,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(f"[TIMING] Stage1 LLM generation: {time.time() - t0:.2f}s", flush=True)
    t0 = time.time()
    # Stage 1 parse (with repair if the model didn't output JSON)
    try:
        stage1_obj = parse_llm_json(stage1_text, required_top_keys=["entities"])
    except Exception:
        repair_prompt = (
            "Return JSON ONLY with top-level key 'entities' (a list). No prose.\n"
            "If you previously wrote anything else, convert it into the required JSON now.\n\n"
            "RAW OUTPUT:\n" + stage1_text
        )
        repair_text = generate_with_model(
            model=model,
            tokenizer=tokenizer,
            prompt=repair_prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        stage1_obj = parse_llm_json(repair_text, required_top_keys=["entities"])
    print(f"[TIMING] Stage1 parse+repair: {time.time() - t0:.2f}s", flush=True)

    entities = stage1_obj.get("entities", [])
    if not isinstance(entities, list):
        raise SystemExit("[ERROR] Stage1: 'entities' must be a list.")

    # Ensure each entity has a unique entity_id (normalize if LLM didn't provide one)
    valid_entity_ids = set()
    for idx, e in enumerate(entities):
        if not e.get("entity_id"):
            e["entity_id"] = f"entity_{idx + 1}"
        valid_entity_ids.add(e["entity_id"])

    
    # Post-filter Stage 1 entities to reduce hallucinations (Fix A + Fix D)
    t0 = time.time()
    dropped_stage1: List[Tuple[str, str, str]] = []
    filtered_entities: List[Dict[str, Any]] = []

    def _repair_evidence_with_llm(ent: Dict[str, Any]) -> None:
        """Best-effort: if Stage1 paraphrased, try to recover an EXACT supporting quote."""
        ev = str(ent.get("evidence") or "").strip()
        mention = str(ent.get("mention") or "").strip()
        if ev and _contains_exact_quote(args.description, ev):
            return
        if mention and _contains_exact_quote(args.description, mention):
            ent["evidence"] = mention
            return

        # One-shot repair: ask the model to point to an exact substring.
        try:
            prompt = (
                "Return JSON ONLY: {\"evidence\": \"...\"}.\n"
                "The evidence MUST be an EXACT substring (<=20 words) copied from DESCRIPTION that explicitly mentions the actor (not just a location).\n"
                "If you cannot find any supporting substring, return {\"evidence\": \"\"}.\n\n"
                f"ACTOR_KIND: {ent.get('actor_kind','')}\n"
                f"MENTION (may be paraphrase): {mention}\n\n"
                f"DESCRIPTION:\n{args.description}\n"
            )
            txt = generate_with_model(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=80,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
            )
            obj = parse_llm_json(txt, required_top_keys=["evidence"])
            rep = str(obj.get("evidence") or "").strip()
            if rep and _contains_exact_quote(args.description, rep):
                ent["evidence"] = rep
        except Exception:
            return

    for e in entities:
        _repair_evidence_with_llm(e)

        drop, reason = _should_drop_stage1_entity(e, args.description)
        if drop:
            dropped_stage1.append((str(e.get("entity_id")), reason, str(e.get("mention") or "")))
            continue
        filtered_entities.append(e)
    if dropped_stage1:
        for ent_id, reason, mention in dropped_stage1[:50]:
            print(f"[WARNING] Stage1 drop: {ent_id}: {reason}; mention='{mention[:60]}'")
    entities = filtered_entities
    valid_entity_ids = set(str(e.get("entity_id")) for e in entities if e.get("entity_id"))
    print(f"[TIMING] Stage1 entity filtering+evidence repair: {time.time() - t0:.2f}s", flush=True)
    print(f"[TIMING] Stage1 total: {time.time() - t_stage1_start:.2f}s", flush=True)

    print(f"[INFO] Stage1 extracted {len(entities)} entities: {list(valid_entity_ids)}")
    # Debug: show Stage 1 entity details including motion_hint and when (route phase)
    for e in entities:
        print(f"  - {e.get('entity_id')}: kind={e.get('actor_kind')}, motion_hint={e.get('motion_hint')}, when={e.get('when')}, mention='{e.get('mention', '')[:50]}'")

    # Keyword synonyms for better asset matching
    t0 = time.time()
    KEYWORD_SYNONYMS = {
        "cyclist": ["bike", "bicycle", "crossbike"],
        "bicyclist": ["bike", "bicycle", "crossbike"],
        "biker": ["bike", "bicycle", "crossbike", "motorcycle"],
        "motorcyclist": ["motorcycle", "harley", "yamaha", "kawasaki"],
        "pedestrian": ["pedestrian", "walker", "person"],
        "person": ["pedestrian", "walker"],
        "cone": ["cone", "trafficcone", "constructioncone"],
        "cones": ["cone", "trafficcone", "constructioncone"],
        "traffic cone": ["cone", "trafficcone", "constructioncone"],
        "barrier": ["barrier", "streetbarrier", "construction"],
        "truck": ["truck", "firetruck", "cybertruck", "pickup"],
        "police": ["police", "charger", "crown"],
        "ambulance": ["ambulance"],
        "firetruck": ["firetruck"],
    }
    OCCLUSION_HINTS = (
        "obstruct", "obstructs", "obstructing", "occlude", "occluding", "visibility",
        "view", "block", "blocking", "obscure", "obscuring", "blind", "line of sight",
    )
    LARGE_STATIC_TOKENS = (
        "streetbarrier", "barrier", "chainbarrier", "busstop", "advertisement",
        "container", "clothcontainer", "glasscontainer",
    )
    SMALL_STATIC_TOKENS = (
        "barrel", "cone", "trafficcone", "bin", "box", "trash", "garbage", "bag", "bottle", "can",
    )
    SPECIFIC_STATIC_TOKENS = LARGE_STATIC_TOKENS + SMALL_STATIC_TOKENS

    def _has_occlusion_hint(mention: str, evidence: str, description: str, static_count: int) -> bool:
        text = f"{mention} {evidence}".lower()
        if any(h in text for h in OCCLUSION_HINTS):
            return True
        if static_count == 1:
            dlow = str(description or "").lower()
            return any(h in dlow for h in OCCLUSION_HINTS)
        return False

    def _mentions_specific_static_prop(text: str) -> bool:
        low = str(text or "").lower()
        return any(t in low for t in SPECIFIC_STATIC_TOKENS)

    def _asset_matches_tokens(asset, tokens) -> bool:
        hay = " ".join([asset.asset_id.lower()] + asset.tags)
        return any(t in hay for t in tokens)

    def _asset_area(asset) -> float:
        if not asset.bbox:
            return 0.0
        return float(asset.bbox.length * asset.bbox.width)

    def _merge_assets(primary, secondary, limit: int = 12):
        seen = set()
        out = []
        for a in list(primary) + list(secondary):
            if a.asset_id in seen:
                continue
            seen.add(a.asset_id)
            out.append(a)
            if len(out) >= limit:
                break
        return out

    large_vehicle_assets = [a for a in all_assets if a.category == "vehicle" and a.bbox]
    large_vehicle_assets.sort(key=_asset_area, reverse=True)
    top_vehicle_occluders = large_vehicle_assets[:6]

    static_like_count = sum(
        1 for e in entities if str(e.get("actor_kind", "")) in ("static_prop", "parked_vehicle")
    )

    # For each entity, build small asset option list (keyed by entity_id)
    per_entity_options: Dict[str, List[Dict[str, Any]]] = {}
    for idx, e in enumerate(entities):
        entity_id = e.get("entity_id", f"entity_{idx+1}")
        mention = str(e.get("mention", f"entity_{idx+1}"))
        evidence = str(e.get("evidence", ""))
        kind = str(e.get("actor_kind", "static_prop"))
        low = mention.lower()
        occlusion_hint = False
        if kind in ("static_prop", "parked_vehicle"):
            occlusion_hint = _has_occlusion_hint(mention, evidence, args.description, static_like_count)

        # Extract keywords: split mention into words and check synonyms
        kws = set()
        words = [w.strip(".,!?") for w in low.split()]
        
        # Add all meaningful words from mention
        for word in words:
            if len(word) > 2:  # Skip tiny words
                kws.add(word)
            # Expand synonyms
            if word in KEYWORD_SYNONYMS:
                kws.update(KEYWORD_SYNONYMS[word])
        
        # Also check multi-word phrases
        for phrase, synonyms in KEYWORD_SYNONYMS.items():
            if phrase in low:
                kws.update(synonyms)

        # Add kind-based keywords (always, not just as fallback)
        if kind == "walker":
            categories = ["walker"]
            kws.update(["pedestrian", "walker"])
        elif kind == "cyclist":
            categories = ["vehicle"]
            kws.update(["bike", "bicycle", "crossbike"])
        elif kind in ("parked_vehicle", "npc_vehicle"):
            categories = ["vehicle"]
            # Keep mention-based keywords; add generic fallbacks only if empty
            if not any(w in kws for w in ["car", "truck", "bus", "van", "vehicle"]):
                kws.update(["car", "vehicle"])
        else:
            categories = ["static"]
            kws.update(["prop", "static"])
            if occlusion_hint:
                kws.update(["barrier", "streetbarrier", "chainbarrier", "busstop", "advertisement", "container"])

        kws_list = list(kws)
        options = keyword_filter_assets(all_assets, kws_list, categories=categories, k=12)

        # last-resort fallback options
        if not options:
            # choose a small default set by category
            options = keyword_filter_assets(all_assets, ["vehicle"], categories=categories, k=12) or all_assets[:12]

        if occlusion_hint and kind == "static_prop":
            text = f"{mention} {evidence}"
            if not _mentions_specific_static_prop(text):
                options = _merge_assets(top_vehicle_occluders, options)
            large = [a for a in options if _asset_matches_tokens(a, LARGE_STATIC_TOKENS)]
            if large:
                options = _merge_assets(large, options)
            else:
                options.sort(key=lambda a: (_asset_matches_tokens(a, SMALL_STATIC_TOKENS), a.asset_id))

        # Key by entity_id for Stage 2
        per_entity_options[entity_id] = [
            {"asset_id": a.asset_id, "category": a.category, "tags": a.tags[:6]} for a in options
        ]
    print(f"[TIMING] asset matching for {len(entities)} entities: {time.time() - t0:.2f}s", flush=True)

    # Handle empty entities case early
    if not entities:
        print("[INFO] No entities extracted in Stage 1. Skipping Stage 2.")
        actors = []
        stage2_obj = {"actors": []}
    else:
        # --------------------------
        # Stage 2: resolve anchors
        # --------------------------
        t_stage2_start = time.time()

        if args.placement_mode == "llm_anchor":
            stage2_prompt = build_stage2_prompt(args.description, vehicle_segments, entities, per_entity_options)
            stage2_text = generate_with_model(
                model=model,
                tokenizer=tokenizer,
                prompt=stage2_prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print("\n[DEBUG] Stage2 raw output (full):\n" + stage2_text + "\n", flush=True)

            try:
                stage2_obj = parse_llm_json(stage2_text, required_top_keys=["actors"])
            except ValueError as exc:
                repair_prompt = build_repair_prompt(stage2_text, [f"JSON parse failed: {exc}"])
                repair_text = generate_with_model(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=repair_prompt,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                stage2_obj = parse_llm_json(repair_text, required_top_keys=["actors"])
            actors = stage2_obj.get("actors", [])
            if not isinstance(actors, list):
                raise SystemExit("[ERROR] Stage2: 'actors' must be a list.")

            errs = validate_stage2_output(actors, vehicle_segments)
            if errs:
                repair_prompt = build_repair_prompt(stage2_text, errs)
                repair_text = generate_with_model(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=repair_prompt,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                stage2_obj = parse_llm_json(repair_text, required_top_keys=["actors"])
                actors = stage2_obj.get("actors", [])
                if not isinstance(actors, list):
                    raise SystemExit("[ERROR] Stage2 repair: 'actors' must be a list.")
            print(f"[TIMING] Stage2 (llm_anchor mode) total: {time.time() - t_stage2_start:.2f}s", flush=True)
        else:
            # CSP mode: LLM emits symbolic preferences; solver chooses anchors.
            # We need geometry (seg_by_id) to enumerate candidates; build it here.
            t0 = time.time()
            resolved_nodes_path = resolve_nodes_path(args.picked_paths, str(nodes_field), args.nodes_root)
            if not os.path.exists(resolved_nodes_path):
                raise SystemExit(f"[ERROR] nodes path not found: {resolved_nodes_path}\n"
                                 f"Tip: pass --nodes-root to resolve relative paths.")
            nodes = load_nodes(resolved_nodes_path)
            all_segments = build_segments_from_nodes(nodes)
            seg_by_id: Dict[int, np.ndarray] = {int(s["seg_id"]): s["points"] for s in all_segments}
            # Override with refined polylines from picked paths (so CSP uses accurate path lengths)
            seg_by_id = _override_seg_points_with_picked(picked, seg_by_id)
            merge_min_s_by_vehicle = _compute_merge_min_s_by_vehicle(picked_payload, picked, seg_by_id)
            print(f"[TIMING] Stage2 load nodes+build segments: {time.time() - t0:.2f}s", flush=True)

            t0 = time.time()
            stage2_prompt = build_stage2_constraints_prompt(args.description, vehicle_segments, entities, per_entity_options)
            stage2_text = generate_with_model(
                model=model,
                tokenizer=tokenizer,
                prompt=stage2_prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print(f"[TIMING] Stage2 LLM generation: {time.time() - t0:.2f}s", flush=True)
            print("\n[DEBUG] Stage2(CSP) raw output (full):\n" + stage2_text + "\n", flush=True)

            # Parse with a simple repair-on-failure
            t0 = time.time()
            try:
                stage2_obj = parse_llm_json(stage2_text, required_top_keys=["actor_specs"])
            except Exception:
                repair_prompt = (
                    "Return JSON ONLY with top-level key 'actor_specs' (a list). No prose.\n"
                    "If needed, convert your previous output into the required JSON now.\n\n"
                    "RAW OUTPUT:\n" + stage2_text
                )
                repair_text = generate_with_model(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=repair_prompt,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                stage2_obj = parse_llm_json(repair_text, required_top_keys=["actor_specs"])
            print(f"[TIMING] Stage2 parse+repair: {time.time() - t0:.2f}s", flush=True)

            actor_specs_raw = stage2_obj.get("actor_specs", [])
            actor_specs, warns = validate_actor_specs(actor_specs_raw, entities, per_entity_options, picked=picked)
            for w in warns[:50]:
                print("[WARNING] Stage2(CSP): " + w)

            if not actor_specs:
                actors = []
            else:
                t0 = time.time()
                print(f"[TIMING] Starting CSP solve with {len(actor_specs)} actor specs...", flush=True)
                # Use iterative CSP with path extension - extends paths on-demand when distance constraints can't be met
                # Returns expanded crop_region to include extended path areas
                chosen, dbg, crop_region = solve_weighted_csp_with_extension(
                    actor_specs, picked, seg_by_id, crop_region,
                    all_segments=all_segments,
                    nodes=nodes,
                    merge_min_s_by_vehicle=merge_min_s_by_vehicle,
                    min_sep_scale=1.0,
                    max_extension_iterations=3,
                )
                print(f"[TIMING] CSP solve (with extension): {time.time() - t0:.2f}s", flush=True)
                print("[INFO] CSP solve debug: " + json.dumps(dbg, indent=2))

                actors = []
                for spec in actor_specs:
                    sid = str(spec["id"])
                    cand = chosen.get(sid)
                    if cand is None:
                        continue

                    # Get bounding box info if available
                    asset_id = spec["asset_id"]
                    bbox = get_asset_bbox(asset_id)
                    bbox_info = None
                    if bbox:
                        bbox_info = {
                            "length": bbox.length,
                            "width": bbox.width,
                            "height": bbox.height,
                        }

                    base_actor = {
                        "id": sid,
                        "semantic": spec.get("semantic", sid),
                        "category": spec["category"],
                        "asset_id": asset_id,
                        "placement": {
                            "target_vehicle": f"Vehicle {cand.vehicle_num}" if cand.vehicle_num > 0 else None,
                            "segment_index": int(cand.segment_index),
                            "s_along": float(cand.s_along),
                            "lateral_relation": str(cand.lateral_relation),
                            "seg_id": int(cand.seg_id),  # Store seg_id directly for opposite-lane NPCs
                        },
                        "motion": spec.get("motion", {"type": "static", "speed_profile": "normal"}),
                        "confidence": float(spec.get("confidence", 0.6)),
                        "csp": {
                            "base_score": float(cand.base_score),
                            "path_s_m": float(cand.path_s_m),
                            "relations": spec.get("relations", []),
                        },
                        "bbox": bbox_info,  # Include bounding box if available
                    }

                    expanded = expand_group_to_actors(base_actor, spec, cand, seg_by_id)
                    # Validate expansion: check if quantity matches expected
                    expected_qty = int(spec.get("quantity", 1))
                    if len(expanded) < expected_qty:
                        print(f"[WARNING] {sid}: expected {expected_qty} actors but only got {len(expanded)} "
                              f"(group_pattern={spec.get('group_pattern')}, seg_id={cand.seg_id})", flush=True)
                    actors.extend(expanded)
            print(f"[TIMING] Stage2 (CSP mode) total: {time.time() - t_stage2_start:.2f}s", flush=True)


    # --------------------------
    # Geometry reconstruction
    # --------------------------
    t0 = time.time()
    resolved_nodes_path = resolve_nodes_path(args.picked_paths, str(nodes_field), args.nodes_root)
    if not os.path.exists(resolved_nodes_path):
        raise SystemExit(f"[ERROR] nodes path not found: {resolved_nodes_path}\n"
                         f"Tip: pass --nodes-root to resolve relative paths.")

    nodes = load_nodes(resolved_nodes_path)
    all_segments = build_segments_from_nodes(nodes)
    seg_by_id: Dict[int, np.ndarray] = {int(s["seg_id"]): s["points"] for s in all_segments}
    # If paths were refined (start/end trimming or synthetic segments), prefer those polylines.
    seg_by_id = _override_seg_points_with_picked(picked, seg_by_id)
    print(f"[TIMING] geometry reconstruction: {time.time() - t0:.2f}s", flush=True)

    # --------------------------
    # Convert anchors -> world
    # --------------------------
    t0 = time.time()
    # Guardrail: if Stage1 said "after_turn" but Stage2 placed the actor on a turning connector,
    # shift it onto the inferred post-turn (exit) segment before spawning.
    apply_after_turn_segment_corrections(actors, stage1_obj.get("entities", []), picked, seg_by_id)
    # Guardrail: if Stage1 said "in_intersection", keep it on a turn-connector segment.
    apply_in_intersection_segment_corrections(actors, stage1_obj.get("entities", []), picked, seg_by_id)

    actors_world: List[Dict[str, Any]] = []
    for a in actors:
        placement = a["placement"]
        tv = placement.get("target_vehicle")
        seg_idx = int(placement.get("segment_index", 1))  # 1-based
        s_along = float(placement["s_along"])
        lat_rel = placement["lateral_relation"]
        
        # Check if seg_id is directly specified (for opposite-lane NPCs)
        direct_seg_id = placement.get("seg_id")
        
        if direct_seg_id is not None:
            # Direct seg_id: use it directly without looking up picked paths
            seg_id = int(direct_seg_id)
            seg_pts = seg_by_id.get(seg_id)
        else:
            # Standard lookup via target_vehicle and picked paths
            # Find seg_id from the picked path signature order
            # vehicle_segments contains seg_id list in order via picked signature; easiest: pull from picked itself
            picked_entry = next((p for p in picked if p.get("vehicle") == tv), None)
            if not picked_entry:
                continue
            seg_ids = (picked_entry.get("signature", {}) or {}).get("segment_ids", [])
            if not isinstance(seg_ids, list) or seg_idx < 1 or seg_idx > len(seg_ids):
                continue
            seg_id = int(seg_ids[seg_idx - 1])

            seg_pts = seg_by_id.get(seg_id)
            if seg_pts is None:
                # fall back to polyline_sample if present
                segs_det = (picked_entry.get("signature", {}) or {}).get("segments_detailed", [])
                det = next((d for d in segs_det if int(d.get("seg_id", -1)) == seg_id), None)
                if det and isinstance(det.get("polyline_sample"), list) and det["polyline_sample"]:
                    seg_pts = np.array([[p["x"], p["y"]] for p in det["polyline_sample"]], dtype=float)

        if seg_pts is None:
            print(f"[WARNING] Missing segment geometry for seg_id={seg_id}; skipping actor {a.get('id')}")
            continue

        spawn = compute_spawn_from_anchor(seg_pts, s_along, lat_rel, placement.get("lateral_offset_m"))
        motion = a.get("motion", {}) if isinstance(a.get("motion", {}), dict) else {"type": "static"}
        # Let motion builder know anchor s for some types
        motion.setdefault("anchor_s_along", s_along)
        
        # Pass start_lateral for crossing direction inference
        motion.setdefault("start_lateral", lat_rel)

        # category normalization
        cat = str(a.get("category", "")).lower()
        if cat not in ("vehicle", "walker", "static", "cyclist"):
            # derive from asset category if needed
            # (we don't strictly enforce this)
            cat = "static"

        wps = build_motion_waypoints(motion, cat, spawn, seg_pts)
        if isinstance(wps, list) and wps and isinstance(wps[0], dict) and "x" in wps[0] and "y" in wps[0]:
            # Align spawn to the first waypoint so spawn == trajectory start.
            spawn = {
                "x": float(wps[0]["x"]),
                "y": float(wps[0]["y"]),
                "yaw_deg": float(wps[0].get("yaw_deg", spawn.get("yaw_deg", 0.0))),
            }

        actors_world.append({
            **a,
            "resolved": {
                "seg_id": seg_id,
                "nodes_path": resolved_nodes_path,
            },
            "spawn": spawn,
            "world_waypoints": wps,
        })
    print(f"[TIMING] anchor -> world conversion: {time.time() - t0:.2f}s", flush=True)

    # Enforce non-overlapping spawns using asset bounding boxes.
    def _approx_width(a: Dict[str, Any]) -> float:
        bbox = get_asset_bbox(str(a.get("asset_id", "")))
        if bbox and bbox.width and bbox.width > 0:
            return float(bbox.width)
        cat = str(a.get("category", "")).lower()
        # Fallback widths (meters)
        if cat == "vehicle":
            return 1.8
        if cat == "cyclist":
            return 0.6
        if cat == "walker":
            return 0.5
        return 0.4

    def _total_length(seg_pts: np.ndarray) -> float:
        if not isinstance(seg_pts, np.ndarray) or len(seg_pts) < 2:
            return 0.0
        diffs = seg_pts[1:] - seg_pts[:-1]
        return float(np.linalg.norm(diffs, axis=1).sum())

    def _reposition_forward(idx: int, meters: float) -> bool:
        """Shift actor forward along its segment by given meters; recompute spawn/waypoints."""
        try:
            actor = actors_world[idx]
            seg_id = int((actor.get("resolved") or {}).get("seg_id", -1))
            seg_pts = seg_by_id.get(seg_id)
            if seg_pts is None:
                return False
            total = _total_length(seg_pts)
            s_delta = meters / max(total, 1e-6)
            motion = actor.get("motion", {}) if isinstance(actor.get("motion", {}), dict) else {"type": "static"}
            # Increase anchor position modestly
            anchor_s = float(motion.get("anchor_s_along", 0.5))
            new_s = min(0.98, max(0.0, anchor_s + s_delta))
            motion["anchor_s_along"] = new_s
            lateral = str(motion.get("start_lateral", "center") or "center").lower()
            spawn = compute_spawn_from_anchor(seg_pts, new_s, lateral)
            cat = str(actor.get("category", "")).lower()
            wps = build_motion_waypoints(motion, cat, spawn, seg_pts)
            if isinstance(wps, list) and wps and isinstance(wps[0], dict):
                actor["spawn"] = {
                    "x": float(wps[0]["x"]),
                    "y": float(wps[0]["y"]),
                    "yaw_deg": float(wps[0].get("yaw_deg", spawn.get("yaw_deg", 0.0))),
                }
                actor["world_waypoints"] = wps
                actor["motion"] = motion
                return True
        except Exception:
            return False
        return False

    changed = True
    passes = 0
    while changed and passes < 5:
        changed = False
        passes += 1
        for i in range(len(actors_world)):
            ai = actors_world[i]
            xi, yi = float(ai.get("spawn", {}).get("x", 0.0)), float(ai.get("spawn", {}).get("y", 0.0))
            wi = _approx_width(ai)
            ri = wi * 0.5
            for j in range(i + 1, len(actors_world)):
                aj = actors_world[j]
                xj, yj = float(aj.get("spawn", {}).get("x", 0.0)), float(aj.get("spawn", {}).get("y", 0.0))
                wj = _approx_width(aj)
                rj = wj * 0.5
                d = math.hypot(xi - xj, yi - yj)
                threshold = ri + rj + 0.2  # small margin
                if d < threshold:
                    # Push the later actor forward along its segment by the overlap amount
                    push_m = max(0.5, threshold - d)
                    if _reposition_forward(j, push_m):
                        changed = True
                        # Update positions after move
                        xj, yj = float(actors_world[j]["spawn"]["x"]), float(actors_world[j]["spawn"]["y"])
    if passes > 0:
        print(f"[INFO] Non-overlap enforcement passes: {passes}")

    # Validation: Check if total placed actors matches expected from Stage 1
    stage1_entities = stage1_obj.get("entities", [])
    total_expected = sum(int(e.get("quantity", 1)) for e in stage1_entities)
    total_placed = len(actors_world)
    if total_placed < total_expected:
        print(f"[WARNING] VALIDATION FAILED: Stage 1 expected {total_expected} total actors "
              f"but only {total_placed} were placed. Check group expansion logs above.", flush=True)
        for e in stage1_entities:
            qty = int(e.get("quantity", 1))
            if qty > 1:
                eid = e.get("entity_id", "unknown")
                # Count how many actors have this entity as base
                placed_for_entity = sum(1 for a in actors_world if str(a.get("id", "")).startswith(eid))
                if placed_for_entity < qty:
                    print(f"  - {eid}: expected {qty}, placed {placed_for_entity}", flush=True)

    out_payload = {
        "source_picked_paths": args.picked_paths,
        "nodes": resolved_nodes_path,
        "crop_region": crop_region,
        "ego_picked": picked,
        "actors": actors_world,
        "macro_plan": stage1_obj.get("entities", []),
    }

    t0 = time.time()
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"[INFO] Wrote scene objects to: {args.out} (actors={len(actors_world)})")

    if args.viz:
        t_viz = time.time()
        visualize(
            picked=picked,
            seg_by_id=seg_by_id,
            actors_world=actors_world,
            crop_region=crop_region if isinstance(crop_region, dict) else None,
            out_path=args.viz_out,
            description=args.description,
            show=args.viz_show,
        )
        print(f"[TIMING] visualization: {time.time() - t_viz:.2f}s", flush=True)
    print(f"[TIMING] object_placer total (internal): {time.time() - t_obj_start:.2f}s", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--picked-paths", required=True, help="picked_paths_detailed.json")
    ap.add_argument("--carla-assets", required=True, help="carla_assets.json")

    ap.add_argument("--description", required=True, help="Natural-language scene description")
    ap.add_argument("--out", default="scene_objects.json", help="Output IR + placements JSON")
    ap.add_argument("--viz-out", default="scene_objects.png", help="Output visualization image")
    ap.add_argument("--viz", action="store_true", help="Enable visualization")
    ap.add_argument("--viz-show", action="store_true", help="Show plot window (if supported)")

    ap.add_argument("--nodes-root", default=None, help="Optional root to resolve relative nodes path")
    ap.add_argument("--placement-mode", default="csp", choices=["csp","llm_anchor"], help="Placement stage: weighted CSP (solver) or legacy LLM anchors")

    # LLM gen controls
    ap.add_argument("--max-new-tokens", type=int, default=1200)
    ap.add_argument("--do-sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.95)

    args = ap.parse_args()
    run_object_placer(args)


if __name__ == "__main__":
    main()
