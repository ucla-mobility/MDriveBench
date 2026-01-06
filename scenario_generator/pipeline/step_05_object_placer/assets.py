from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import json


@dataclass
class AssetBBox:
    """Bounding box dimensions for an asset (in meters)."""
    length: float  # extent in forward direction (x)
    width: float   # extent in lateral direction (y)
    height: float  # extent in vertical direction (z)


@dataclass
class Asset:
    category: str
    asset_id: str
    tags: List[str]
    attributes: List[Dict[str, Any]]
    bbox: Optional[AssetBBox] = None


# Global lookup for asset bounding boxes by asset_id
_ASSET_BBOX_CACHE: Dict[str, Optional[AssetBBox]] = {}


def load_assets(path: str) -> List[Asset]:
    global _ASSET_BBOX_CACHE
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: List[Asset] = []
    assets = data.get("assets", {})
    for cat, lst in assets.items():
        if not isinstance(lst, list):
            continue
        for a in lst:
            if not isinstance(a, dict):
                continue
            # Parse bounding box if present
            bbox_data = a.get("bbox")
            asset_bbox = None
            if isinstance(bbox_data, dict):
                try:
                    # Use length/width/height if available, else compute from extents
                    length = float(bbox_data.get("length", bbox_data.get("extent_x", 0) * 2))
                    width = float(bbox_data.get("width", bbox_data.get("extent_y", 0) * 2))
                    height = float(bbox_data.get("height", bbox_data.get("extent_z", 0) * 2))
                    if length > 0 and width > 0:
                        asset_bbox = AssetBBox(length=length, width=width, height=height)
                except (TypeError, ValueError):
                    pass
            
            asset_id = str(a.get("id", ""))
            out.append(Asset(
                category=str(cat),
                asset_id=asset_id,
                tags=[str(t).lower() for t in a.get("tags", []) if t is not None],
                attributes=a.get("attributes", []) if isinstance(a.get("attributes", []), list) else [],
                bbox=asset_bbox,
            ))
            # Cache the bbox for quick lookup
            _ASSET_BBOX_CACHE[asset_id] = asset_bbox
    return out


def get_asset_bbox(asset_id: str) -> Optional[AssetBBox]:
    """Get bounding box for an asset by its ID. Returns None if not available."""
    return _ASSET_BBOX_CACHE.get(asset_id)


def keyword_filter_assets(all_assets: List[Asset], keywords: List[str], categories: Optional[List[str]] = None, k: int = 12) -> List[Asset]:
    kws = [kw.lower().strip() for kw in keywords if kw and kw.strip()]
    cats = set([c.lower() for c in categories]) if categories else None

    scored: List[tuple] = []
    for a in all_assets:
        if cats and a.category.lower() not in cats:
            continue
        hay = " ".join([a.asset_id.lower()] + a.tags)
        score = 0.0
        for kw in kws:
            if kw in hay:
                score += 1.0
        if score > 0:
            scored.append((score, a))

    scored.sort(key=lambda x: (-x[0], x[1].asset_id))
    return [a for _, a in scored[:k]]


__all__ = [
    "Asset",
    "AssetBBox",
    "get_asset_bbox",
    "keyword_filter_assets",
    "load_assets",
]
