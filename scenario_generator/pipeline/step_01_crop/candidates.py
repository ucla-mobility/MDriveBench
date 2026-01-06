from typing import List

import generate_legal_paths as glp

from .features import compute_crop_features, detect_junction_centers
from .models import CropFeatures, CropKey


def build_candidate_crops_for_town(
    town_name: str,
    town_json_path: str,
    radii: List[float],
    min_path_len: float,
    max_paths: int,
    max_depth: int,
) -> List[CropFeatures]:
    data = glp.load_nodes(town_json_path)
    segments_full = glp.build_segments(data, min_points=6)
    adj_full = glp.build_connectivity(segments_full)
    jcenters = detect_junction_centers(segments_full, adj_full)

    feats: List[CropFeatures] = []
    for jc in jcenters:
        cx, cy = float(jc[0]), float(jc[1])
        for r in radii:
            ck = CropKey(cx - r, cx + r, cy - r, cy + r)
            f = compute_crop_features(
                town_name=town_name,
                segments_full=segments_full,
                junction_centers=jcenters,
                center_xy=(cx, cy),
                crop=ck,
                min_path_len=min_path_len,
                max_paths=max_paths,
                max_depth=max_depth,
            )
            if f is not None:
                feats.append(f)

    uniq = {}
    for f in feats:
        k = f.crop.to_str()
        if k not in uniq:
            uniq[k] = f
        else:
            a = (f.junction_count, f.area)
            b = (uniq[k].junction_count, uniq[k].area)
            if a < b:
                uniq[k] = f

    out = list(uniq.values())
    out.sort(key=lambda x: (x.junction_count, x.area))
    return out
