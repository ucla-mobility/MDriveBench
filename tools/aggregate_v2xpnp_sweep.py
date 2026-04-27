"""Aggregate AP numbers from a v2xpnp sweep directory.

Reads all `<scene>__<detector>__<mode>.log` files produced by
sweep_v2xpnp_probe.sh, extracts the BEV IoU metrics from each, and prints
mean AP@0.3/0.5/0.7 grouped by (detector, mode).
"""
from __future__ import annotations

import argparse
import collections
import os
import re
from pathlib import Path

# Regex for one IoU row inside a metrics block:
#     0.50    12   73   19      0.141   0.632  0.478
_ROW_RX = re.compile(
    r"^\s*(0\.\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)$",
    re.M,
)


def parse_log(path: Path):
    """Return dict { (block_label, iou) → (tp, fp, gt, precision, recall, ap) }."""
    txt = path.read_text(errors="replace")
    blocks = re.split(r"── BEV IoU metrics — ", txt)
    out = {}
    for blk in blocks[1:]:
        # First line of block is the label, e.g. "REAL movers only (NPCs + ego_1) ──"
        label_line, _, body = blk.partition("\n")
        label = label_line.strip().rstrip("─").strip()
        for m in _ROW_RX.finditer(body):
            iou = float(m.group(1))
            row = (int(m.group(2)), int(m.group(3)), int(m.group(4)),
                   float(m.group(5)), float(m.group(6)), float(m.group(7)))
            out[(label, iou)] = row
    return out


def parse_filename(name: str):
    # e.g. 2023-03-17-16-03-02_11_0__fcooper__cooperative.log
    base = name[:-4] if name.endswith(".log") else name
    parts = base.split("__")
    if len(parts) != 3:
        return None
    return parts  # [scene, detector, mode]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, type=Path)
    ap.add_argument("--label", default="REAL movers only",
                    help="Substring of the AP-block label to aggregate (default: real movers).")
    args = ap.parse_args()

    # bucket: (detector, mode, iou) → list of (ap, recall, precision)
    bucket = collections.defaultdict(list)
    n_scenes = collections.defaultdict(set)

    for f in sorted(args.logs.glob("*.log")):
        parts = parse_filename(f.name)
        if not parts:
            continue
        scene, detector, mode = parts
        rows = parse_log(f)
        if not rows:
            continue
        for (label, iou), (tp, fp, gt, prec, rec, ap_v) in rows.items():
            if args.label.lower() not in label.lower():
                continue
            bucket[(detector, mode, iou)].append((ap_v, rec, prec))
            n_scenes[(detector, mode)].add(scene)

    if not bucket:
        print("[aggregate] no matching results found in", args.logs)
        return

    # Pretty table.
    detectors = sorted({k[0] for k in bucket})
    modes     = sorted({k[1] for k in bucket})
    ious      = sorted({k[2] for k in bucket})
    label_used = next(iter({k for k in bucket}))  # any
    print(f"\nv2xpnp sweep aggregate ({args.label} block) — averaged over scenes\n")
    header = f"{'detector':<10} {'mode':<12} {'#scenes':>7}"
    for iou in ious:
        header += f"  {'AP@'+str(iou):>7}  {'R@'+str(iou):>6}  {'P@'+str(iou):>6}"
    print(header)
    print("-" * len(header))
    for d in detectors:
        for m in modes:
            row = f"{d:<10} {m:<12} {len(n_scenes[(d,m)]):>7}"
            for iou in ious:
                vals = bucket.get((d, m, iou), [])
                if not vals:
                    row += f"  {'—':>7}  {'—':>6}  {'—':>6}"
                else:
                    aps = [v[0] for v in vals]
                    rs  = [v[1] for v in vals]
                    ps  = [v[2] for v in vals]
                    row += f"  {sum(aps)/len(aps):>7.3f}  {sum(rs)/len(rs):>6.3f}  {sum(ps)/len(ps):>6.3f}"
            print(row)


if __name__ == "__main__":
    main()
