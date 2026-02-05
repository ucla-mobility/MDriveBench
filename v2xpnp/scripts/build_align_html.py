#!/usr/bin/env python3
"""
Build a self-contained HTML alignment tool that overlays:
  - PKL map polylines (fixed)
  - CARLA map polylines (draggable/offsettable)
  - Initial vehicle positions from the first timestep of the dataset

The HTML lets you drag the CARLA layer, toggle Y-flip, and read off tx/ty (meters).

Usage:
  python v2xpnp/scripts/build_align_html.py \
      --pkl-map v2xpnp/map/v2x_intersection_vector_map.pkl \
      --carla-map v2xpnp/map/carla_map_cache.pkl \
      --dataset /data2/marco/CoLMDriver/v2xpnp/Sample_Dataset/2023-03-17-16-12-12_3_0 \
      --out v2xpnp/pkl_xodr_manual_align.html
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def list_yaml_timesteps(folder: Path) -> List[Path]:
    files = [p for p in folder.iterdir() if p.suffix.lower() == ".yaml"]
    files.sort()
    return files


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def extract_lines_generic(obj: Any, out: List[List[Tuple[float, float]]], depth: int = 0) -> None:
    """Heuristic extraction of polylines from arbitrary pickle structures."""
    if obj is None or depth > 10:
        return
    if isinstance(obj, dict):
        if "x" in obj and "y" in obj:
            try:
                out.append([(float(obj["x"]), float(obj["y"]))])
            except Exception:
                pass
        for v in obj.values():
            extract_lines_generic(v, out, depth + 1)
        return
    if isinstance(obj, (list, tuple)):
        if len(obj) >= 2 and all(hasattr(it, "__len__") and len(it) >= 2 for it in obj if it is not None):
            try:
                pts = [(float(p[0]), float(p[1])) for p in obj if p is not None and len(p) >= 2]
                if len(pts) >= 2:
                    out.append(pts)
                    return
            except Exception:
                pass
        for v in obj:
            extract_lines_generic(v, out, depth + 1)
        return
    if hasattr(obj, "x") and hasattr(obj, "y"):
        try:
            out.append([(float(obj.x), float(obj.y))])
        except Exception:
            pass
        return
    if hasattr(obj, "__dict__"):
        extract_lines_generic(obj.__dict__, out, depth + 1)


def load_polyline_pkl(path: Path) -> List[List[Tuple[float, float]]]:
    class _Stub:
        def __init__(self, *args, **kwargs):
            self.__dict__.update(kwargs)
            for i, a in enumerate(args):
                setattr(self, f"_arg{i}", a)

    class _SafeUnpickler(pickle.Unpickler):
        def find_class(self, module, name):  # pragma: no cover
            try:
                return super().find_class(module, name)
            except Exception:
                return _Stub

    try:
        with path.open("rb") as f:
            data = pickle.load(f)
    except Exception:
        with path.open("rb") as f:
            data = _SafeUnpickler(f).load()

    if isinstance(data, dict) and "lines" in data:
        return [[(float(x), float(y)) for (x, y) in line] for line in data["lines"]]
    lines: List[List[Tuple[float, float]]] = []
    extract_lines_generic(data, lines)
    return lines


def load_initial_positions(dataset: Path) -> List[Tuple[float, float, int]]:
    """Return list of (x, y, id) from earliest timestep found across subfolders."""
    candidates = []
    for sub in sorted(dataset.iterdir()):
        if sub.is_dir():
            ys = list_yaml_timesteps(sub)
            if ys:
                candidates.append(ys[0])
    if not candidates:
        ys = list_yaml_timesteps(dataset)
        if ys:
            candidates.append(ys[0])
    if not candidates:
        return []
    # pick earliest by name
    first = sorted(candidates)[0]
    data = load_yaml(first)
    vehs = data.get("vehicles", {}) or {}
    pts: List[Tuple[float, float, int]] = []
    for vid_str, payload in vehs.items():
        try:
            vid = int(vid_str)
        except Exception:
            continue
        loc = payload.get("location")
        if isinstance(loc, list) and len(loc) >= 2:
            try:
                pts.append((float(loc[0]), float(loc[1]), vid))
            except Exception:
                continue
    return pts


HTML_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>PKL vs CARLA Map Alignment</title>
  <style>
    body { margin: 0; font-family: sans-serif; display: flex; height: 100vh; }
    #left { flex: 1; position: relative; background: #111; color: #eee; }
    #ui { width: 320px; padding: 12px; background: #1e1e1e; color: #eee; overflow-y: auto; }
    canvas { width: 100%; height: 100%; display: block; }
    button, input, label { font-size: 14px; }
    .row { margin: 6px 0; }
    code { color: #ffd479; }
  </style>
</head>
<body>
  <div id="left">
    <canvas id="c"></canvas>
  </div>
  <div id="ui">
    <h3>Align CARLA → PKL</h3>
    <div class="row">Drag on canvas to move CARLA map.</div>
    <div class="row">
      <label>Scale (px/m): <input id="scale" type="number" step="0.5" value="2.0"></label>
    </div>
    <div class="row">
      <label><input id="flipY" type="checkbox"> Flip CARLA Y</label>
    </div>
    <div class="row">
      <button id="reset">Reset</button>
      <button id="copy">Copy tx/ty</button>
    </div>
    <div class="row">tx: <code id="tx">0</code> | ty: <code id="ty">0</code></div>
    <div class="row" style="font-size:12px; color:#aaa;">
      Offsets are in meters, applied after optional Y flip.
    </div>
  </div>
  <script>
    const pklLines = __PKL_LINES__;
    const carlaLinesRaw = __CARLA_LINES__;
    const points = __POINTS__; // [x, y, id]

    const canvas = document.getElementById('c');
    const ctx = canvas.getContext('2d');
    const scaleInput = document.getElementById('scale');
    const flipYEl = document.getElementById('flipY');
    const txEl = document.getElementById('tx');
    const tyEl = document.getElementById('ty');

    let pxPerM = 2.0;
    let tx = 0.0, ty = 0.0;
    let dragging = false;
    let last = null;

    function resize() {
      canvas.width = canvas.clientWidth * window.devicePixelRatio;
      canvas.height = canvas.clientHeight * window.devicePixelRatio;
      draw();
    }
    window.addEventListener('resize', resize);

    function worldBounds(lines, pts=[]) {
      let minx=Infinity,maxx=-Infinity,miny=Infinity,maxy=-Infinity;
      const add = (x,y)=>{minx=Math.min(minx,x);maxx=Math.max(maxx,x);miny=Math.min(miny,y);maxy=Math.max(maxy,y);};
      lines.forEach(line=>line.forEach(([x,y])=>add(x,y)));
      pts.forEach(([x,y])=>add(x,y));
      if(!isFinite(minx)) return null;
      return {minx,maxx,miny,maxy};
    }

    const baseBounds = worldBounds(pklLines, points.map(p=>[p[0],p[1]]));
    let center = baseBounds ? [(baseBounds.minx+baseBounds.maxx)/2, (baseBounds.miny+baseBounds.maxy)/2] : [0,0];

    function applyFlip(lines, flip) {
      if(!flip) return lines;
      return lines.map(line=>line.map(([x,y])=>[x,-y]));
    }

    function draw() {
      ctx.clearRect(0,0,canvas.width,canvas.height);
      ctx.save();
      ctx.translate(canvas.width/2, canvas.height/2);
      ctx.scale(1, -1); // screen Y up
      const s = pxPerM * window.devicePixelRatio;

      // draw PKL map
      ctx.strokeStyle = '#888';
      ctx.lineWidth = 1;
      pklLines.forEach(line=>{
        if(line.length<2) return;
        ctx.beginPath();
        line.forEach(([x,y],i)=>{
          const sx = (x-center[0])*s;
          const sy = (y-center[1])*s;
          if(i===0) ctx.moveTo(sx,sy); else ctx.lineTo(sx,sy);
        });
        ctx.stroke();
      });

      // draw CARLA map with offset and optional flip
      const carlaLines = applyFlip(carlaLinesRaw, flipYEl.checked);
      ctx.strokeStyle = '#cfa93a';
      ctx.lineWidth = 1;
      carlaLines.forEach(line=>{
        if(line.length<2) return;
        ctx.beginPath();
        line.forEach(([x,y],i)=>{
          const sx = ((x+tx)-center[0])*s;
          const sy = ((y+ty)-center[1])*s;
          if(i===0) ctx.moveTo(sx,sy); else ctx.lineTo(sx,sy);
        });
        ctx.stroke();
      });

      // draw vehicle points
      ctx.fillStyle = '#4ac';
      points.forEach(([x,y,id])=>{
        const sx = (x-center[0])*s;
        const sy = (y-center[1])*s;
        ctx.beginPath();
        ctx.arc(sx, sy, 3, 0, Math.PI*2);
        ctx.fill();
      });

      ctx.restore();
      txEl.textContent = tx.toFixed(4);
      tyEl.textContent = ty.toFixed(4);
    }

    canvas.addEventListener('mousedown', e=>{
      dragging = true;
      last = {x:e.clientX, y:e.clientY};
    });
    window.addEventListener('mouseup', ()=> dragging=false);
    window.addEventListener('mousemove', e=>{
      if(!dragging || !last) return;
      const dx = e.clientX - last.x;
      const dy = e.clientY - last.y;
      const s = pxPerM;
      // dx in pixels -> meters
      tx += dx / s;
      ty -= dy / s; // screen y grows down, world up
      last = {x:e.clientX, y:e.clientY};
      draw();
    });

    scaleInput.addEventListener('change', ()=>{
      const v = parseFloat(scaleInput.value);
      if(v>0) { pxPerM = v; draw(); }
    });
    flipYEl.addEventListener('change', draw);
    document.getElementById('reset').addEventListener('click', ()=>{
      tx=0; ty=0; flipYEl.checked=false; draw();
    });
    document.getElementById('copy').addEventListener('click', ()=>{
      const txt = JSON.stringify({tx: parseFloat(tx.toFixed(6)), ty: parseFloat(ty.toFixed(6)), flip_y: flipYEl.checked});
      navigator.clipboard.writeText(txt).then(()=>alert("Copied: "+txt)).catch(()=>{});
    });

    resize();
  </script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Build interactive PKL vs CARLA alignment HTML")
    ap.add_argument("--pkl-map", required=True, type=Path, help="PKL map with polylines")
    ap.add_argument("--carla-map", required=True, type=Path, help="CARLA map cache PKL (dict with 'lines' or generic polylines)")
    ap.add_argument("--dataset", required=True, type=Path, help="Dataset root containing YAML frames")
    ap.add_argument("--out", default=Path("v2xpnp/pkl_xodr_manual_align.html"), type=Path, help="Output HTML file")
    args = ap.parse_args()

    pkl_lines = load_polyline_pkl(args.pkl_map)
    carla_lines = load_polyline_pkl(args.carla_map)
    points = load_initial_positions(args.dataset)

    html = HTML_TEMPLATE.replace("__PKL_LINES__", json.dumps(pkl_lines)) \
        .replace("__CARLA_LINES__", json.dumps(carla_lines)) \
        .replace("__POINTS__", json.dumps(points))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"[OK] Wrote {args.out} (pkl lines={len(pkl_lines)}, carla lines={len(carla_lines)}, points={len(points)})")


if __name__ == "__main__":
    main()
