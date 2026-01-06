import os
from typing import List

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except Exception:
    plt = None
    mpatches = None

from .models import CropBox, LaneSegment, LegalPath
from .signatures import build_path_signature, make_path_name


def visualize_legal_paths(segments: List[LaneSegment],
                          legal_paths: List[LegalPath],
                          crop: CropBox,
                          out_path: str):
    if plt is None:
        raise RuntimeError("matplotlib is not available for visualization")

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(crop.xmin - 5, crop.xmax + 5)
    ax.set_ylim(crop.ymin - 5, crop.ymax + 5)
    ax.invert_xaxis()
    ax.set_xlabel("X (meters)", fontsize=12)
    ax.set_ylabel("Y (meters)", fontsize=12)
    ax.set_title(f"Legal Path Segments (Total: {len(legal_paths)} paths)",
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    rect = mpatches.Rectangle(
        (crop.xmin, crop.ymin),
        crop.xmax - crop.xmin,
        crop.ymax - crop.ymin,
        linewidth=2,
        edgecolor='red',
        facecolor='none',
        linestyle='--',
        label='Crop Region'
    )
    ax.add_patch(rect)

    for seg in segments:
        ax.plot(seg.points[:, 0], seg.points[:, 1],
                color='lightgray', linewidth=1.0, alpha=0.5, zorder=1)

    cmap = plt.cm.get_cmap('tab20')
    for idx, path in enumerate(legal_paths):
        color = cmap(idx % 20)
        for seg in path.segments:
            ax.plot(seg.points[:, 0], seg.points[:, 1],
                    color=color, linewidth=2.5, alpha=0.7, zorder=2)
            ax.plot(seg.points[0, 0], seg.points[0, 1],
                    'o', color=color, markersize=6, zorder=3)
            ax.plot(seg.points[-1, 0], seg.points[-1, 1],
                    's', color=color, markersize=6, zorder=3)

    legend_elements = [
        mpatches.Patch(color='lightgray', label='All Segments'),
        mpatches.Patch(color='red', label='Crop Region'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='black', markersize=8, label='Start Point'),
        plt.Line2D([0], [0], marker='s', color='w',
                   markerfacecolor='black', markersize=8, label='End Point'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Visualization saved to: {out_path}")

    return fig, ax


def visualize_individual_paths(segments: List[LaneSegment],
                               legal_paths: List[LegalPath],
                               crop: CropBox,
                               output_dir: str):
    if plt is None:
        raise RuntimeError("matplotlib is not available for visualization")

    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Creating individual visualizations in: {output_dir}")

    for path_idx, path in enumerate(legal_paths):
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(crop.xmin - 5, crop.xmax + 5)
        ax.set_ylim(crop.ymin - 5, crop.ymax + 5)
        ax.invert_xaxis()
        ax.set_xlabel("X (meters)", fontsize=12)
        ax.set_ylabel("Y (meters)", fontsize=12)
        ax.set_title(
            f"Path {path_idx + 1}: {len(path.segments)} segments, {path.total_length:.1f}m total length",
            fontsize=13, fontweight='bold'
        )
        ax.grid(True, alpha=0.3)

        rect = mpatches.Rectangle(
            (crop.xmin, crop.ymin),
            crop.xmax - crop.xmin,
            crop.ymax - crop.ymin,
            linewidth=2,
            edgecolor='red',
            facecolor='none',
            linestyle='--',
            alpha=0.5
        )
        ax.add_patch(rect)

        for seg in segments:
            ax.plot(seg.points[:, 0], seg.points[:, 1],
                    color='lightgray', linewidth=0.8, alpha=0.3, zorder=1)

        cmap = plt.cm.get_cmap('viridis')
        for seg_idx, seg in enumerate(path.segments):
            color = cmap(seg_idx / max(1, len(path.segments) - 1))
            ax.plot(seg.points[:, 0], seg.points[:, 1],
                    color=color, linewidth=3.5, alpha=0.9, zorder=2,
                    label=f"Seg {seg_idx + 1}: road={seg.road_id}, lane={seg.lane_id}")
            ax.plot(seg.points[0, 0], seg.points[0, 1],
                    'o', color=color, markersize=8, zorder=3,
                    markeredgecolor='black', markeredgewidth=1)
            ax.plot(seg.points[-1, 0], seg.points[-1, 1],
                    's', color=color, markersize=8, zorder=3,
                    markeredgecolor='black', markeredgewidth=1)

        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        plt.tight_layout()

        # Name from signature (world-frame)
        sig = build_path_signature(path)
        name = make_path_name(path_idx, sig)
        out_file = os.path.join(output_dir, name + ".png")
        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        plt.close(fig)

        if (path_idx + 1) % 10 == 0:
            print(f"[INFO] Generated {path_idx + 1}/{len(legal_paths)} visualizations")

    print(f"[INFO] All {len(legal_paths)} individual visualizations saved to: {output_dir}")
