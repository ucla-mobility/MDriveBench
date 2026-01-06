"""NumPy implementation of the IoU helpers used by OpenCOOD.

The original repository provides a Cython extension (`box_overlaps.pyx`)
for performance.  In this deployment the compiled module fails to load
because the ABI does not match the NumPy installation that ships with
the `colmdriver` environment.  A pure Python fallback keeps the
evaluation pipeline functional without requiring the binary extension.
"""

from __future__ import annotations

import numpy as np


def _ensure_array(inputs: np.ndarray) -> np.ndarray:
    """Return a contiguous float32 numpy array view of the input."""
    array = np.asarray(inputs, dtype=np.float32)
    if not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array, dtype=np.float32)
    return array


def bbox_overlaps(boxes: np.ndarray, query_boxes: np.ndarray) -> np.ndarray:
    """Compute IoU overlaps between two sets of boxes."""
    boxes = _ensure_array(boxes)
    query_boxes = _ensure_array(query_boxes)

    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float32)
    if N == 0 or K == 0:
        return overlaps

    box_areas = (boxes[:, 2] - boxes[:, 0] + 1.0) * (boxes[:, 3] - boxes[:, 1] + 1.0)
    for k in range(K):
        query = query_boxes[k]
        iw = np.minimum(boxes[:, 2], query[2]) - np.maximum(boxes[:, 0], query[0]) + 1.0
        ih = np.minimum(boxes[:, 3], query[3]) - np.maximum(boxes[:, 1], query[1]) + 1.0

        valid = (iw > 0.0) & (ih > 0.0)
        if not np.any(valid):
            continue

        inter = iw[valid] * ih[valid]
        query_area = (query[2] - query[0] + 1.0) * (query[3] - query[1] + 1.0)
        union = box_areas[valid] + query_area - inter
        overlaps[valid, k] = inter / np.clip(union, a_min=1e-6, a_max=None)

    return overlaps


def bbox_intersections(boxes: np.ndarray, query_boxes: np.ndarray) -> np.ndarray:
    """Ratio of each query box covered by the reference boxes."""
    boxes = _ensure_array(boxes)
    query_boxes = _ensure_array(query_boxes)

    N = boxes.shape[0]
    K = query_boxes.shape[0]
    intersections = np.zeros((N, K), dtype=np.float32)
    if N == 0 or K == 0:
        return intersections

    query_areas = (query_boxes[:, 2] - query_boxes[:, 0] + 1.0) * (
        query_boxes[:, 3] - query_boxes[:, 1] + 1.0
    )

    for k in range(K):
        query = query_boxes[k]
        iw = np.minimum(boxes[:, 2], query[2]) - np.maximum(boxes[:, 0], query[0]) + 1.0
        ih = np.minimum(boxes[:, 3], query[3]) - np.maximum(boxes[:, 1], query[1]) + 1.0

        valid = (iw > 0.0) & (ih > 0.0)
        if not np.any(valid):
            continue

        intersections[valid, k] = (iw[valid] * ih[valid]) / np.clip(
            query_areas[k], a_min=1e-6, a_max=None
        )

    return intersections


def box_vote(dets_nms: np.ndarray, dets_all: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Bounding box voting as described in the original extension."""
    dets_nms = _ensure_array(dets_nms)
    dets_all = _ensure_array(dets_all)

    num = dets_nms.shape[0]
    voted = np.zeros_like(dets_nms, dtype=np.float32)
    if num == 0 or dets_all.shape[0] == 0:
        return voted

    scores = dets_all[:, 4]
    boxes_all = dets_all[:, :4]

    for i in range(num):
        det = dets_nms[i]
        box = det[:4]

        iw = np.minimum(boxes_all[:, 2], box[2]) - np.maximum(boxes_all[:, 0], box[0]) + 1.0
        ih = np.minimum(boxes_all[:, 3], box[3]) - np.maximum(boxes_all[:, 1], box[1]) + 1.0
        inter = np.maximum(iw, 0.0) * np.maximum(ih, 0.0)

        area_det = (box[2] - box[0] + 1.0) * (box[3] - box[1] + 1.0)
        area_all = (boxes_all[:, 2] - boxes_all[:, 0] + 1.0) * (
            boxes_all[:, 3] - boxes_all[:, 1] + 1.0
        )
        union = area_det + area_all - inter

        iou = np.zeros_like(inter)
        valid = union > 0.0
        iou[valid] = inter[valid] / union[valid]

        matches = iou >= threshold
        if not np.any(matches):
            voted[i] = det
            continue

        weights = scores[matches]
        weighted_boxes = boxes_all[matches] * weights[:, None]
        total_weight = np.sum(weights)
        if total_weight <= 0.0:
            voted[i] = det
            continue

        fused_box = np.sum(weighted_boxes, axis=0) / total_weight
        voted[i, :4] = fused_box
        voted[i, 4] = det[4]

    return voted
