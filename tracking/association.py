# tracking/association.py
from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
except Exception:
    linear_sum_assignment = None


def _center_from_bbox(b):
    x0, y0, x1, y1 = b
    return (0.5 * (x0 + x1), 0.5 * (y0 + y1))


def assign_tracks_to_detections(
    track_centers: List[Tuple[float, float]],
    det_centers: List[Tuple[float, float]],
    max_dist: float,
):
    """
    반환:
      matches: List[(ti, di)]
      unmatched_tracks: List[ti]
      unmatched_dets: List[di]
    """
    nT = len(track_centers)
    nD = len(det_centers)
    if nT == 0:
        return [], [], list(range(nD))
    if nD == 0:
        return [], list(range(nT)), []

    cost = np.zeros((nT, nD), dtype=np.float32)
    for i, (tx, ty) in enumerate(track_centers):
        for j, (dx, dy) in enumerate(det_centers):
            dist = ((tx - dx) ** 2 + (ty - dy) ** 2) ** 0.5
            cost[i, j] = dist

    # Hungarian이 있으면 사용, 없으면 greedy
    if linear_sum_assignment is not None:
        ti, dj = linear_sum_assignment(cost)
        matches = []
        used_t = set()
        used_d = set()
        for a, b in zip(ti.tolist(), dj.tolist()):
            if cost[a, b] <= max_dist:
                matches.append((a, b))
                used_t.add(a)
                used_d.add(b)
        unmatched_tracks = [i for i in range(nT) if i not in used_t]
        unmatched_dets = [j for j in range(nD) if j not in used_d]
        return matches, unmatched_tracks, unmatched_dets

    # greedy fallback
    pairs = []
    for i in range(nT):
        for j in range(nD):
            pairs.append((float(cost[i, j]), i, j))
    pairs.sort(key=lambda x: x[0])

    matches = []
    used_t = set()
    used_d = set()
    for dist, i, j in pairs:
        if dist > max_dist:
            break
        if i in used_t or j in used_d:
            continue
        matches.append((i, j))
        used_t.add(i)
        used_d.add(j)

    unmatched_tracks = [i for i in range(nT) if i not in used_t]
    unmatched_dets = [j for j in range(nD) if j not in used_d]
    return matches, unmatched_tracks, unmatched_dets
