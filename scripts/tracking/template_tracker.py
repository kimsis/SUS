"""
Template matching tracker.

MOG2 detects foreground blobs each frame.  New blobs seed tracks by storing
a BGR template patch cropped from the detected object.  Each subsequent frame
the tracker searches a padded region around the last known position using
normalised cross-correlation (cv2.TM_CCOEFF_NORMED) and moves the track to
the best-matching location.  The template is refreshed periodically on
confident matches so the tracker adapts to gradual appearance changes.

Classical approach — simple, no deep learning, works well on objects with
distinctive texture.  Struggles with heavy occlusion or motion faster than
the search window.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import cv2
import numpy as np

from scripts.tracking.tracker_base import Tracker, TrackResult, track_color, clean_mask


@dataclass
class _TemplateTrack:
    id: int
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    template: np.ndarray             # BGR patch
    hits: int = 1
    miss: int = 0
    age: int = 0
    trajectory: list[tuple[int, int]] = field(default_factory=list)
    refresh_count: int = 0


class TemplateTracker(Tracker):
    """
    Template matching tracker.

    Args:
        min_blob_area:    Minimum foreground contour area to seed a track.
        search_pad:       Pixels to expand the search region around the last
                          known bounding box on each side.
        match_threshold:  Minimum TM_CCOEFF_NORMED score to accept a match.
        max_miss:         Frames without a confident match before dropping track.
        min_hits:         Minimum hits before a track is shown.
        iou_threshold:    Maximum IoU with existing tracks to allow spawning.
        template_refresh: Re-capture the template every N successful hits.
        max_tracks:       Hard cap on simultaneous active tracks.
        max_trajectory:   Maximum stored centre-points per track.
        warmup_frames:    Suppress output while MOG2 builds its background model.
    """

    def __init__(
        self,
        min_blob_area: int = 3000,
        max_blob_area: int = 30000,
        min_side: int = 50,
        max_side: int = 180,
        search_pad: int = 60,
        match_threshold: float = 0.45,
        max_miss: int = 5,
        min_hits: int = 2,
        iou_threshold: float = 0.3,
        suppress_iou: float = 0.4,
        template_refresh: int = 15,
        max_tracks: int = 30,
        max_trajectory: int = 80,
        warmup_frames: int = 25,
    ):
        self._min_blob_area = min_blob_area
        self._max_blob_area = max_blob_area
        self._min_side = min_side
        self._max_side = max_side
        self._search_pad = search_pad
        self._match_threshold = match_threshold
        self._max_miss = max_miss
        self._min_hits = min_hits
        self._iou_threshold = iou_threshold
        self._suppress_iou = suppress_iou
        self._template_refresh = template_refresh
        self._max_tracks = max_tracks
        self._max_trajectory = max_trajectory
        self._warmup_frames = warmup_frames

        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=25, varThreshold=10, detectShadows=False
        )
        self._tracks: list[_TemplateTrack] = []
        self._next_id: int = 0
        self._frame_count: int = 0

        self._viz: np.ndarray | None = None
        self._mask: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "Template Matching"

    def reset(self):
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=25, varThreshold=10, detectShadows=False
        )
        self._tracks = []
        self._next_id = 0
        self._frame_count = 0
        self._viz = None
        self._mask = None

    def get_viz(self) -> np.ndarray | None:
        return self._viz

    def get_mask(self) -> np.ndarray | None:
        return self._mask

    def update(self, frame: np.ndarray) -> list[TrackResult]:
        self._frame_count += 1
        h, w = frame.shape[:2]

        fg_mask = self._mog2.apply(frame)
        fg_mask = clean_mask(fg_mask)
        self._mask = fg_mask

        if self._frame_count <= self._warmup_frames:
            return []

        viz = frame.copy()

        # --- Update existing tracks via template matching ---
        live_tracks = []
        for t in self._tracks:
            x1, y1, x2, y2 = t.bbox
            tw, th = x2 - x1, y2 - y1

            sx1 = max(0, x1 - self._search_pad)
            sy1 = max(0, y1 - self._search_pad)
            sx2 = min(w, x2 + self._search_pad)
            sy2 = min(h, y2 + self._search_pad)

            search_img = frame[sy1:sy2, sx1:sx2]

            matched = False
            if search_img.shape[0] > th and search_img.shape[1] > tw:
                result = cv2.matchTemplate(search_img, t.template, cv2.TM_CCOEFF_NORMED)
                _, score, _, max_loc = cv2.minMaxLoc(result)

                if score >= self._match_threshold:
                    nx1 = sx1 + max_loc[0]
                    ny1 = sy1 + max_loc[1]
                    t.bbox = (nx1, ny1, nx1 + tw, ny1 + th)
                    t.hits += 1
                    t.miss = 0
                    t.age += 1
                    t.refresh_count += 1
                    if t.refresh_count >= self._template_refresh:
                        roi = frame[ny1:ny1 + th, nx1:nx1 + tw]
                        if roi.size > 0:
                            t.template = roi.copy()
                            t.refresh_count = 0
                    matched = True

            if not matched:
                t.miss += 1

            cv2.rectangle(viz, (sx1, sy1), (sx2, sy2), track_color(t.id), 1)

            if t.miss <= self._max_miss:
                live_tracks.append(t)

        self._tracks = live_tracks

        # --- Detect new blobs and seed tracks ---
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
            if len(self._tracks) >= self._max_tracks:
                break
            area = cv2.contourArea(cnt)
            if area < self._min_blob_area:
                break
            if area > self._max_blob_area:
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bw < self._min_side or bh < self._min_side:
                continue
            if bw > self._max_side or bh > self._max_side:
                continue
            if not self._is_new(bx, by, bx + bw, by + bh):
                continue
            roi = frame[by:by + bh, bx:bx + bw]
            if roi.size == 0:
                continue
            self._tracks.append(_TemplateTrack(
                id=self._next_id,
                bbox=(bx, by, bx + bw, by + bh),
                template=roi.copy(),
            ))
            self._next_id += 1

        self._viz = viz

        # --- Suppress duplicate tracks that heavily overlap ---
        self._suppress_overlapping()

        # --- Compose TrackResults (confirmed tracks only) ---
        results = []
        for t in self._tracks:
            if t.hits < self._min_hits and t.miss > 0:
                continue
            x1, y1, x2, y2 = t.bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            t.trajectory.append((cx, cy))
            if len(t.trajectory) > self._max_trajectory:
                t.trajectory.pop(0)
            results.append(TrackResult(
                id=t.id,
                bbox=(x1, y1, x2, y2),
                trajectory=list(t.trajectory),
            ))
        return results


    def _suppress_overlapping(self):
        """Greedy NMS: keep more-established tracks, drop weaker duplicates."""
        sorted_tracks = sorted(self._tracks, key=lambda t: t.hits, reverse=True)
        kept = []
        for candidate in sorted_tracks:
            for keeper in kept:
                if _iou(candidate.bbox, keeper.bbox) >= self._suppress_iou:
                    break
            else:
                kept.append(candidate)
        self._tracks = kept

    def _is_new(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        for t in self._tracks:
            if _iou((x1, y1, x2, y2), t.bbox) >= self._iou_threshold:
                return False
        return True


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    iou = inter / union if union > 0 else 0.0
    # Also catch containment: one box fully inside the other scores 1.0.
    iou_min = inter / min(area_a, area_b) if min(area_a, area_b) > 0 else 0.0
    return max(iou, iou_min)
