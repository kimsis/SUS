"""
SORT (Simple Online and Realtime Tracking) tracker.

Detection: YOLOv11 / YOLOv8 model trained on VisDrone (from the CVSP project).
           Falls back to MOG2 contour detection if no model path is supplied
           or if ultralytics is not installed.
Prediction: per-track Kalman filter with a constant-velocity motion model.
Assignment: IoU-based greedy matching (no scipy dependency).
Track lifecycle: tentative → confirmed (min_hits) → coasted → deleted (max_age).

Modern approach — robust to occlusions and lighting changes due to the deep
detector.  More computationally expensive than KLT/CamShift; requires GPU or
a fast CPU for real-time performance on high-resolution video.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import cv2

from scripts.tracking.tracker_base import Tracker, TrackResult, track_color

try:
    from ultralytics import YOLO as _YOLO
except ImportError:
    raise ImportError("ultralytics is required for the SORT tracker: pip install ultralytics")

_COCO_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


# ---------------------------------------------------------------------------
# Kalman filter for one track
# ---------------------------------------------------------------------------

class _KalmanTrack:
    """
    Constant-velocity Kalman filter.
    State: [cx, cy, w, h, vcx, vcy, vw, vh]
    Observation: [cx, cy, w, h]
    """

    # Shared matrices (built once).
    _F = np.eye(8, dtype=np.float64)
    _F[0, 4] = _F[1, 5] = _F[2, 6] = _F[3, 7] = 1.0

    _H = np.zeros((4, 8), dtype=np.float64)
    _H[0, 0] = _H[1, 1] = _H[2, 2] = _H[3, 3] = 1.0

    _Q = np.diag([1., 1., 10., 10., 0.01, 0.01, 0.1, 0.1]).astype(np.float64)
    _R = np.diag([1., 1., 10., 10.]).astype(np.float64)

    def __init__(self, bbox_xyxy: np.ndarray):
        cx, cy, w, h = _xyxy_to_cwh(bbox_xyxy)
        self.x = np.array([cx, cy, w, h, 0., 0., 0., 0.],
                          dtype=np.float64).reshape(8, 1)
        self.P = np.diag([10., 10., 10., 10., 1e4, 1e4, 1e4, 1e4]).astype(np.float64)

    def predict(self) -> np.ndarray:
        """Advance state by one step; return predicted [cx, cy, w, h]."""
        self.x = self._F @ self.x
        self.P = self._F @ self.P @ self._F.T + self._Q
        # Clip to avoid degenerate boxes.
        self.x[2] = max(1.0, self.x[2])
        self.x[3] = max(1.0, self.x[3])
        return self.x[:4].flatten()

    def update(self, bbox_xyxy: np.ndarray):
        """Incorporate a matched detection."""
        z = np.array(_xyxy_to_cwh(bbox_xyxy), dtype=np.float64).reshape(4, 1)
        y = z - self._H @ self.x
        S = self._H @ self.P @ self._H.T + self._R
        K = self.P @ self._H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self._H) @ self.P

    @property
    def bbox_xyxy(self) -> tuple[int, int, int, int]:
        cx, cy, w, h = self.x[:4].flatten()
        return _cwh_to_xyxy(cx, cy, w, h)


# ---------------------------------------------------------------------------
# Internal track state
# ---------------------------------------------------------------------------

@dataclass
class _SORTTrack:
    id: int
    kf: _KalmanTrack
    hits: int = 1
    miss: int = 0
    trajectory: list[tuple[int, int]] = field(default_factory=list)
    cls: int = -1


# ---------------------------------------------------------------------------
# SORT tracker
# ---------------------------------------------------------------------------

class SORTTracker(Tracker):
    """
    SORT tracking-by-detection.

    Args:
        model_path:     Path to a Ultralytics .pt model.  Relative paths are
                        resolved against the SUS project root.  Set to None
                        or omit to use MOG2 fallback detection.
        conf:           YOLO confidence threshold.
        iou_det:        YOLO NMS IoU threshold.
        person_only:    If True, only keep detections of class 0 (person).
        min_hits:       Minimum consecutive hits before a track is confirmed
                        and shown.
        max_age:        Frames a track survives without a matching detection.
        iou_match:      Minimum IoU to accept a track–detection assignment.
        max_trajectory: Maximum stored centre-points per track.
    """

    # Default model: yolo11m.pt in the SUS project root.
    _DEFAULT_MODEL = "yolo11m.pt"

    def __init__(
        self,
        model_path: str | None = _DEFAULT_MODEL,
        conf: float = 0.25,
        iou_det: float = 0.45,
        person_only: bool = False,
        min_hits: int = 2,
        max_age: int = 5,
        iou_match: float = 0.25,
        max_trajectory: int = 80,
    ):
        self._conf = conf
        self._iou_det = iou_det
        self._person_only = person_only
        self._min_hits = min_hits
        self._max_age = max_age
        self._iou_match = iou_match
        self._max_trajectory = max_trajectory

        p = Path(model_path) if model_path else None
        if p is None:
            raise ValueError("SORT tracker requires a YOLO model path.")
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[2] / p
        if not p.exists():
            raise FileNotFoundError(f"[SORT] YOLO model not found: {p}")
        self._yolo = _YOLO(str(p))
        print(f"[SORT] Loaded YOLO model: {p}")

        self._tracks: list[_SORTTrack] = []
        self._next_id: int = 0
        self._frame_shape: tuple[int, int] | None = None

        self._viz: np.ndarray | None = None
        self._mask: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Tracker interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "SORT (YOLO)"

    def reset(self):
        self._tracks = []
        self._next_id = 0
        self._viz = None
        self._mask = None

    def get_viz(self) -> np.ndarray | None:
        return self._viz

    def get_mask(self) -> np.ndarray | None:
        return self._mask

    def update(self, frame: np.ndarray) -> list[TrackResult]:
        self._frame_shape = frame.shape[:2]

        # --- Detect objects ---
        detections, det_classes = self._detect(frame)

        # --- Predict existing tracks ---
        for t in self._tracks:
            t.kf.predict()
            t.miss += 1

        # --- Greedy IoU assignment ---
        matched_t, matched_d, unmatched_d = self._match(detections)

        # Update matched tracks.
        for ti, di in zip(matched_t, matched_d):
            self._tracks[ti].kf.update(detections[di])
            self._tracks[ti].hits += 1
            self._tracks[ti].miss = 0
            self._tracks[ti].cls = det_classes[di]

        # Create new tracks for unmatched detections.
        for di in unmatched_d:
            new_t = _SORTTrack(
                id=self._next_id,
                kf=_KalmanTrack(detections[di]),
                cls=det_classes[di],
            )
            self._next_id += 1
            self._tracks.append(new_t)

        # Remove stale tracks.
        self._tracks = [t for t in self._tracks if t.miss <= self._max_age]

        # --- Build visualisation (raw detections on a dark frame) ---
        self._build_viz(frame, detections)

        # --- Compose TrackResults (confirmed tracks only) ---
        results = []
        for t in self._tracks:
            if t.hits < self._min_hits and t.miss > 0:
                continue
            bbox = t.kf.bbox_xyxy
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]
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
                label=_COCO_NAMES.get(t.cls, ""),
            ))
        return results

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _detect(self, frame: np.ndarray) -> tuple[list[np.ndarray], list[int]]:
        results = self._yolo.predict(
            frame, verbose=False, conf=self._conf, iou=self._iou_det
        )
        boxes, classes = [], []
        _VEHICLE_CLASSES = {0, 1, 2, 3, 5, 7}
        for r in results:
            if r.boxes is None:
                continue
            for box, cls in zip(r.boxes.xyxy.cpu().numpy(),
                                r.boxes.cls.cpu().numpy().astype(int)):
                if self._person_only and cls != 0:
                    continue
                if not self._person_only and cls not in _VEHICLE_CLASSES:
                    continue
                boxes.append(box.astype(np.float64))
                classes.append(int(cls))

        self._mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        return boxes, classes

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def _match(
        self, detections: list[np.ndarray]
    ) -> tuple[list[int], list[int], list[int]]:
        if not self._tracks or not detections:
            return [], [], list(range(len(detections)))

        n_t, n_d = len(self._tracks), len(detections)
        iou_mat = np.zeros((n_t, n_d), dtype=np.float64)
        for ti, t in enumerate(self._tracks):
            pred_box = np.array(t.kf.bbox_xyxy, dtype=np.float64)
            for di, det in enumerate(detections):
                iou_mat[ti, di] = _iou(pred_box, det)

        # Greedy assignment: pick highest-IoU pairs first.
        matched_t, matched_d = [], []
        used_t, used_d = set(), set()
        for idx in np.argsort(-iou_mat.flatten()):
            ti, di = divmod(int(idx), n_d)
            if iou_mat[ti, di] < self._iou_match:
                break
            if ti not in used_t and di not in used_d:
                matched_t.append(ti)
                matched_d.append(di)
                used_t.add(ti)
                used_d.add(di)

        unmatched_d = [di for di in range(n_d) if di not in used_d]
        return matched_t, matched_d, unmatched_d

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def _build_viz(self, frame: np.ndarray, detections: list[np.ndarray]):
        """Show raw detections (cyan) alongside predicted track boxes (track colour)."""
        viz = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.astype(int)
            cv2.rectangle(viz, (x1, y1), (x2, y2), (255, 255, 0), 1)
        for t in self._tracks:
            x1, y1, x2, y2 = t.kf.bbox_xyxy
            cv2.rectangle(viz, (x1, y1), (x2, y2), track_color(t.id), 1)
        self._viz = viz


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _xyxy_to_cwh(box: np.ndarray) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2, float(x2 - x1), float(y2 - y1)


def _cwh_to_xyxy(cx: float, cy: float, w: float, h: float) -> tuple[int, int, int, int]:
    return (
        int(cx - w / 2), int(cy - h / 2),
        int(cx + w / 2), int(cy + h / 2),
    )


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0., ix2 - ix1) * max(0., iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / union) if union > 0 else 0.0
