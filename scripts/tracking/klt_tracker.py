"""
KLT (Kanade-Lucas-Tomasi) optical-flow tracker.

Feature points (Shi-Tomasi corners) are detected inside MOG2 foreground
blobs and tracked frame-to-frame with the Lucas-Kanade pyramid tracker.
Nearby surviving points are clustered into bounding boxes; each cluster
is one track.  Track IDs are maintained across frames by matching cluster
centres with a simple nearest-neighbour step.

Classical approach — fast, no deep learning, works well for rigid objects
with distinct texture.  Struggles when the camera moves significantly or
objects overlap (points from two objects merge into one cluster).
"""

from dataclasses import dataclass, field
import cv2
import numpy as np

from scripts.tracking.tracker_base import Tracker, TrackResult, track_color, clean_mask


# Lucas-Kanade pyramid parameters.
_LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
)

# Shi-Tomasi feature-detection parameters.
_FEATURE_PARAMS = dict(
    maxCorners=300,
    qualityLevel=0.01,
    minDistance=7,
    blockSize=7,
)


# ---------------------------------------------------------------------------
# Internal per-track state
# ---------------------------------------------------------------------------

@dataclass
class _Track:
    id: int
    # All feature points belonging to this track, shape (N, 2) float32.
    points: np.ndarray
    trajectory: list[tuple[int, int]] = field(default_factory=list)
    age: int = 0

    @property
    def center(self) -> tuple[int, int]:
        m = self.points.mean(axis=0)
        return int(m[0]), int(m[1])

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        pad = 15
        x1 = int(self.points[:, 0].min()) - pad
        y1 = int(self.points[:, 1].min()) - pad
        x2 = int(self.points[:, 0].max()) + pad
        y2 = int(self.points[:, 1].max()) + pad
        return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# KLT tracker
# ---------------------------------------------------------------------------

class KLTTracker(Tracker):
    """
    KLT tracker: Shi-Tomasi detection + Lucas-Kanade optical flow.

    Args:
        cluster_radius:    Pixels within this distance are merged into one
                           cluster (one tracked object).
        min_cluster_pts:   Minimum number of surviving points to keep a track.
        redetect_every:    Re-run feature detection every N frames to replace
                           lost points and pick up newly appearing objects.
        min_blob_area:     Minimum foreground contour area (px²) to seed a
                           new track from.
        max_trajectory:    Maximum number of centre-points stored per track.
    """

    def __init__(
        self,
        cluster_radius: int = 45,
        min_cluster_pts: int = 3,
        redetect_every: int = 15,
        min_blob_area: int = 1500,
        min_side: int = 25,
        max_trajectory: int = 60,
        warmup_frames: int = 25,
    ):
        self._cluster_radius = cluster_radius
        self._min_cluster_pts = min_cluster_pts
        self._redetect_every = redetect_every
        self._min_blob_area = min_blob_area
        self._min_side = min_side
        self._max_trajectory = max_trajectory
        self._warmup_frames = warmup_frames

        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=25, varThreshold=5, detectShadows=False
        )
        self._prev_gray: np.ndarray | None = None
        self._tracks: list[_Track] = []
        self._next_id: int = 0
        self._frame_count: int = 0

        # Outputs written each frame for get_viz() / get_mask().
        self._viz: np.ndarray | None = None
        self._mask: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Tracker interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "KLT Optical Flow"

    def reset(self):
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=25, varThreshold=5, detectShadows=False
        )
        self._prev_gray = None
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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_mask = self._mog2.apply(frame)
        fg_mask = clean_mask(fg_mask)
        self._mask = fg_mask

        if self._frame_count <= self._warmup_frames:
            self._prev_gray = gray
            return []

        # --- Track existing points with LK ---
        if self._prev_gray is not None and self._tracks:
            self._lk_update(gray)
            # Drop tracks that have drifted off a real moving object.
            self._tracks = [t for t in self._tracks
                            if self._has_foreground_support(t, fg_mask)]

        # --- Periodically (re-)detect features in unoccupied foreground ---
        if self._frame_count % self._redetect_every == 1 or not self._tracks:
            self._detect_new(gray, fg_mask, frame.shape[:2])

        # --- Build flow-vector visualisation ---
        self._build_viz(frame)

        self._prev_gray = gray

        # --- Compose TrackResults ---
        results = []
        for t in self._tracks:
            t.trajectory.append(t.center)
            if len(t.trajectory) > self._max_trajectory:
                t.trajectory.pop(0)
            x1, y1, x2, y2 = t.bbox
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            results.append(TrackResult(id=t.id, bbox=(x1, y1, x2, y2),
                                       trajectory=list(t.trajectory)))
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _lk_update(self, gray: np.ndarray):
        """Track all points forward one frame; drop lost points; remove empty tracks."""
        # Stack all points into one array for a single LK call (fast).
        all_pts = np.vstack([t.points for t in self._tracks]).reshape(-1, 1, 2)
        sizes = [len(t.points) for t in self._tracks]

        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, all_pts, None, **_LK_PARAMS
        )

        # Distribute results back to individual tracks.
        offset = 0
        live_tracks = []
        for t, sz in zip(self._tracks, sizes):
            s = status[offset: offset + sz].flatten()
            pts = new_pts[offset: offset + sz].reshape(-1, 2)
            offset += sz

            good = pts[s == 1]
            if len(good) >= self._min_cluster_pts:
                t.points = good.astype(np.float32)
                t.age += 1
                live_tracks.append(t)

        self._tracks = live_tracks

    def _detect_new(self, gray: np.ndarray, fg_mask: np.ndarray,
                    frame_size: tuple[int, int]):
        """Detect Shi-Tomasi corners in foreground blobs; spawn new tracks."""
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < self._min_blob_area:
                continue

            # Restrict feature detection to this contour's bounding box.
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bw < self._min_side or bh < self._min_side:
                continue
            roi_mask = np.zeros((bh, bw), np.uint8)
            shifted = cnt - np.array([[[bx, by]]])
            cv2.drawContours(roi_mask, [shifted], -1, 255, -1)

            new_pts = cv2.goodFeaturesToTrack(
                gray[by:by+bh, bx:bx+bw], mask=roi_mask, **_FEATURE_PARAMS
            )
            if new_pts is None:
                continue
            new_pts = new_pts.reshape(-1, 2) + np.array([bx, by], dtype=np.float32)

            # Skip if these points are already covered by an existing track.
            if self._overlap_with_existing(new_pts):
                continue

            track = _Track(id=self._next_id, points=new_pts.astype(np.float32))
            self._next_id += 1
            self._tracks.append(track)

    def _overlap_with_existing(self, pts: np.ndarray) -> bool:
        """Return True if the centre of pts is already inside a live track's bbox."""
        cx, cy = pts.mean(axis=0)
        for t in self._tracks:
            x1, y1, x2, y2 = t.bbox
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return True
        return False

    def _has_foreground_support(self, t: "_Track", fg_mask: np.ndarray,
                                min_fg_fraction: float = 0.04) -> bool:
        """Return True if the track's bbox contains enough foreground pixels."""
        fh, fw = fg_mask.shape
        x1, y1, x2, y2 = t.bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)
        if x2 <= x1 or y2 <= y1:
            return False
        roi = fg_mask[y1:y2, x1:x2]
        return roi.mean() >= min_fg_fraction * 255

    def _build_viz(self, frame: np.ndarray):
        """Draw sparse flow vectors for the TOP-RIGHT visualisation panel."""
        viz = frame.copy()
        if self._prev_gray is None:
            self._viz = viz
            return

        for t in self._tracks:
            color = track_color(t.id)
            for pt in t.points:
                cv2.circle(viz, (int(pt[0]), int(pt[1])), 2, color, -1)

        self._viz = viz
