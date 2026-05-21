from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import cv2
import numpy as np


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """
    Standard MOG2 mask cleanup shared by all trackers.
    Mirrors the pre/post processing from FrameDifferencing:
    1. GaussianBlur  — smooths noisy edges (pre-processing equivalent).
    2. Re-threshold  — restores a clean binary mask after blurring.
    3. Dilate        — connects nearby blobs into solid object silhouettes.
    """
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


# Deterministic per-ID color palette (BGR).
_PALETTE = [
    (0,   200,  255), (0,   255,  80),  (255, 60,   60),  (255, 200,  0),
    (180, 0,    255), (0,   160,  255), (255, 120,  0),   (0,   220, 150),
    (200, 0,    180), (80,  255,  0),   (0,   100,  255), (255, 0,    120),
    (0,   255, 200),  (255, 80,   200), (100, 200,  255), (200, 255,  0),
]


def track_color(track_id: int) -> tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


@dataclass
class TrackResult:
    """Unified output for a single tracked object in one frame."""
    id: int
    # Axis-aligned bounding box: (x1, y1, x2, y2) in pixel coordinates.
    bbox: tuple[int, int, int, int]
    # Center-point history for this track (most recent last).
    trajectory: list[tuple[int, int]] = field(default_factory=list)
    # Human-readable class label (empty string if unknown).
    label: str = ""

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) // 2, (y1 + y2) // 2

    @property
    def color(self) -> tuple[int, int, int]:
        return track_color(self.id)


class Tracker(ABC):
    """
    Abstract base class for all tracking algorithms.

    Subclasses must implement update(), reset(), and the name property.
    Optionally override get_viz() and get_mask() to provide per-frame
    visualisations used by the 2×2 display in run_tracking.py.
    """

    @abstractmethod
    def update(self, frame: np.ndarray) -> list[TrackResult]:
        """
        Process a single BGR frame and return all active tracks.

        Args:
            frame: BGR image, uint8, shape (H, W, 3).

        Returns:
            List of TrackResult, one per currently active track.
        """

    def get_viz(self) -> np.ndarray | None:
        """
        Algorithm-specific visualisation (TOP-RIGHT panel).

        Examples: sparse optical-flow arrows (KLT), back-projection
        probability map (CamShift), raw YOLO detections (SORT).

        Returns a BGR image the same size as the last processed frame,
        or None if not implemented.
        """
        return None

    def get_mask(self) -> np.ndarray | None:
        """
        Internal motion / detection mask (BOTTOM-LEFT panel).

        Returns a single-channel uint8 mask (values 0 or 255),
        or None if not implemented.
        """
        return None

    @abstractmethod
    def reset(self):
        """Reset all internal state (call between video files)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable algorithm name for window titles and logs."""
