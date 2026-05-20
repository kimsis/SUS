from abc import ABC, abstractmethod
import numpy as np


class BackgroundSubtractor(ABC):
    """
    Abstract base class for all background subtraction algorithms.
    Each algorithm must implement apply() and reset().
    apply() receives a raw BGR frame and returns a binary foreground mask.
    """

    @abstractmethod
    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame.

        Args:
            frame: BGR image as a numpy array (H, W, 3), uint8.

        Returns:
            Binary foreground mask (H, W), uint8, values 0 or 255.
        """

    @abstractmethod
    def reset(self):
        """Reset all internal state (call between video files)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable algorithm name shown in window titles and logs."""
