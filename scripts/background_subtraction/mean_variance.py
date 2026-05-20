import cv2
import numpy as np
from scripts.background_subtraction.bg_subtractor_base import BackgroundSubtractor


class MeanVariance(BackgroundSubtractor):
    """
    Mean and Variance (Adaptive Gaussian) background subtraction.

    Maintains a per-pixel running estimate of the background mean (μ) and
    variance (σ²) using an exponential moving average.  A pixel is classified
    as foreground when it deviates from the background model by more than
    `k` standard deviations.

    The model is updated only for pixels classified as background, preventing
    detected foreground objects from contaminating the background estimate.

    Advantages:
      - Full object silhouettes are detected (not just moving edges).
      - Adapts gradually to slow illumination changes via the learning rate.
      - Stationary objects are eventually absorbed into the background model.
      - More robust to noise than frame differencing.

    Disadvantages:
      - Requires a warm-up period (first `warmup_frames` frames) before the
        model stabilises — produces noisy output in the meantime.
      - Slow adaptation means sudden, large lighting changes cause large
        false-positive regions until the model catches up.
      - A single Gaussian per pixel cannot model bimodal backgrounds
        (e.g. waving trees, rippling water) — use MoG for those scenes.
      - Higher memory usage than frame differencing (two float32 arrays).

    Args:
        learning_rate:  α in the EMA update (0 < α < 1).  Smaller values
                        mean slower adaptation to background change.
        k:              Number of standard deviations a pixel must deviate
                        to be classified as foreground.  Higher = stricter.
        warmup_frames:  Number of frames used to initialise the model before
                        foreground detection begins.
        min_variance:   Floor on the per-pixel variance to avoid division by
                        zero and prevent over-sensitivity in flat regions.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        k: float = 2.5,
        warmup_frames: int = 30,
        min_variance: float = 16.0,
    ):
        self.learning_rate = learning_rate
        self.k = k
        self.warmup_frames = warmup_frames
        self.min_variance = min_variance

        self._mean: np.ndarray | None = None
        self._variance: np.ndarray | None = None
        self._frame_count: int = 0

    @property
    def name(self) -> str:
        return "Mean and Variance"

    def reset(self):
        self._mean = None
        self._variance = None
        self._frame_count = 0

    def apply(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if self._mean is None:
            # Initialise the background model on the first frame.
            self._mean = gray.copy()
            self._variance = np.full(gray.shape, 64.0, dtype=np.float32)
            self._frame_count = 1
            return np.zeros(gray.shape, dtype=np.uint8)

        self._frame_count += 1

        # During warmup, simply update the model and return an empty mask.
        if self._frame_count <= self.warmup_frames:
            self._mean = (1 - self.learning_rate) * self._mean + self.learning_rate * gray
            diff_sq = (gray - self._mean) ** 2
            self._variance = (
                (1 - self.learning_rate) * self._variance + self.learning_rate * diff_sq
            )
            return np.zeros(gray.shape, dtype=np.uint8)

        # Classify pixels as foreground.
        std = np.sqrt(np.maximum(self._variance, self.min_variance))
        fg_mask = np.abs(gray - self._mean) > (self.k * std)

        # Update the model only for background pixels.
        bg_mask = ~fg_mask
        alpha = self.learning_rate
        self._mean[bg_mask] = (
            (1 - alpha) * self._mean[bg_mask] + alpha * gray[bg_mask]
        )
        diff_sq = (gray - self._mean) ** 2
        self._variance[bg_mask] = (
            (1 - alpha) * self._variance[bg_mask] + alpha * diff_sq[bg_mask]
        )

        # Convert boolean mask to uint8 (0 / 255).
        mask = fg_mask.astype(np.uint8) * 255

        # Morphological cleanup: remove small noise blobs, fill holes.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask
