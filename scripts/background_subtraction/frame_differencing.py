import cv2
import numpy as np
from scripts.background_subtraction.bg_subtractor_base import BackgroundSubtractor


class FrameDifferencing(BackgroundSubtractor):
    """
    Frame Differencing background subtraction.

    Computes the absolute difference between the current frame and the
    previous frame, then thresholds the result to produce a foreground mask.

    Advantages:
      - Extremely fast, minimal memory footprint.
      - Adapts instantly to lighting changes since there is no long-term model.
      - Simple to understand and tune.

    Disadvantages:
      - Detects only moving edges, not the full extent of moving objects
        (double-ghosting / hollow object problem).
      - Cannot detect objects that stop moving.
      - Highly sensitive to camera shake and noise.
      - No background model is built, so a stationary camera is assumed.

    Args:
        threshold:      Pixel-difference value above which a pixel is
                        considered foreground (0–255). Higher = less noise,
                        but misses subtle motion.
        blur_kernel:    Size of the Gaussian blur kernel applied before
                        differencing.  Must be odd. Larger = more noise
                        suppression but blurrier edges.
        dilate_iters:   Number of dilation iterations applied to the mask to
                        close small gaps between moving edges.
    """

    def __init__(self, threshold: int = 40, blur_kernel: int = 7, dilate_iters: int = 2):
        self.threshold = threshold
        self.blur_kernel = blur_kernel
        self.dilate_iters = dilate_iters
        self._prev_gray: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "Frame Differencing"

    def reset(self):
        self._prev_gray = None

    def apply(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)

        if self._prev_gray is None:
            # First frame — nothing to diff against yet, return empty mask.
            self._prev_gray = gray
            return np.zeros(gray.shape, dtype=np.uint8)

        diff = cv2.absdiff(self._prev_gray, gray)
        self._prev_gray = gray

        _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)

        if self.dilate_iters > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.dilate(mask, kernel, iterations=self.dilate_iters)

        return mask
