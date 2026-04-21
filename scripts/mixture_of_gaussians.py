import cv2
import numpy as np
from scripts.bg_subtractor_base import BackgroundSubtractor


class MixtureOfGaussians(BackgroundSubtractor):
    """
    Mixture of Gaussians (MoG2) background subtraction.

    Wraps OpenCV's highly optimised MOG2 implementation (Zivkovic, 2004 /
    2006), which models each pixel's intensity history as a mixture of K
    Gaussians.  Gaussian components are dynamically added and removed as the
    scene changes, allowing the model to represent multimodal backgrounds
    (e.g. swaying trees, flickering screens, rippling water).

    Shadow detection is enabled by default: shadow pixels are labelled 127
    in the raw mask and can be treated as background or a separate class.

    Advantages:
      - Handles multimodal / dynamic backgrounds robustly.
      - Built-in adaptive shadow detection.
      - Highly optimised C++ backend — fast even at high resolution.
      - No manual warmup needed; the model bootstraps itself automatically.
      - Per-pixel learning rates adapt based on how well each Gaussian fits.

    Disadvantages:
      - More parameters than the simpler approaches; harder to reason about
        failure modes.
      - Can slowly absorb slow-moving foreground objects into the background
        if they remain stationary for many frames.
      - Shadow suppression may misclassify dark foreground pixels as shadows.
      - Slightly higher latency than frame differencing for the first few
        seconds while the mixture initialises.
      - Black-box compared to the hand-crafted mean/variance approach.

    Args:
        history:            Number of frames used to build the background
                            model.  Longer history = slower adaptation.
        var_threshold:      Mahalanobis distance threshold for classifying a
                            pixel as background.  Higher = more permissive
                            (less foreground detected).
        detect_shadows:     If True, shadow pixels are marked 127 and then
                            suppressed in the returned mask (treated as BG).
        learning_rate:      How quickly the model adapts (-1 = auto, which
                            is recommended and uses 1/history).
        morph_kernel_size:  Size of the elliptical structuring element used
                            in the opening + closing cleanup pass.
    """

    def __init__(
        self,
        history: int = 25,
        var_threshold: float = 5.0,
        detect_shadows: bool = False,
        learning_rate: float = -1,
        morph_kernel_size: int = 5,
    ):
        self.history = history
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows
        self.learning_rate = learning_rate
        self.morph_kernel_size = morph_kernel_size

        self._mog2 = self._build_mog2()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_mog2(self) -> cv2.BackgroundSubtractorMOG2:
        mog2 = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=self.detect_shadows,
        )
        return mog2

    # ------------------------------------------------------------------
    # BackgroundSubtractor interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "Mixture of Gaussians (MOG2)"

    def reset(self):
        """Rebuild the MOG2 object to clear all internal state."""
        self._mog2 = self._build_mog2()

    def apply(self, frame: np.ndarray) -> np.ndarray:
        raw_mask = self._mog2.apply(frame, learningRate=self.learning_rate)

        # Shadow pixels are labelled 127 — suppress them (treat as background).
        if self.detect_shadows:
            raw_mask[raw_mask == 127] = 0

        # Morphological cleanup: opening removes noise, closing fills holes.
        if self.morph_kernel_size > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.morph_kernel_size, self.morph_kernel_size),
            )
            raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
            raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)

        return raw_mask
