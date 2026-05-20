"""
Background subtraction pipeline.

Runs one or more background subtraction algorithms on a list of video sources
and displays / saves the results side-by-side with the original frame.

Usage examples
--------------
Single algorithm, live camera:
    python run_pipeline.py --algorithm frame_diff

All three algorithms on a video file:
    python run_pipeline.py --algorithms frame_diff,mean_variance,mog \
                           --video_path path/to/video.mp4

Multiple videos from a folder, save outputs:
    python run_pipeline.py --algorithm mog \
                           --video_folder path/to/folder \
                           --save_output True

Available algorithm names
-------------------------
    frame_diff      Frame Differencing
    mean_variance   Mean and Variance (Adaptive Gaussian)
    mog             Mixture of Gaussians (MOG2)
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from scripts.background_subtraction import BackgroundSubtractor, FrameDifferencing, MeanVariance, MixtureOfGaussians
from scripts.video_utils import draw_label, resize_for_display, get_video_sources


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

ALGORITHM_REGISTRY: dict[str, BackgroundSubtractor] = {
    "frame_diff":    FrameDifferencing(),
    "mean_variance": MeanVariance(),
    "mog":           MixtureOfGaussians(),
}


def build_algorithms(names: list[str]) -> list[BackgroundSubtractor]:
    """Return algorithm instances for the given list of names."""
    algorithms = []
    for name in names:
        key = name.strip().lower()
        if key not in ALGORITHM_REGISTRY:
            available = ", ".join(ALGORITHM_REGISTRY.keys())
            raise ValueError(f"Unknown algorithm '{key}'. Available: {available}")
        algorithms.append(ALGORITHM_REGISTRY[key])
    return algorithms


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def fill_mask(mask: np.ndarray) -> np.ndarray:
    """
    Fill holes inside the existing mask blobs using morphological closing.
    Works at 1/4 resolution for speed, then upscales back.
    """
    small = cv2.resize(mask, (mask.shape[1] // 4, mask.shape[0] // 4),
                       interpolation=cv2.INTER_NEAREST)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    filled = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cv2.resize(filled, (mask.shape[1], mask.shape[0]),
                      interpolation=cv2.INTER_NEAREST)


def overlay_contours(frame: np.ndarray, mask: np.ndarray, color: tuple = (0, 255, 0),
                     thickness: int = 2) -> np.ndarray:
    """
    Draw contours of the foreground mask on a copy of frame.
    Returns the annotated frame without modifying the original.
    """
    vis = frame.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 200]
    cv2.drawContours(vis, contours, -1, color, thickness)
    return vis


# ---------------------------------------------------------------------------
# In-place 2x2 display builder
# ---------------------------------------------------------------------------

def build_display_frame_inplace(
    canvas: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    filled: np.ndarray,
    algorithm_name: str,
    frame_idx: int,
) -> None:
    """
    Write the 2x2 grid directly into a pre-allocated canvas. No copies.

    Layout:
        TOP-LEFT:     Original frame with contour overlays
        TOP-RIGHT:    Raw foreground mask (colourised green)
        BOTTOM-LEFT:  Background only (moving objects blacked out)
        BOTTOM-RIGHT: Foreground objects only (background removed)
    """
    h, w = original.shape[:2]
    bar_h = 28

    def write_label(region, text):
        cv2.rectangle(region, (0, 0), (w, bar_h), (20, 20, 20), -1)
        cv2.putText(region, text, (8, bar_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    # Views into the 4 quadrants — no allocation, writes go directly to canvas
    top_left     = canvas[0:h,         0:w]
    top_right    = canvas[0:h,         w+3:w*2+3]
    bottom_left  = canvas[h+3:h*2+3,   0:w]
    bottom_right = canvas[h+3:h*2+3,   w+3:w*2+3]

    # TOP-LEFT: original + contours
    np.copyto(top_left, original)
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 200]
    cv2.drawContours(top_left, contours, -1, (0, 255, 0), 2)
    write_label(top_left, f"{algorithm_name}  |  frame {frame_idx}")

    # TOP-RIGHT: mask colourised
    top_right[:] = 30
    top_right[mask > 0] = (0, 220, 80)
    write_label(top_right, "Raw foreground mask")

    # BOTTOM-LEFT: background only (objects blacked out)
    np.copyto(bottom_left, original)
    bottom_left[filled > 0] = 0
    write_label(bottom_left, "Background (objects removed)")

    # BOTTOM-RIGHT: foreground objects only (background blacked out)
    bottom_right[:] = 0
    bottom_right[filled > 0] = original[filled > 0]
    write_label(bottom_right, "Foreground objects only")


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    algorithms: list[BackgroundSubtractor],
    video_sources: list,
    save_output: bool = False,
    output_dir: str | None = None,
    headless: bool = False,
    collect_metrics: bool = False,
) -> dict | None:
    """
    Run each background subtraction algorithm on every video source.

    Display layout (unless headless):
        TOP-LEFT:     Original + contours
        TOP-RIGHT:    Raw foreground mask
        BOTTOM-LEFT:  Background only
        BOTTOM-RIGHT: Foreground objects only

    Keyboard controls (when headless=False):
        n  — skip to next video
        q  — quit immediately

    Args:
        algorithms:       List of BackgroundSubtractor instances to run.
        video_sources:    List of video file paths or camera indices (int).
        save_output:      Write output videos to disk.
        output_dir:       Destination directory for saved videos.
        headless:         Process without opening any display windows.
        collect_metrics:  Accumulate and return performance metrics.

    Returns:
        Metrics dict if collect_metrics=True, otherwise None.
    """
    metrics = {
        "status": "SUCCESS",
        "algorithms": [],
    } if collect_metrics else None

    for algo in algorithms:
        print(f"\n{'='*60}")
        print(f"Algorithm: {algo.name}")
        print(f"{'='*60}")

        algo_metrics = {
            "algorithm": algo.name,
            "videos": [],
            "total_frames": 0,
            "total_time": 0.0,
            "avg_fps": 0.0,
        } if collect_metrics else None

        for i, video_source in enumerate(video_sources, 1):
            is_camera = isinstance(video_source, int)
            video_name = (
                Path(str(video_source)).name if not is_camera
                else f"Camera_{video_source}"
            )
            print(f"\n[{i}/{len(video_sources)}] {video_name}")

            # Reset the model between videos so state from one clip does not
            # bleed into the next.
            algo.reset()

            capture = cv2.VideoCapture(video_source)
            if not capture.isOpened():
                print(f"  Error: could not open {video_source}")
                continue

            src_fps = capture.get(cv2.CAP_PROP_FPS) or 30
            target_delay_ms = max(1, int(1000 / src_fps))

            # Read and validate the first frame before entering the loop.
            ret, first_frame = capture.read()
            if not ret or first_frame is None:
                print(f"  Error: could not read frames from {video_source}")
                capture.release()
                continue

            fh, fw = first_frame.shape[:2]
            if fh < 64 or fw < 64:
                print(f"  Error: implausibly small frame ({fw}x{fh}) — "
                      f"video source '{video_source}' likely cannot be opened.")
                capture.release()
                continue

            print(f"  Resolution: {fw}x{fh}  |  FPS: {src_fps:.1f}")

            # Pre-allocate the display canvas once per video
            canvas = np.zeros((fh * 2 + 3, fw * 2 + 3, 3), dtype=np.uint8)
            # Pre-draw static dividers
            canvas[fh:fh+3, :]  = 60   # horizontal divider
            canvas[:, fw:fw+3]  = 60   # vertical divider

            # Output video writer
            out_writer = None
            if save_output:
                out_path = _make_output_path(video_name, algo.name, output_dir)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_writer = cv2.VideoWriter(
                    str(out_path), fourcc, src_fps, (fw * 2 + 3, fh * 2 + 3)
                )
                print(f"  Saving to: {out_path}")

            # Fullscreen window
            window_title = f"{algo.name}  —  {video_name}"
            if not headless:
                cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(window_title, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)

            frame_count = 0
            video_start = time.time()
            pending_frame = first_frame

            while True:
                if pending_frame is not None:
                    frame = pending_frame
                    pending_frame = None
                else:
                    ret, frame = capture.read()
                    if not ret or frame is None:
                        break

                frame_count += 1
                t0 = time.perf_counter()

                mask = algo.apply(frame)
                t1 = time.perf_counter()

                filled = fill_mask(mask)
                t2 = time.perf_counter()

                build_display_frame_inplace(canvas, frame, mask, filled,
                                            algo.name, frame_count)
                t3 = time.perf_counter()

                if not headless:
                    cv2.imshow(window_title, canvas)
                t4 = time.perf_counter()

                if frame_count % 30 == 0:
                    print(f"  algo={1000*(t1-t0):.1f}ms  fill={1000*(t2-t1):.1f}ms  "
                          f"build={1000*(t3-t2):.1f}ms  show={1000*(t4-t3):.1f}ms")

                elapsed = t4 - t0
                if not headless:
                    remaining_ms = max(1, target_delay_ms - int(elapsed * 1000))
                    key = cv2.waitKey(remaining_ms) & 0xFF
                    if key == ord("n"):
                        break
                    elif key == ord("q"):
                        capture.release()
                        if out_writer:
                            out_writer.release()
                        cv2.destroyAllWindows()
                        if collect_metrics:
                            metrics["status"] = "INTERRUPTED"
                            return metrics
                        return None

                if save_output and out_writer:
                    out_writer.write(canvas)

            # ---- Cleanup after each video ----
            capture.release()
            if out_writer:
                out_writer.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)

            video_elapsed = time.time() - video_start
            avg_fps = frame_count / video_elapsed if video_elapsed > 0 else 0.0
            print(f"  Frames: {frame_count}  |  Avg FPS: {avg_fps:.2f}  "
                  f"|  Time: {video_elapsed:.1f}s")

            if collect_metrics:
                algo_metrics["videos"].append({
                    "video_name":   video_name,
                    "resolution":   f"{fw}x{fh}",
                    "total_frames": frame_count,
                    "total_time":   video_elapsed,
                    "avg_fps":      avg_fps,
                })
                algo_metrics["total_frames"] += frame_count
                algo_metrics["total_time"]   += video_elapsed

        # ---- Per-algorithm summary ----
        if collect_metrics:
            t = algo_metrics["total_time"]
            f = algo_metrics["total_frames"]
            algo_metrics["avg_fps"] = f / t if t > 0 else 0.0
            metrics["algorithms"].append(algo_metrics)

    print("\nAll done.")
    return metrics


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _make_output_path(video_name: str, algo_name: str, output_dir: str | None) -> Path:
    """Build an output file path that encodes the algorithm name."""
    safe_algo = algo_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    stem = Path(video_name).stem
    filename = f"{stem}__{safe_algo}.mp4"
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / filename
    return Path(filename)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_algorithm_names(args: argparse.Namespace) -> list[str]:
    names = []
    if getattr(args, "algorithms", None):
        names.extend([n.strip() for n in args.algorithms.split(",") if n.strip()])
    if getattr(args, "algorithm", None):
        names.append(args.algorithm.strip())
    return names or list(ALGORITHM_REGISTRY.keys())


def main(args: argparse.Namespace):
    algo_names = _parse_algorithm_names(args)
    try:
        algorithms = build_algorithms(algo_names)
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    video_sources = get_video_sources(args)
    if not video_sources:
        print("No video source specified or found.")
        return

    run_pipeline(
        algorithms=algorithms,
        video_sources=video_sources,
        save_output=args.save_output,
        output_dir=getattr(args, "output_dir", None),
        headless=False,
        collect_metrics=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Background subtraction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--algorithm", type=str, required=False,
        help="Single algorithm name: frame_diff | mean_variance | mog",
    )
    parser.add_argument(
        "--algorithms", type=str, required=False,
        help="Comma-separated algorithm names, e.g. 'frame_diff,mog'",
    )
    parser.add_argument(
        "--video_path", type=str, required=False, default="0",
        help="Path to a video file, or '0' for the default camera",
    )
    parser.add_argument(
        "--video_folder", type=str, required=False,
        help="Folder containing MP4 video files",
    )
    parser.add_argument(
        "--save_output", type=bool, required=False, default=False,
        help="Save side-by-side output videos to disk",
    )
    parser.add_argument(
        "--output_dir", type=str, required=False, default=None,
        help="Directory to save output videos (default: current directory)",
    )

    args = parser.parse_args()
    main(args)