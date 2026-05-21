"""
Object tracking pipeline.

Mirrors the structure of run_pipeline.py but runs tracking algorithms
instead of background subtractors.  Each tracker produces a 2×2 display:

    TOP-LEFT:     Original frame  +  bounding boxes, track IDs, trajectories
    TOP-RIGHT:    Algorithm visualisation (flow points / back-projection / raw detections)
    BOTTOM-LEFT:  Internal motion / detection mask
    BOTTOM-RIGHT: Accumulated trajectory map (paths drawn on a dark canvas)

Keyboard controls:
    n — skip to next video
    q — quit immediately

Usage examples
--------------
Single tracker on a video file:
    python run_tracking.py --tracker klt --video_path path/to/video.mp4

SORT tracker with a specific YOLO model:
    python run_tracking.py --tracker sort \\
        --model_path ../CVSP/runs/train/yolo11s-finetune/weights/best.pt \\
        --video_folder path/to/folder

Available tracker names
-----------------------
    klt       KLT Optical Flow (Shi-Tomasi + Lucas-Kanade)
    camshift  CamShift (multi-object colour-histogram tracking)
    sort      SORT (YOLO detection + Kalman filter + greedy IoU matching)
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from scripts.tracking.tracker_base import Tracker, TrackResult, track_color
from scripts.tracking.klt_tracker import KLTTracker
from scripts.tracking.template_tracker import TemplateTracker
from scripts.tracking.sort_tracker import SORTTracker
from scripts.video_utils import draw_label, resize_for_display, get_video_sources


# ---------------------------------------------------------------------------
# Tracker registry
# ---------------------------------------------------------------------------

def _build_registry(model_path: str | None) -> dict[str, Tracker]:
    return {
        "klt":      KLTTracker(),
        "template": TemplateTracker(),
        "sort":     SORTTracker(model_path=model_path),
    }


def build_trackers(names: list[str], model_path: str | None) -> list[Tracker]:
    registry = _build_registry(model_path)
    trackers = []
    for name in names:
        key = name.strip().lower()
        if key not in registry:
            available = ", ".join(registry.keys())
            raise ValueError(f"Unknown tracker '{key}'. Available: {available}")
        trackers.append(registry[key])
    return trackers


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _draw_tracks(canvas: np.ndarray, frame: np.ndarray,
                 tracks: list[TrackResult], tracker_name: str, frame_idx: int):
    """Draw bounding boxes, IDs, and trajectory lines on *canvas* (in-place)."""
    np.copyto(canvas, frame)
    bar_h = 28
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], bar_h), (20, 20, 20), -1)
    cv2.putText(canvas, f"{tracker_name}  |  frame {frame_idx}",
                (8, bar_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220),
                1, cv2.LINE_AA)

    for t in tracks:
        c = t.color
        x1, y1, x2, y2 = t.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), c, 2)
        tag = f"ID {t.id}  {t.label}" if t.label else f"ID {t.id}"
        cv2.putText(canvas, tag, (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)


def _draw_viz(canvas: np.ndarray, viz: np.ndarray | None,
              frame: np.ndarray, label: str):
    """Fill *canvas* with the tracker's viz frame (or a placeholder)."""
    bar_h = 28
    if viz is not None:
        src = viz if viz.shape == frame.shape else frame
        np.copyto(canvas, src)
    else:
        canvas[:] = 30
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], bar_h), (20, 20, 20), -1)
    cv2.putText(canvas, label, (8, bar_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)


def _draw_mask(canvas: np.ndarray, mask: np.ndarray | None, label: str):
    """Fill *canvas* with a colourised mask (or a dark placeholder)."""
    bar_h = 28
    canvas[:] = 30
    if mask is not None:
        if mask.ndim == 2:
            colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            colored = mask
        np.copyto(canvas, colored)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], bar_h), (20, 20, 20), -1)
    cv2.putText(canvas, label, (8, bar_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)


def _update_traj_canvas(traj_canvas: np.ndarray, tracks: list[TrackResult]):
    """Draw new trajectory segments onto the persistent trajectory canvas."""
    for t in tracks:
        pts = t.trajectory
        if len(pts) < 2:
            continue
        cv2.line(traj_canvas, pts[-2], pts[-1], t.color, 1, cv2.LINE_AA)


def build_display_frame_inplace(
    canvas: np.ndarray,
    original: np.ndarray,
    tracks: list[TrackResult],
    viz: np.ndarray | None,
    mask: np.ndarray | None,
    traj_canvas: np.ndarray,
    tracker_name: str,
    frame_idx: int,
) -> None:
    """Write the 2×2 tracking grid directly into the pre-allocated *canvas*."""
    h, w = original.shape[:2]

    top_left     = canvas[0:h,       0:w]
    top_right    = canvas[0:h,       w+3:w*2+3]
    bottom_left  = canvas[h+3:h*2+3, 0:w]
    bottom_right = canvas[h+3:h*2+3, w+3:w*2+3]

    _draw_tracks(top_left, original, tracks, tracker_name, frame_idx)

    # TOP-RIGHT: current trajectories on a dark canvas.
    top_right[:] = 20
    bar_h = 28
    cv2.rectangle(top_right, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.putText(top_right, "Current trajectories", (8, bar_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    for t in tracks:
        pts = t.trajectory
        for i in range(1, len(pts)):
            cv2.line(top_right, pts[i - 1], pts[i], t.color, 1, cv2.LINE_AA)

    _draw_mask(bottom_left, mask, "Internal detection / motion mask")

    # BOTTOM-RIGHT: persistent trajectory map.
    np.copyto(bottom_right, traj_canvas)
    bar_h = 28
    cv2.rectangle(bottom_right, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.putText(bottom_right, "Trajectory map",
                (8, bar_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (220, 220, 220), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    trackers: list[Tracker],
    video_sources: list,
    save_output: bool = False,
    output_dir: str | None = None,
    headless: bool = False,
    collect_metrics: bool = False,
) -> dict | None:
    """
    Run each tracker on every video source and display / save the 2×2 grid.

    Args:
        trackers:        List of Tracker instances to run.
        video_sources:   Video file paths or camera indices.
        save_output:     Write output videos to disk.
        output_dir:      Destination directory for saved videos.
        headless:        Process without opening any display windows.
        collect_metrics: Accumulate and return performance metrics.

    Returns:
        Metrics dict if collect_metrics=True, otherwise None.
    """
    metrics = {"status": "SUCCESS", "trackers": []} if collect_metrics else None

    for tracker in trackers:
        print(f"\n{'='*60}")
        print(f"Tracker: {tracker.name}")
        print(f"{'='*60}")

        tracker_metrics = {
            "tracker": tracker.name,
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

            tracker.reset()

            capture = cv2.VideoCapture(video_source)
            if not capture.isOpened():
                print(f"  Error: could not open {video_source}")
                continue

            src_fps = capture.get(cv2.CAP_PROP_FPS) or 30
            target_delay_ms = max(1, int(1000 / src_fps))

            ret, first_frame = capture.read()
            if not ret or first_frame is None:
                print(f"  Error: could not read frames from {video_source}")
                capture.release()
                continue

            fh, fw = first_frame.shape[:2]
            if fh < 64 or fw < 64:
                print(f"  Error: implausibly small frame ({fw}x{fh})")
                capture.release()
                continue

            print(f"  Resolution: {fw}x{fh}  |  FPS: {src_fps:.1f}")

            # Pre-allocate display canvas and trajectory canvas.
            canvas = np.zeros((fh * 2 + 3, fw * 2 + 3, 3), dtype=np.uint8)
            canvas[fh:fh+3, :] = 60
            canvas[:, fw:fw+3] = 60
            traj_canvas = np.zeros((fh, fw, 3), dtype=np.uint8)

            # Output writer.
            out_writer = None
            if save_output:
                out_path = _make_output_path(video_name, tracker.name, output_dir)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_writer = cv2.VideoWriter(
                    str(out_path), fourcc, src_fps, (fw * 2 + 3, fh * 2 + 3)
                )
                print(f"  Saving to: {out_path}")

            window_title = f"{tracker.name}  —  {video_name}"
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

                tracks = tracker.update(frame)
                t1 = time.perf_counter()

                _update_traj_canvas(traj_canvas, tracks)
                build_display_frame_inplace(
                    canvas, frame, tracks,
                    tracker.get_viz(), tracker.get_mask(),
                    traj_canvas, tracker.name, frame_count,
                )
                t2 = time.perf_counter()

                if not headless:
                    cv2.imshow(window_title, canvas)
                t3 = time.perf_counter()

                if frame_count % 30 == 0:
                    print(f"  track={1000*(t1-t0):.1f}ms  "
                          f"display={1000*(t2-t1):.1f}ms  "
                          f"show={1000*(t3-t2):.1f}ms  "
                          f"tracks={len(tracks)}")

                if not headless:
                    elapsed = t3 - t0
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

            # ---- Cleanup ----
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
                tracker_metrics["videos"].append({
                    "video_name":   video_name,
                    "resolution":   f"{fw}x{fh}",
                    "total_frames": frame_count,
                    "total_time":   video_elapsed,
                    "avg_fps":      avg_fps,
                })
                tracker_metrics["total_frames"] += frame_count
                tracker_metrics["total_time"]   += video_elapsed

        if collect_metrics:
            t = tracker_metrics["total_time"]
            f = tracker_metrics["total_frames"]
            tracker_metrics["avg_fps"] = f / t if t > 0 else 0.0
            metrics["trackers"].append(tracker_metrics)

    print("\nAll done.")
    return metrics


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _make_output_path(video_name: str, tracker_name: str,
                      output_dir: str | None) -> Path:
    safe = tracker_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    stem = Path(video_name).stem
    filename = f"{stem}__{safe}.mp4"
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / filename
    return Path(filename)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_tracker_names(args: argparse.Namespace) -> list[str]:
    names = []
    if getattr(args, "trackers", None):
        names.extend([n.strip() for n in args.trackers.split(",") if n.strip()])
    if getattr(args, "tracker", None):
        names.append(args.tracker.strip())
    return names or ["klt", "template", "sort"]


def main(args: argparse.Namespace):
    tracker_names = _parse_tracker_names(args)
    try:
        trackers = build_trackers(tracker_names, getattr(args, "model_path", None))
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    video_sources = get_video_sources(args)
    if not video_sources:
        print("No video source specified or found.")
        return

    run_pipeline(
        trackers=trackers,
        video_sources=video_sources,
        save_output=args.save_output,
        output_dir=getattr(args, "output_dir", None),
        headless=False,
        collect_metrics=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Object tracking pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tracker", type=str, required=False,
        help="Single tracker name: klt | template | sort",
    )
    parser.add_argument(
        "--trackers", type=str, required=False,
        help="Comma-separated tracker names, e.g. 'klt,sort'",
    )
    parser.add_argument(
        "--video_path", type=str, required=False, default=None,
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
    parser.add_argument(
        "--model_path", type=str, required=False,
        default="yolo11m.pt",
        help="Path to YOLO .pt model for the SORT tracker (default: yolo11m.pt)",
    )

    args = parser.parse_args()
    main(args)
