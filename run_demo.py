"""
Demo runner: runs each tracker on its designated video in sequence.
"""

import subprocess
import sys

DEMOS = [
    ("klt",      "datasets/VisDrone-VID/test/videos/3727445-hd_1920_1080_30fps.mp4"),
    ("klt",      "datasets/VisDrone-VID/test/videos/uav0000120_04775_v.mp4"),
    ("template", "datasets/VisDrone-VID/test/videos/uav0000120_04775_v.mp4"),
    ("sort",     "datasets/VisDrone-VID/test/videos/uav0000009_03358_v.mp4"),
]

for tracker, video in DEMOS:
    print(f"\n>>> Running {tracker} on {video}")
    result = subprocess.run(
        [sys.executable, "run_tracking.py", "--tracker", tracker, "--video_path", video]
    )
    if result.returncode != 0:
        print(f"[!] {tracker} exited with code {result.returncode}, stopping.")
        sys.exit(result.returncode)

print("\nAll demos complete.")
