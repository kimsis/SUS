"""
Video processing utilities for CVSP project.
Contains helper functions for camera management, display, and video source handling.
"""

from pathlib import Path
import cv2


def list_cameras(max_cameras=10):
    """List all available cameras with details"""
    available = []
    print("\n" + "="*80)
    print("AVAILABLE CAMERAS:")
    print("="*80)
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                backend = cap.getBackendName()
                
                print(f"\n[{len(available) + 1}] Camera Index: {i}")
                print(f"    Resolution: {width}x{height}")
                print(f"    FPS: {fps}")
                print(f"    Backend: {backend}")
                
                available.append(i)
            cap.release()
    
    print("\n" + "="*80)
    
    if not available:
        print("No cameras found!")
        return None
    
    # Let user select
    while True:
        try:
            choice = input(f"\nSelect camera [1-{len(available)}] or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                return None
            
            choice = int(choice)
            if 1 <= choice <= len(available):
                selected_index = available[choice - 1]
                print(f"\nSelected: Camera Index {selected_index}")
                return selected_index
            else:
                print(f"Please enter a number between 1 and {len(available)}")
        except ValueError:
            print("Invalid input. Please enter a number.")


def draw_label(frame, label, position, color, font_scale=0.6, thickness=2, 
               bg_color=None, text_color=(255, 255, 255)):
    """
    Draw text label with background rectangle.
    
    Args:
        frame: Image to draw on
        label: Text to display
        position: Tuple (x, y) - bottom-left corner of where label should appear
        color: Color for background rectangle (B, G, R)
        font_scale: Size of the font (default 0.6)
        thickness: Thickness of text (default 2)
        bg_color: Background color (if None, uses 'color')
        text_color: Text color (default white)
    """
    x, y = position
    
    # Calculate text size
    (label_width, label_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    
    # Use provided bg_color or default to main color
    background = bg_color if bg_color is not None else color
    
    # Draw background rectangle
    cv2.rectangle(
        frame,
        (x, y - label_height - baseline),
        (x + label_width, y),
        background,
        1
    )
    
    # Draw text
    cv2.putText(
        frame,
        label,
        (x, y - baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        thickness
    )


def resize_for_display(frame, max_width=1920, max_height=1080):
    """
    Resize frame to fit within max dimensions while maintaining aspect ratio.
    
    Args:
        frame: Input frame
        max_width: Maximum width for display
        max_height: Maximum height for display
    
    Returns:
        Resized frame
    """
    height, width = frame.shape[:2]
    
    # If frame is already smaller, return as-is
    if width <= max_width and height <= max_height:
        return frame
    
    # Calculate scaling factor to fit within max dimensions
    scale = min(max_width / width, max_height / height)
    
    new_width = int(width * scale)
    new_height = int(height * scale)
    # print(f"width: {new_width}, height: {new_height}")
    
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def get_video_sources(args):
    """
    Get list of video sources based on arguments.

    Supports:
        --video_folder  folder of .mp4 files
        --video_path    single video file or '0' for camera

    Returns:
        list: Video file paths or camera index.
    """
    if args.video_folder:
        folder = Path(args.video_folder)
        video_files = sorted(folder.glob("*.mp4"))
        if not video_files:
            print(f"No MP4 files found in {args.video_folder}")
            return []
        print(f"Found {len(video_files)} video(s)")
        return [str(v) for v in video_files]

    if args.video_path:
        if args.video_path == '0':
            camera_index = list_cameras()
            return [camera_index] if camera_index is not None else []
        return [args.video_path]

    return []


