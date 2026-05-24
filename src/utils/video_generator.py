"""
Create test videos from VisDrone image sequences for pipeline testing.
"""

import cv2
import os
from pathlib import Path


def create_video_from_sequence(
    sequence_dir,
    output_path,
    fps=20,
    max_frames=None,
):
    """Create a video from a sequence of images."""
    sequence_dir = Path(sequence_dir)
    images = sorted(sequence_dir.glob("*.jpg"))

    if not images:
        print(f"No images found in {sequence_dir}")
        return

    if max_frames:
        images = images[:max_frames]

    print(f"Creating video from {len(images)} images...")

    # Read first image to get dimensions
    first_img = cv2.imread(str(images[0]))
    height, width = first_img.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_path in images:
        img = cv2.imread(str(img_path))
        out.write(img)

    out.release()
    print(f"Video saved to: {output_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {len(images)}")


def main():
    sequences_dir = Path("VisDrone2019-MOT-val/sequences")
    output_dir = Path("test_videos")
    output_dir.mkdir(exist_ok=True)

    # Create videos from first few sequences
    sequences = sorted(sequences_dir.iterdir())

    for seq_dir in sequences[:3]:  # First 3 sequences
        if seq_dir.is_dir():
            output_path = output_dir / f"{seq_dir.name}.mp4"
            create_video_from_sequence(
                seq_dir,
                output_path,
                fps=20,
                max_frames=200,  # Limit to 200 frames for quick testing
            )


if __name__ == "__main__":
    main()
