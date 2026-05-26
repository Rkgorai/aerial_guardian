"""
Video writer abstraction for Aerial Guardian pipeline.
Provides three implementations:
- OpenCVVideoWriter: uses cv2.VideoWriter (software encoding).
- FFmpegNVENCWriter: uses ffmpeg with NVIDIA NVENC hardware acceleration.
- NullVideoWriter: discards frames (benchmark mode).
"""

import subprocess
import shutil
import sys
from typing import Tuple

import cv2
import numpy as np


class VideoWriterBase:
    """Abstract base class for video writers."""

    def write(self, frame: np.ndarray):
        raise NotImplementedError

    def release(self):
        pass


class OpenCVVideoWriter(VideoWriterBase):
    def __init__(self, output_path: str, fps: float, frame_size: Tuple[int, int]):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open OpenCV VideoWriter for {output_path}")

    def write(self, frame: np.ndarray):
        self.writer.write(frame)

    def release(self):
        self.writer.release()


class FFmpegNVENCWriter(VideoWriterBase):
    """Writes video using ffmpeg with NVENC hardware acceleration.
    The ffmpeg process receives raw BGR frames via stdin.
    """

    def __init__(self, output_path: str, fps: float, frame_size: Tuple[int, int]):
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg not found in PATH; cannot use FFmpegNVENCWriter.")
        width, height = frame_size
        cmd = [
            "ffmpeg",
            "-y",  # overwrite output
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgr24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "-",  # stdin
            "-c:v",
            "h264_nvenc",
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]
        self.process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        if self.process.stdin is None:
            raise RuntimeError("Failed to open ffmpeg stdin pipe.")
        self._stderr = self.process.stderr

    def write(self, frame: np.ndarray):
        # Ensure frame is contiguous BGR uint8 array
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)
        try:
            self.process.stdin.write(frame.tobytes())
        except BrokenPipeError as e:
            raise RuntimeError(f"ffmpeg pipe broken: {e}")

    def release(self):
        if self.process.stdin:
            self.process.stdin.close()
        self.process.wait()
        # Capture any ffmpeg error output for debugging
        if self._stderr:
            err = self._stderr.read().decode()
            if err:
                sys.stderr.write(err)
            self._stderr.close()


class NullVideoWriter(VideoWriterBase):
    """A no‑op writer used for benchmarking without video output."""

    def write(self, frame: np.ndarray):
        pass

    def release(self):
        pass


def get_video_writer(
    output_path: str | None,
    fps: float,
    frame_size: Tuple[int, int],
    encoder: str = "opencv",
) -> VideoWriterBase:
    """Factory returning the appropriate writer.

    Parameters
    ----------
    output_path: str | None
        Destination file path. If None or empty, a NullVideoWriter is returned.
    fps: float
        Frame rate.
    frame_size: Tuple[int, int]
        (width, height).
    encoder: str
        One of "opencv", "ffmpeg", or "none".
    """
    if not output_path:
        return NullVideoWriter()
    encoder = encoder.lower()
    if encoder == "none":
        return NullVideoWriter()
    if encoder == "ffmpeg":
        try:
            return FFmpegNVENCWriter(output_path, fps, frame_size)
        except Exception as e:
            sys.stderr.write(f"[WARN] FFmpegNVENCWriter failed ({e}), falling back to OpenCV writer.\n")
            # fall back to OpenCV
    # default/openCV fallback
    return OpenCVVideoWriter(output_path, fps, frame_size)
