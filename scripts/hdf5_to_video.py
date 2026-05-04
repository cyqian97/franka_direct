"""Convert an episode HDF5 file to two MP4 videos (primary + wrist cameras).

Usage:
    python scripts/hdf5_to_video.py episode_000001_fail.hdf5 [--fps 15] [--out_dir .]
"""

import argparse
import os
import sys

import cv2
import h5py
import numpy as np


def write_video(frames: np.ndarray, path: str, fps: float) -> None:
    """Write (T, H, W, 3) RGB uint8 array to an MP4 file."""
    T, H, W, _ = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open: {path}")
    for frame in frames:
        writer.write(frame[:, :, ::-1])   # RGB → BGR for OpenCV
    writer.release()
    print(f"  wrote {T} frames → {path}")


def main():
    parser = argparse.ArgumentParser(description="Export HDF5 episode images to MP4.")
    parser.add_argument("hdf5", help="Path to episode HDF5 file")
    parser.add_argument("--fps", type=float, default=None,
                        help="Playback fps (default: derived from step_timestamp, "
                             "falls back to 15)")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory (default: same directory as HDF5)")
    args = parser.parse_args()

    if not os.path.isfile(args.hdf5):
        print(f"[ERROR] File not found: {args.hdf5}")
        sys.exit(1)

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.hdf5))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.hdf5))[0]

    with h5py.File(args.hdf5, "r") as f:
        imgs = f.get("observations/images")
        if imgs is None:
            print("[ERROR] No observations/images group found in this file.")
            sys.exit(1)

        available = list(imgs.keys())
        print(f"[INFO] Image datasets: {available}")

        # Derive fps from recorded step timestamps if available and not overridden
        fps = args.fps
        if fps is None:
            ts = f.get("observations/step_timestamp")
            if ts is not None and len(ts) >= 2:
                intervals = np.diff(ts[:])
                intervals = intervals[intervals > 0]
                if len(intervals):
                    fps = float(np.mean(1.0 / intervals))
                    print(f"[INFO] Derived fps from step_timestamp: {fps:.2f}")
            if fps is None:
                fps = 15.0
                print(f"[INFO] Using default fps: {fps}")

        for name in ("primary", "wrist"):
            if name not in imgs:
                print(f"[SKIP] {name} not present")
                continue
            frames = imgs[name][:]   # (T, H, W, 3) uint8 RGB
            out_path = os.path.join(out_dir, f"{stem}_{name}.mp4")
            write_video(frames, out_path, fps)


if __name__ == "__main__":
    main()
