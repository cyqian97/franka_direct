#!/usr/bin/env python3
"""
Build a franka_fr3 RLDS/TFDS dataset from raw teleoperation HDF5 episodes.

Combines the two-step pipeline into a single command:
  1. Preprocess: resize images to img_size × img_size, split into train/val.
  2. Build RLDS: convert preprocessed HDF5 files to TensorFlow Datasets format
     for use with openvla-oft fine-tuning.

Usage:
    python scripts/build_rlds.py --config config/process.yaml
"""

import argparse
import glob
import os
import random
import sys

import h5py
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "rlds_dataset"))


# ── Preprocessing helpers ─────────────────────────────────────────────────────

def _load_episode(path: str) -> dict:
    with h5py.File(path, "r") as f:
        ep = {
            "sim":                  f.attrs.get("sim", False),
            "language_instruction": str(f.attrs.get("language_instruction", "")),
            "qpos":    f["/observations/qpos"][()],
            "qvel":    f["/observations/qvel"][()],
            "ee_pose": f["/observations/ee_pose"][()],
            "gripper": f["/observations/gripper"][()],
            "action":     f["/action"][()],
            "action_eef": f["/action_eef"][()] if "/action_eef" in f else None,
            "images": {},
        }
        if "/observations/images" in f:
            for cam in f["/observations/images"].keys():
                ep["images"][cam] = f[f"/observations/images/{cam}"][()]
    return ep


def _resize_images(ep: dict, img_size: int) -> dict:
    for cam in list(ep["images"].keys()):
        arr = ep["images"][cam]
        if arr.ndim == 4 and arr.shape[-1] == 3:   # RGB uint8
            ep["images"][cam] = np.stack([
                np.array(Image.fromarray(fr).resize((img_size, img_size), Image.BICUBIC))
                for fr in arr
            ]).astype(np.uint8)
        elif arr.ndim == 3:                          # depth float32 [T, H, W]
            ep["images"][cam] = np.stack([
                np.array(Image.fromarray(fr).resize((img_size, img_size), Image.BILINEAR))
                for fr in arr
            ]).astype(np.float32)
    return ep


def _save_episode(ep: dict, path: str) -> None:
    with h5py.File(path, "w") as f:
        f.attrs["sim"] = ep["sim"]
        f.attrs["language_instruction"] = ep["language_instruction"]

        obs = f.create_group("observations")
        obs.create_dataset("qpos",    data=ep["qpos"],    dtype=np.float32)
        obs.create_dataset("qvel",    data=ep["qvel"],    dtype=np.float32)
        obs.create_dataset("ee_pose", data=ep["ee_pose"], dtype=np.float32)
        obs.create_dataset("gripper", data=ep["gripper"], dtype=np.float32)

        if ep["images"]:
            imgs = obs.create_group("images")
            for cam, frames in ep["images"].items():
                imgs.create_dataset(cam, data=frames,
                                    dtype=np.uint8 if frames.dtype == np.uint8 else np.float32)

        f.create_dataset("action", data=ep["action"], dtype=np.float32)
        if ep["action_eef"] is not None:
            f.create_dataset("action_eef", data=ep["action_eef"], dtype=np.float32)


def preprocess(data_dir: str, prep_dir: str, img_size: int,
               percent_val: float, seed: int) -> None:
    """Resize images and split episodes into train/ and val/ subdirectories."""
    paths = sorted(glob.glob(os.path.join(data_dir, "episode_*.hdf5")))
    if not paths:
        raise FileNotFoundError(f"No episode_*.hdf5 files found in {data_dir}")
    print(f"Found {len(paths)} episodes in {data_dir}")

    rng = random.Random(seed)
    shuffled = list(paths)
    rng.shuffle(shuffled)
    n_val  = max(1, int(len(shuffled) * percent_val))
    splits = {
        "train": shuffled[: len(shuffled) - n_val],
        "val":   shuffled[len(shuffled) - n_val :],
    }
    print(f"Split: {len(splits['train'])} train, {len(splits['val'])} val")

    for split, split_paths in splits.items():
        out_split = os.path.join(prep_dir, split)
        os.makedirs(out_split, exist_ok=True)
        for i, src in enumerate(tqdm(split_paths, desc=f"  {split}")):
            ep  = _load_episode(src)
            if ep["images"] and img_size:
                ep = _resize_images(ep, img_size)
            _save_episode(ep, os.path.join(out_split, f"episode_{i:06d}.hdf5"))


def build_rlds(prep_dir: str, rlds_dir: str) -> None:
    """Run the TFDS dataset builder on preprocessed HDF5 files."""
    from franka_fr3.franka_fr3_dataset_builder import FrankaFr3
    FrankaFr3.PREPROCESSED_DIR = prep_dir
    builder = FrankaFr3(data_dir=rlds_dir)
    builder.download_and_prepare()
    print(f"\nRLDS dataset written to: {rlds_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build franka_fr3 RLDS dataset from raw HDF5 episodes")
    parser.add_argument("--config", default="config/process.yaml",
                        help="Path to YAML processing config (default: config/process.yaml)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}

    data_dir    = cfg["data_dir"]
    rlds_dir    = cfg["rlds_dir"]
    img_size    = int(cfg.get("img_size", 256))
    percent_val = float(cfg.get("percent_val", 0.1))
    seed        = int(cfg.get("seed", 42))
    prep_dir    = cfg.get("prep_dir") or (rlds_dir + "_prep")

    print("=" * 60)
    print("franka_fr3 RLDS dataset builder")
    print("=" * 60)
    print(f"  data_dir  : {data_dir}")
    print(f"  prep_dir  : {prep_dir}")
    print(f"  rlds_dir  : {rlds_dir}")
    print(f"  img_size  : {img_size}")
    print(f"  val split : {percent_val:.0%}")
    print()

    # Step 1 — preprocess
    print("=== Step 1: Preprocessing (resize + split) ===")
    preprocess(data_dir, prep_dir, img_size, percent_val, seed)

    # Step 2 — build RLDS
    print("\n=== Step 2: Building RLDS dataset ===")
    try:
        build_rlds(prep_dir, rlds_dir)
    except ImportError as e:
        print(f"\n[WARN] RLDS builder unavailable ({e})")
        print(f"       Preprocessed data is ready at: {prep_dir}")
        print(f"       To build manually:")
        print(f"         cd scripts/rlds_dataset")
        print(f"         tfds build franka_fr3 --data_dir {rlds_dir} "
              f"--config_kwarg data_dir={prep_dir}")


if __name__ == "__main__":
    main()
