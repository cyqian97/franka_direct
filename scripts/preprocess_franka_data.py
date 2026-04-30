"""
Preprocess raw franka_direct HDF5 episodes for openvla-oft fine-tuning.

Steps:
  1. Resize images to 256x256 (openvla-oft standard).
  2. Randomly split episodes into train/ and val/ sub-directories.

Input layout:
    /PATH/TO/DATA/
        episode_000000.hdf5
        episode_000001.hdf5
        ...

Output layout:
    /PATH/TO/PREPROCESSED/
        train/
            episode_000000.hdf5
            ...
        val/
            episode_000000.hdf5
            ...

Usage:
    python scripts/preprocess_franka_data.py \\
        --data_dir data/pick_up_cup \\
        --out_dir  data/pick_up_cup_preprocessed \\
        --percent_val 0.1 \\
        --img_size 256
"""

import argparse
import glob
import os
import random

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_episode(path: str) -> dict:
    with h5py.File(path, "r") as f:
        ep = {
            "sim":                  f.attrs.get("sim", False),
            "language_instruction": f.attrs.get("language_instruction", ""),
            "qpos":    f["/observations/qpos"][()],
            "qvel":    f["/observations/qvel"][()],
            "ee_pose": f["/observations/ee_pose"][()],
            "gripper": f["/observations/gripper"][()],
            "action":  f["/action"][()],
            "images":  {},
        }
        if "/observations/images" in f:
            for cam in f["/observations/images"].keys():
                ep["images"][cam] = f[f"/observations/images/{cam}"][()]
    return ep


def resize_images(ep: dict, img_size: int) -> dict:
    for cam, frames in ep["images"].items():
        resized = []
        for frame in frames:
            resized.append(
                np.array(
                    Image.fromarray(frame).resize((img_size, img_size),
                                                  resample=Image.BICUBIC)
                )
            )
        ep["images"][cam] = np.stack(resized)
    return ep


def save_episode(ep: dict, path: str) -> None:
    T = ep["qpos"].shape[0]
    with h5py.File(path, "w", rdcc_nbytes=1024 ** 2 * 2) as f:
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
                imgs.create_dataset(cam, data=frames, dtype=np.uint8)

        f.create_dataset("action", data=ep["action"], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess franka_direct HDF5 episodes for openvla-oft")
    parser.add_argument("--data_dir",    required=True,
                        help="Directory containing raw episode_XXXXXX.hdf5 files")
    parser.add_argument("--out_dir",     required=True,
                        help="Output directory for preprocessed train/ and val/")
    parser.add_argument("--percent_val", type=float, default=0.1,
                        help="Fraction of episodes for validation (default: 0.1)")
    parser.add_argument("--img_size",    type=int, default=256,
                        help="Resize images to img_size × img_size (default: 256)")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    paths = sorted(glob.glob(os.path.join(args.data_dir, "episode_*.hdf5")))
    if not paths:
        print(f"[ERROR] No episode_*.hdf5 files found in {args.data_dir}")
        return

    print(f"Found {len(paths)} episodes in {args.data_dir}")

    random.shuffle(paths)
    n_val   = max(1, int(len(paths) * args.percent_val))
    n_train = len(paths) - n_val
    splits  = {"train": paths[:n_train], "val": paths[n_train:]}
    print(f"Split: {n_train} train, {n_val} val")

    for split, split_paths in splits.items():
        out_split = os.path.join(args.out_dir, split)
        os.makedirs(out_split, exist_ok=True)
        for i, src in enumerate(tqdm(split_paths, desc=split)):
            ep = load_episode(src)
            if ep["images"]:
                ep = resize_images(ep, args.img_size)
            out_path = os.path.join(out_split, f"episode_{i:06d}.hdf5")
            save_episode(ep, out_path)

    print(f"\nPreprocessed episodes saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
