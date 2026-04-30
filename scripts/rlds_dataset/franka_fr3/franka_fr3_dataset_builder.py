"""
TFDS dataset builder for franka_direct episodes.

Converts preprocessed franka_direct HDF5 files (output of preprocess_franka_data.py)
into the RLDS format expected by openvla-oft.

Episode HDF5 layout expected:
    /observations/
        qpos        [T, 7]   float32  joint positions (rad)
        qvel        [T, 7]   float32  joint velocities (rad/s)
        ee_pose     [T, 16]  float32  O_T_EE column-major
        gripper     [T, 1]   float32  normalized [0=closed, 1=open]
        images/
            primary [T, H, W, 3]  uint8 RGB
            wrist   [T, H, W, 3]  uint8 RGB
    /action         [T, 8]   float32  [q_target(7), gripper_norm(1)]
    @language_instruction   str
    @sim            bool

Usage:
    # From the franka_direct repo root:
    cd scripts/rlds_dataset
    tfds build franka_fr3 --data_dir /PATH/TO/RLDS/OUTPUT/ \\
        --config_kwarg data_dir=/PATH/TO/PREPROCESSED/HDF5/

    # Or call the builder programmatically:
    python franka_fr3/franka_fr3_dataset_builder.py \\
        --preprocessed_dir /PATH/TO/PREPROCESSED/ \\
        --out_dir          /PATH/TO/RLDS/OUTPUT/
"""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, Iterator, Tuple

import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# Image dimensions after preprocessing (set by preprocess_franka_data.py)
IMG_SIZE = 256

# Action/proprio dimensions
ACTION_DIM = 8    # q_target(7) + gripper_norm(1)
PROPRIO_DIM = 8   # qpos(7)    + gripper(1)


class FrankaFr3(tfds.core.GeneratorBasedBuilder):
    """RLDS dataset builder for franka_direct Franka FR3 teleoperation data."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release."}

    # Override at instantiation time to point at your preprocessed HDF5 dir.
    PREPROCESSED_DIR: str = ""

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=(
                "Franka FR3 teleoperation episodes collected with the "
                "franka_direct stack. Action = absolute joint targets (7 DOF) "
                "+ normalized gripper (0=closed, 1=open)."
            ),
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "primary": tfds.features.Image(
                                        shape=(IMG_SIZE, IMG_SIZE, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                    ),
                                    "wrist": tfds.features.Image(
                                        shape=(IMG_SIZE, IMG_SIZE, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                    ),
                                    "qpos": tfds.features.Tensor(
                                        shape=(7,), dtype=np.float32
                                    ),
                                    "gripper": tfds.features.Tensor(
                                        shape=(1,), dtype=np.float32
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(ACTION_DIM,), dtype=np.float32,
                                doc="Absolute joint targets (7) + gripper_norm (1).",
                            ),
                            "action_eef": tfds.features.Tensor(
                                shape=(7,), dtype=np.float32,
                                doc="Delta EEF: pos_delta (3, m) + rot_vec (3, rad) + gripper_norm (1).",
                            ),
                            "language_instruction": tfds.features.Text(),
                            "is_first": tf.bool,
                            "is_last": tf.bool,
                            "is_terminal": tf.bool,
                            "reward": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="1.0 on the last step, 0.0 otherwise.",
                            ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(
                                doc="Path to the source HDF5 file."
                            ),
                        }
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/cyqian/franka_direct",
            citation="",
        )

    def _split_generators(
        self, dl_manager: tfds.download.DownloadManager
    ) -> Dict[str, Any]:
        preprocessed_dir = self.PREPROCESSED_DIR or os.environ.get(
            "FRANKA_FR3_DATA_DIR", ""
        )
        if not preprocessed_dir:
            raise ValueError(
                "Set FrankaFr3.PREPROCESSED_DIR or the FRANKA_FR3_DATA_DIR "
                "environment variable to the preprocessed HDF5 directory."
            )
        train_dir = os.path.join(preprocessed_dir, "train")
        val_dir   = os.path.join(preprocessed_dir, "val")
        return {
            "train": self._generate_examples(train_dir),
            "val":   self._generate_examples(val_dir),
        }

    def _generate_examples(
        self, data_dir: str
    ) -> Iterator[Tuple[str, Dict]]:
        paths = sorted(glob.glob(os.path.join(data_dir, "episode_*.hdf5")))
        if not paths:
            raise FileNotFoundError(
                f"No episode_*.hdf5 files found in {data_dir}"
            )

        for ep_idx, path in enumerate(paths):
            with h5py.File(path, "r") as f:
                lang       = str(f.attrs.get("language_instruction", ""))
                qpos       = f["/observations/qpos"][()]    # [T, 7]
                gripper    = f["/observations/gripper"][()] # [T, 1]
                action     = f["/action"][()]               # [T, 8]
                action_eef = f["/action_eef"][()]           # [T, 7]

                has_images = "/observations/images" in f
                if has_images:
                    primary_imgs = f["/observations/images/primary"][()]  # [T,H,W,3]
                    wrist_imgs   = f["/observations/images/wrist"][()]
                else:
                    T = qpos.shape[0]
                    primary_imgs = np.zeros((T, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                    wrist_imgs   = np.zeros((T, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

            T = qpos.shape[0]
            steps = []
            for t in range(T):
                steps.append(
                    {
                        "observation": {
                            "primary": primary_imgs[t],
                            "wrist":   wrist_imgs[t],
                            "qpos":    qpos[t].astype(np.float32),
                            "gripper": gripper[t].astype(np.float32),
                        },
                        "action":               action[t].astype(np.float32),
                            "action_eef":           action_eef[t].astype(np.float32),
                        "language_instruction": lang,
                        "is_first":   t == 0,
                        "is_last":    t == T - 1,
                        "is_terminal": t == T - 1,
                        "reward":     float(t == T - 1),
                    }
                )

            yield ep_idx, {
                "steps":            steps,
                "episode_metadata": {"file_path": path},
            }


# ── Standalone build helper ───────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build franka_fr3 RLDS dataset from preprocessed HDF5 files")
    parser.add_argument("--preprocessed_dir", required=True,
                        help="Directory with train/ and val/ sub-dirs of HDF5 files")
    parser.add_argument("--out_dir",          required=True,
                        help="Output directory for the TFDS RLDS dataset")
    args = parser.parse_args()

    FrankaFr3.PREPROCESSED_DIR = args.preprocessed_dir
    builder = FrankaFr3(data_dir=args.out_dir)
    builder.download_and_prepare()
    print(f"\nRLDS dataset written to {args.out_dir}")
    print("Register 'franka_fr3' in openvla-oft before fine-tuning.")
