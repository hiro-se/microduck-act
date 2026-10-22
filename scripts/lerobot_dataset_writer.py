#!/usr/bin/env python3
"""Receive live Microduck frames and write LeRobotDataset v3 episodes.

Standalone copy of the keyboard-action writer from the modified Microduck RL
workspace, including compatible dataset append support. See README.md.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np


FPS = 50
IMAGE_KEY = "observation.images.head"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
ACTION_NAMES = ("forward_velocity", "lateral_velocity", "yaw_rate", "kick_right")
IMAGE_WIDTH = 240
IMAGE_HEIGHT = 320
JOINT_NAMES = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)
STATE_NAMES = (
    "imu_angular_velocity_x", "imu_angular_velocity_y", "imu_angular_velocity_z",
    "projected_gravity_x", "projected_gravity_y", "projected_gravity_z",
    *(f"joint_position_{name}" for name in JOINT_NAMES),
    *(f"joint_velocity_{name}" for name in JOINT_NAMES),
    *(f"previous_action_{name}" for name in JOINT_NAMES),
)


def respond(event: str, **payload) -> None:
    sys.stdout.write(json.dumps({"event": event, **payload}) + "\n")
    sys.stdout.flush()


def open_dataset(repo_id: str, root: Path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        IMAGE_KEY: {
            "dtype": "video",
            "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
            "names": ["height", "width", "channel"],
        },
        STATE_KEY: {"dtype": "float32", "shape": (48,), "names": list(STATE_NAMES)},
        ACTION_KEY: {"dtype": "float32", "shape": (4,), "names": list(ACTION_NAMES)},
    }
    writer_options = {
        "streaming_encoding": True,
        "encoder_queue_maxsize": 100,
        "encoder_threads": 4,
    }
    info_path = root / "meta" / "info.json"
    if info_path.is_file():
        dataset = LeRobotDataset.resume(
            repo_id=repo_id,
            root=root,
            **writer_options,
        )
        expected = {
            IMAGE_KEY: ("video", features[IMAGE_KEY]["shape"]),
            STATE_KEY: ("float32", features[STATE_KEY]["shape"]),
            ACTION_KEY: ("float32", features[ACTION_KEY]["shape"]),
        }
        try:
            if dataset.meta.repo_id != repo_id:
                raise ValueError(
                    f"Existing dataset repo id is {dataset.meta.repo_id!r}, not {repo_id!r}"
                )
            for name, (dtype, shape) in expected.items():
                actual = dataset.features.get(name)
                if actual is None or actual["dtype"] != dtype or tuple(actual["shape"]) != tuple(shape):
                    raise ValueError(
                        f"Existing dataset feature {name!r} is incompatible with keyboard ACT: "
                        f"expected {dtype} {shape}, got {actual}"
                    )
        except Exception:
            dataset.finalize()
            raise
        return dataset

    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"Dataset root is non-empty but has no LeRobot metadata: {root}"
        )
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        robot_type="microduck",
        fps=FPS,
        features=features,
        use_videos=True,
        **writer_options,
    )


def run(args: argparse.Namespace) -> int:
    dataset = None
    episode_open = False
    episode_frames = 0
    respond("ready")
    try:
        while True:
            message = pickle.load(sys.stdin.buffer)
            event = message["event"]

            if event == "start":
                if episode_open:
                    raise RuntimeError("An episode is already being recorded")
                if dataset is None:
                    dataset = open_dataset(args.repo_id, args.root)
                episode_open = True
                episode_frames = 0
                respond("started")
            elif event == "frame":
                if not episode_open or dataset is None:
                    raise RuntimeError("Received a frame outside an active episode")
                dataset.add_frame(
                    {
                        IMAGE_KEY: message["image"],
                        STATE_KEY: message["state"],
                        ACTION_KEY: message["action"],
                        "task": "Find and kick the ball",
                    }
                )
                episode_frames += 1
            elif event == "finish":
                if not episode_open or dataset is None:
                    respond("empty")
                    continue
                if episode_frames:
                    dataset.save_episode()
                    saved_frames = episode_frames
                else:
                    dataset.clear_episode_buffer()
                    saved_frames = 0
                episode_open = False
                episode_frames = 0
                respond("saved", frames=saved_frames, episodes=dataset.num_episodes)
            elif event == "close":
                if dataset is not None:
                    if episode_open and dataset.has_pending_frames():
                        dataset.clear_episode_buffer()
                    dataset.finalize()
                respond("closed", episodes=dataset.num_episodes if dataset else 0)
                return 0
            else:
                raise ValueError(f"Unknown writer event: {event}")
    except EOFError:
        pass
    except Exception as error:
        respond("error", message=f"{type(error).__name__}: {error}")
        return 1
    finally:
        if dataset is not None:
            if episode_open and dataset.has_pending_frames():
                dataset.clear_episode_buffer()
            dataset.finalize()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()