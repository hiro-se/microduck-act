#!/usr/bin/env python3
"""Run LeRobot ACT inference in its isolated environment over a pickle pipe.

Standalone copy of the 4D keyboard-action worker from the modified Microduck RL
workspace. See README.md for attribution and the observation contract.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

IMAGE_KEY = "observation.images.head"
STATE_KEY = "observation.state"


def respond(event: str, **payload) -> None:
    sys.stdout.write(json.dumps({"event": event, **payload}) + "\n")
    sys.stdout.flush()


def run(
    checkpoint: Path,
    n_action_steps: int | None = None,
    temporal_ensemble_coeff: float | None = None,
) -> int:
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.policies.utils import prepare_observation_for_inference
    from lerobot.configs.policies import PreTrainedConfig

    try:
        with contextlib.redirect_stdout(sys.stderr):
            config = PreTrainedConfig.from_pretrained(checkpoint)
            if n_action_steps is not None:
                if not 1 <= n_action_steps <= config.chunk_size:
                    raise ValueError(
                        f"n_action_steps must be in [1, {config.chunk_size}], got {n_action_steps}"
                    )
                config.n_action_steps = n_action_steps
            if temporal_ensemble_coeff is not None:
                config.temporal_ensemble_coeff = temporal_ensemble_coeff
            policy_class = get_policy_class(config.type)
            policy = policy_class.from_pretrained(checkpoint, config=config)
            preprocessor, postprocessor = make_pre_post_processors(
                config, pretrained_path=str(checkpoint)
            )
        if IMAGE_KEY not in config.input_features or STATE_KEY not in config.input_features:
            raise ValueError(
                f"Checkpoint inputs must include {IMAGE_KEY} and {STATE_KEY}; "
                f"got {list(config.input_features)}"
            )
        if config.input_features[IMAGE_KEY].shape != (3, 320, 240):
            raise ValueError(
                f"Expected head image feature (3, 320, 240), got "
                f"{config.input_features[IMAGE_KEY].shape}"
            )
        if config.input_features[STATE_KEY].shape != (48,):
            raise ValueError(
                f"Expected 48D state feature, got {config.input_features[STATE_KEY].shape}"
            )
        if config.output_features["action"].shape != (4,):
            raise ValueError(
                f"Expected 4D keyboard-command action feature, got "
                f"{config.output_features['action'].shape}; retrain from the keyboard-labeled dataset"
            )
        device = torch.device(config.device)
        policy.eval()
        respond(
            "ready",
            device=str(device),
            action_dim=4,
            n_action_steps=config.n_action_steps,
            temporal_ensemble_coeff=config.temporal_ensemble_coeff,
        )
        while True:
            message = pickle.load(sys.stdin.buffer)
            event = message["event"]
            if event == "reset":
                policy.reset()
                preprocessor.reset()
                postprocessor.reset()
                respond("reset")
            elif event == "step":
                observation = {
                    IMAGE_KEY: message["image"],
                    STATE_KEY: message["state"],
                }
                observation = prepare_observation_for_inference(
                    observation, device, robot_type="microduck"
                )
                observation = preprocessor(observation)
                with torch.inference_mode():
                    action = policy.select_action(observation)
                    action = postprocessor(action)
                values = action.detach().cpu().numpy().reshape(-1)
                if values.shape != (4,) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid ACT output: shape={values.shape}")
                respond("action", values=values.tolist())
            elif event == "close":
                respond("closed")
                return 0
            else:
                raise ValueError(f"Unknown worker event: {event}")
    except EOFError:
        return 0
    except Exception as error:
        respond("error", message=f"{type(error).__name__}: {error}")
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--temporal-ensemble-coeff", type=float, default=None)
    args = parser.parse_args()
    raise SystemExit(
        run(args.checkpoint, args.n_action_steps, args.temporal_ensemble_coeff)
    )


if __name__ == "__main__":
    main()
