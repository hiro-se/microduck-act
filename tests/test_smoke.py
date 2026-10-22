import importlib.util
from pathlib import Path
import unittest

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("infer_policy", ROOT / "scripts/infer_policy.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class StandaloneSmokeTest(unittest.TestCase):
    def test_keyboard_labels(self):
        action = RUNNER.keyboard_policy_action(np.array([0.2, 0.0, 0.5]), 0.0)
        np.testing.assert_allclose(action, [0.2, 0.0, 0.5, 0.0])
        velocity, kick = RUNNER.decode_keyboard_policy_action(action, 0.3, -0.3, 0.2, 1.5)
        np.testing.assert_allclose(velocity, action[:3])
        self.assertEqual(kick, 0.0)

    def test_bam_camera_and_ball_resets(self):
        self.assertTrue(
            (ROOT / "src/mjlab_microduck/robot/microduck/robot_groundcontact.xml").is_file(),
            "Run scripts/prepare_assets.py --accept-model-license first",
        )
        bam = RUNNER.load_bam_model(RUNNER.BAM_KP_FW, 7.4, RUNNER.BAM_MAX_CURRENT)
        model, data, controller, names = RUNNER.load_mujoco_with_bam(
            RUNNER.MICRODUCK_BALL_XML, bam, 0.005, 0.1, RUNNER.BAM_VIN_MIN
        )
        self.assertEqual(len(names), 14)
        root_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        root_qpos = model.jnt_qposadr[root_joint]
        data.qpos[root_qpos:root_qpos + 7] = [0, 0, 0.125, 1, 0, 0, 0]
        data.qpos[model.jnt_qposadr[model.actuator_trnid[:, 0]]] = RUNNER.DEFAULT_POSE
        controller.reset(data.qpos)
        controller.q_target[:] = RUNNER.DEFAULT_POSE
        RUNNER.configure_head_camera_preview(model)
        ball_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        ball_qpos = model.jnt_qposadr[ball_joint]
        ball_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        rng = np.random.default_rng(17)
        with mujoco.Renderer(
            model, height=RUNNER.RAW_CAMERA_HEIGHT, width=RUNNER.RAW_CAMERA_WIDTH
        ) as renderer:
            renderer.enable_segmentation_rendering()
            for sampler, visible in (
                (RUNNER.sample_visible_ball_position, True),
                (RUNNER.sample_hidden_ball_position, False),
            ):
                for _ in range(12):
                    position = sampler(0, 0, 0, rng)
                    data.qpos[ball_qpos:ball_qpos + 7] = [*position, 1, 0, 0, 0]
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera="head_camera")
                    segmentation = renderer.render()
                    ball_pixels = (
                        (segmentation[:, :, 0] == ball_geom)
                        & (segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM)
                    )
                    self.assertEqual(bool(np.any(ball_pixels)), visible)
            renderer.disable_segmentation_rendering()
            frame = RUNNER.orient_camera_frame(renderer.render())
            self.assertEqual(frame.shape, (320, 240, 3))
            self.assertGreater(float(np.std(frame)), 1)
        for _ in range(100):
            controller.update()
            mujoco.mj_step(model, data)
        self.assertTrue(np.isfinite(data.qpos).all())


if __name__ == "__main__":
    unittest.main()