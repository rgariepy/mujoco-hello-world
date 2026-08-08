"""Standing / balance controller for the quadruped."""

from __future__ import annotations

import numpy as np
import mujoco

# Actuator order matches model.xml
ACTUATOR_NAMES = [
    "fl_abad", "fl_hip", "fl_knee",
    "fr_abad", "fr_hip", "fr_knee",
    "hl_abad", "hl_hip", "hl_knee",
    "hr_abad", "hr_hip", "hr_knee",
]

# Nominal crouch pose (abad, hip, knee) per leg
STAND_POSE = {
    "fl": np.array([0.15, 0.45, -0.90]),
    "fr": np.array([-0.15, 0.45, -0.90]),
    "hl": np.array([0.15, 0.55, -0.95]),
    "hr": np.array([-0.15, 0.55, -0.95]),
}

# Gains for orientation → posture correction
K_ROLL = 0.55
K_PITCH = 0.70
K_HEIGHT = 1.2
TARGET_HEIGHT = 0.22


def torso_roll_pitch(data: mujoco.MjData) -> tuple[float, float]:
    """Return (roll, pitch) of the torso from its world rotation matrix."""
    R = data.body("torso").xmat.reshape(3, 3)
    # Body z-axis in world frame
    zx, zy, zz = R[:, 2]
    roll = np.arctan2(zy, zz)
    pitch = np.arctan2(-zx, np.sqrt(zy * zy + zz * zz))
    return float(roll), float(pitch)


def reset_to_stand(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Place the robot in the standing keyframe and clear velocities."""
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)


def apply_stability_control(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """PD posture targets with roll/pitch/height balance corrections.

    Position actuators track these targets; this layer adds a light
    virtual-model style correction so the robot resists tip-over and
    settles after disturbances (e.g. the falling red box).
    """
    roll, pitch = torso_roll_pitch(data)
    height = float(data.body("torso").xpos[2])
    height_err = TARGET_HEIGHT - height

    # Height: bend/unbend knees (and a bit of hip) to hold torso height
    knee_adj = np.clip(-K_HEIGHT * height_err, -0.35, 0.35)
    hip_h_adj = np.clip(0.35 * K_HEIGHT * height_err, -0.2, 0.2)

    # Pitch: lean forward → straighten front / crouch rear
    front_hip = np.clip(-K_PITCH * pitch, -0.45, 0.45)
    rear_hip = np.clip(K_PITCH * pitch, -0.45, 0.45)

    # Roll: tip right (positive roll about x) → push right legs down
    left_abad = np.clip(K_ROLL * roll, -0.35, 0.35)
    right_abad = np.clip(-K_ROLL * roll, -0.35, 0.35)
    left_knee_roll = np.clip(0.4 * K_ROLL * roll, -0.25, 0.25)
    right_knee_roll = np.clip(-0.4 * K_ROLL * roll, -0.25, 0.25)

    targets = {
        "fl": STAND_POSE["fl"] + np.array([left_abad, front_hip + hip_h_adj, knee_adj + left_knee_roll]),
        "fr": STAND_POSE["fr"] + np.array([right_abad, front_hip + hip_h_adj, knee_adj + right_knee_roll]),
        "hl": STAND_POSE["hl"] + np.array([left_abad, rear_hip + hip_h_adj, knee_adj + left_knee_roll]),
        "hr": STAND_POSE["hr"] + np.array([right_abad, rear_hip + hip_h_adj, knee_adj + right_knee_roll]),
    }

    for i, name in enumerate(ACTUATOR_NAMES):
        leg = name[:2]
        joint = name[3:]  # abad / hip / knee
        idx = {"abad": 0, "hip": 1, "knee": 2}[joint]
        data.ctrl[i] = float(np.clip(targets[leg][idx], -1.8, 1.8))
