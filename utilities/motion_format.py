"""The 36-dim motion representation shared across all components.

A motion is a float array of shape ``(T, 36)`` (Unitree G1, 50 fps):

    motion[:, 0:3]   root position (x, y, z), meters
    motion[:, 3:7]   root quaternion (w, x, y, z), unit-norm, wxyz convention
    motion[:, 7:36]  29 joint positions / angles (radians), G1 joint order below

This mirrors the format documented in the public ``JianuoCao/TEXEDO`` dataset card.
"""

from __future__ import annotations

NFEATS = 36
ROOT_POS_SLICE = slice(0, 3)
ROOT_QUAT_SLICE = slice(3, 7)
JOINT_SLICE = slice(7, 36)
NUM_JOINTS = 29

# Order of motion[:, 7:36].
JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

assert len(JOINT_NAMES) == NUM_JOINTS

FPS = 50

__all__ = [
    "NFEATS",
    "ROOT_POS_SLICE",
    "ROOT_QUAT_SLICE",
    "JOINT_SLICE",
    "NUM_JOINTS",
    "JOINT_NAMES",
    "FPS",
]
