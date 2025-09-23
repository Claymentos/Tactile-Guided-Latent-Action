from typing import Dict

import tensorflow as tf


def zero_relative_action_filter(traj: Dict, action_stats: Dict) -> bool:
    """
    Filters transitions whose actions are all-0 (only relative actions, no gripper action).
    Note: this filter is applied *after* action normalization, so need to compare to "normalized 0".
    """
    q01 = tf.convert_to_tensor(action_stats["q01"][:-1], dtype=traj["action"].dtype)
    q99 = tf.convert_to_tensor(action_stats["q99"][:-1], dtype=traj["action"].dtype)

    norm_zero_action = 2 * (tf.zeros_like(traj["action"][:, :-1]) - q01) / (q99 - q01 + 1e-8) - 1

    if not tf.reduce_any(tf.math.abs(traj["action"][:, :-1] - norm_zero_action) > 1e-6):
        tf.print("Non-zero action found:", tf.math.abs(traj["action"][:, :-1] - norm_zero_action))

    return tf.reduce_any(tf.math.abs(traj["action"][:, :-1] - norm_zero_action) > 1e-6)


def zero_absolute_action_filter(traj: Dict) -> bool:
    """
    Filters transitions whose actions are all-0 (on absolute actions, no gripper action).
    Considers a single gripper action at the end of the action vector.
    Note: this filter is applied *after* action normalization, but since we compute the relative actions from 2 normalized actions, 
    we have to compare them to zero, rather than a normalized zero.
    """
    relative_action = traj["action"][1:, :-1] - traj["action"][:-1, :-1]

    if not tf.reduce_any(tf.math.abs(relative_action) > 1e-5):
        tf.print("Non-zero action found:", relative_action)

    return tf.reduce_any(tf.math.abs(relative_action) > 1e-5)


def zero_proprio_filter(traj: Dict) -> bool:
    """
    Filters transitions whose proprioception difference between first and last frame is zero.
    Useful when skipping frames based on dataset frequency.
    Note that if not loading a single pair of frames, this could filter out valid transitions, even though is unlikely.
    If loading a single frame, this will filter out all transitions.
    """
    if "proprio" not in traj["observation"]:
        raise ValueError("Proprioception not found in trajectory observations. Use a different filter or load proprioception.")

    proprio_diff = traj["observation"]["proprio"][-1] - traj["observation"]["proprio"][0]

    # if not tf.reduce_any(tf.math.abs(proprio_diff) > 1e-5):
    #     tf.print("Non-zero proprioception difference found:", proprio_diff)

    return tf.reduce_any(tf.math.abs(proprio_diff) > 1e-5)