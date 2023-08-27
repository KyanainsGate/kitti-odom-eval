from typing import Dict, Tuple

import numpy as np

from kitti_odom_eval_conf import *

"""Preprocess"""


def __trajectory_distances(poses):
    """Compute distance for each pose w.r.t frame-0
    Args:
        poses (dict): {idx: 4x4 array}
    Returns:
        dist (float list): distance of each pose w.r.t frame-0
    """
    dist = [0]
    sort_frame_idx = sorted(poses.keys())
    for i in range(len(sort_frame_idx) - 1):
        cur_frame_idx = sort_frame_idx[i]
        next_frame_idx = sort_frame_idx[i + 1]
        P1 = poses[cur_frame_idx]
        P2 = poses[next_frame_idx]
        dx = P1[0, 3] - P2[0, 3]
        dy = P1[1, 3] - P2[1, 3]
        dz = P1[2, 3] - P2[2, 3]
        dist.append(dist[i] + np.sqrt(dx ** 2 + dy ** 2 + dz ** 2))
    return dist


def __last_frame_from_segment_length(dist, first_frame, length):
    """Find frame (index) that away from the first_frame with
    the required distance
    Args:
        dist (float list): distance of each pose w.r.t frame-0
        first_frame (int): start-frame index
        length (float): required distance
    Returns:
        i (int) / -1: end-frame index. if not found return -1
    """
    for i in range(first_frame, len(dist), 1):
        if dist[i] > (dist[first_frame] + length):
            return i
    return -1


"""Metrics"""


def __rotation_error(pose_error):
    """Compute rotation error
    Args:
        pose_error (4x4 array): relative pose error
    Returns:
        rot_error (float): rotation error
    """
    a = pose_error[0, 0]
    b = pose_error[1, 1]
    c = pose_error[2, 2]
    d = 0.5 * (a + b + c - 1.0)
    rot_error = np.arccos(max(min(d, 1.0), -1.0))
    return rot_error


def __translation_error(pose_error):
    """Compute translation error
    Args:
        pose_error (4x4 array): relative pose error
    Returns:
        trans_error (float): translation error
    """
    dx = pose_error[0, 3]
    dy = pose_error[1, 3]
    dz = pose_error[2, 3]
    trans_error = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    return trans_error


def __calc_kitti_sequence_errors(poses_gt, poses_result):
    """calculate sequence error
    Args:
        poses_gt (dict): {idx: 4x4 array}, ground truth poses
        poses_result (dict): {idx: 4x4 array}, predicted poses
    Returns:
        err (list list): [first_frame, rotation error, translation error, length, speed]
            - first_frame: frist frame index
            - rotation error: rotation error per length
            - translation error: translation error per length
            - length: evaluation trajectory length
            - speed: car speed (#FIXME: 10FPS is assumed)
    """
    err = []
    dist = __trajectory_distances(poses_gt)
    step_size = SAMPLE_STEP

    for first_frame in range(0, len(poses_gt), step_size):
        for i in range(len(LENGTH)):
            len_ = LENGTH[i]
            last_frame = __last_frame_from_segment_length(
                dist, first_frame, len_
            )

            # Continue if sequence not long enough
            if last_frame == -1 or \
                    not (last_frame in poses_result.keys()) or \
                    not (first_frame in poses_result.keys()):
                continue

            # compute rotational and translational errors
            pose_delta_gt = np.dot(
                np.linalg.inv(poses_gt[first_frame]),
                poses_gt[last_frame]
            )
            pose_delta_result = np.dot(
                np.linalg.inv(poses_result[first_frame]),
                poses_result[last_frame]
            )
            pose_error = np.dot(
                np.linalg.inv(pose_delta_result),
                pose_delta_gt
            )

            r_err = __rotation_error(pose_error)
            t_err = __translation_error(pose_error)

            # compute speed
            num_frames = last_frame - first_frame + 1.0
            speed = len_ / (0.1 * num_frames)

            err.append([first_frame, r_err / len_, t_err / len_, len_, speed])
    return err


"""Public"""


def compute_kitti_rels(gt: Dict[int, np.ndarray], pred: Dict[int, np.ndarray],
                       per_100m_deg=True, transl_is_100p=True) -> Tuple[float, float]:
    """Compute average translation & rotation errors
    Args:
        gt (4x4 array dict): ground-truth poses
        pred (4x4 array dict): predicted poses
        per_100m_deg (bool): Flag to convert the rotation error unit (deg/100m)
        transl_is_100p: Flag to multiply 100 to relative translation error
    Returns:
        ave_t_err (float): average translation error
        ave_r_err (float): average rotation error
    """
    t_err = 0
    r_err = 0

    seq_err = __calc_kitti_sequence_errors(gt, pred)
    seq_len = len(seq_err)

    if seq_len > 0:
        for item in seq_err:
            r_err += item[1]  # r_err / len_
            t_err += item[2]  # t_err / len_
        ave_t_err = t_err / seq_len
        ave_r_err = r_err / seq_len
        if per_100m_deg:
            ave_r_err = ave_r_err / np.pi * 180 * 100
        if transl_is_100p:
            ave_t_err = ave_t_err * 100
        return ave_t_err, ave_r_err
    else:
        return 0, 0


def compute_rpe(gt: Dict[int, np.ndarray], pred: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute RPE
    Args:
        gt (4x4 array dict): ground-truth poses
        pred (4x4 array dict): predicted poses
    Returns:
        rpe_trans
        rpe_rot
    """
    trans_errors = []
    rot_errors = []
    for i in list(pred.keys())[:-1]:
        gt1 = gt[i]
        gt2 = gt[i + 1]
        gt_rel = np.linalg.inv(gt1) @ gt2

        pred1 = pred[i]
        pred2 = pred[i + 1]
        pred_rel = np.linalg.inv(pred1) @ pred2
        rel_err = np.linalg.inv(gt_rel) @ pred_rel

        trans_errors.append(__translation_error(rel_err))
        rot_errors.append(__rotation_error(rel_err))
    rpe_trans = np.mean(np.asarray(trans_errors))
    rpe_rot = np.mean(np.asarray(rot_errors))
    return rpe_trans, rpe_rot


def compute_ate(gt: Dict[int, np.ndarray], pred: Dict[int, np.ndarray]) -> float:
    """Compute RMSE of ATE

    Parameters
    ----------
    gt : Dict[int, np.ndarray]
        gt (4x4 array dict): ground-truth poses
    pred : Dict[int, np.ndarray]
        pred (4x4 array dict): predicted poses

    Returns
    -------
    float


    """
    errors = []
    idx_0 = list(pred.keys())[0]

    for i in pred:
        cur_gt = gt[i]
        gt_xyz = cur_gt[:3, 3]

        cur_pred = pred[i]
        pred_xyz = cur_pred[:3, 3]

        align_err = gt_xyz - pred_xyz

        errors.append(np.sqrt(np.sum(align_err ** 2)))
    ate = np.sqrt(np.mean(np.asarray(errors) ** 2))
    return ate
