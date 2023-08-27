from typing import Dict, Tuple

import numpy as np


def _rotation_error(pose_error):
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


def _translation_error(pose_error):
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

        trans_errors.append(_translation_error(rel_err))
        rot_errors.append(_rotation_error(rel_err))
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
