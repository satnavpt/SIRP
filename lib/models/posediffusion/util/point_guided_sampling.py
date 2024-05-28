# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from typing import Dict, List, Optional, Union
from .camera_transform import pose_encoding_to_camera, camera_to_pose_encoding, _convert_pixels_to_ndc, transform_to_extrinsics, convert_data_to_perspective_camera, convert_pose_solver_to_perspective_camera, opencv_from_visdom_projection, pose_encoding_to_visdom, get_cropped_images, pose_solver_camera_to_pose_encoding
from .get_fundamental_matrix import get_fundamental_matrices
from pytorch3d.renderer.cameras import CamerasBase, PerspectiveCameras
import os
from pytorch3d.transforms import se3_exp_map, se3_log_map, Transform3d, so3_relative_angle, quaternion_invert, quaternion_multiply, quaternion_apply
from pytorch3d.transforms.rotation_conversions import matrix_to_quaternion, quaternion_to_matrix
import numpy as np
import cv2 as cv
from lib.models.matching.pose_solver import backproject_3d, backproject_3d_tensor
from visdom import Visdom
from pytorch3d.structures.pointclouds import *
from pytorch3d.vis.plotly_vis import plot_scene, AxisArgs
import time
import csv
from transforms3d.quaternions import qinverse, rotate_vector, qmult

VARIANTS_ANGLE_SIN = 'sin'
VARIANTS_ANGLE_COS = 'cos'

def quat_angle_error(label, pred, variant=VARIANTS_ANGLE_SIN) -> np.ndarray:
    assert label.shape == (4,)
    assert pred.shape == (4,)
    assert variant in (VARIANTS_ANGLE_SIN, VARIANTS_ANGLE_COS), \
        f"Need variant to be in ({VARIANTS_ANGLE_SIN}, {VARIANTS_ANGLE_COS})"

    if len(label.shape) == 1:
        label = np.expand_dims(label, axis=0)
    if len(label.shape) != 2 or label.shape[0] != 1 or label.shape[1] != 4:
        raise RuntimeError(f"Unexpected shape of label: {label.shape}, expected: (1, 4)")

    if len(pred.shape) == 1:
        pred = np.expand_dims(pred, axis=0)
    if len(pred.shape) != 2 or pred.shape[0] != 1 or pred.shape[1] != 4:
        raise RuntimeError(f"Unexpected shape of pred: {pred.shape}, expected: (1, 4)")

    label = label.astype(np.float64)
    pred = pred.astype(np.float64)

    q1 = pred / np.linalg.norm(pred, axis=1, keepdims=True)
    q2 = label / np.linalg.norm(label, axis=1, keepdims=True)
    if variant == VARIANTS_ANGLE_COS:
        d = np.abs(np.sum(np.multiply(q1, q2), axis=1, keepdims=True))
        d = np.clip(d, a_min=-1, a_max=1)
        angle = 2. * np.degrees(np.arccos(d))
    elif variant == VARIANTS_ANGLE_SIN:
        if q1.shape[0] != 1 or q2.shape[0] != 1:
            raise NotImplementedError(f"Multiple angles is todo")
        # https://www.researchgate.net/post/How_do_I_calculate_the_smallest_angle_between_two_quaternions/5d6ed4a84f3a3e1ed3656616/citation/download
        sine = qmult(q1[0], qinverse(q2[0]))  # note: takes first element in 2D array
        # 114.59 = 2. * 180. / pi
        angle = np.arcsin(np.linalg.norm(sine[1:], keepdims=True)) * 114.59155902616465
        angle = np.expand_dims(angle, axis=0)

    return angle.astype(np.float64)

class HighLossException(Exception):
    def __init__(self, loss):
        self.loss = loss

def point_guided_sampling(model_mean: torch.Tensor, t: int, disable_retry, matches_dict: Dict, uncropped_data: Dict, viz, pose_scale, pose_solver_conf, flip, multi_sample):
    device = model_mean.device

    def _to_device(tensor):
        return torch.from_numpy(tensor).to(device)

    kp1 = _to_device(matches_dict["kp1"])
    kp2 = _to_device(matches_dict["kp2"])
    
    processed_matches = {
        "kp1": kp1,
        "kp2": kp2,
    }

    def _to_homogeneous(tensor):
        return torch.nn.functional.pad(tensor, [0, 1], value=1)

    pose_encoding_type = "absT_quaR_logFL"

    # optimise
    model_mean_reverse = None
    if flip and t == 9:
        model_mean, rescale_factor, loss = PGS3D_optimize(model_mean, 10*(9-t), uncropped_data, processed_matches, iter_num=25, viz=viz, pose_scale=pose_scale)

        model_mean_reverse = model_mean.detach().clone()
        r = quaternion_to_matrix(model_mean_reverse[0, 1, 3:7])
        model_mean_reverse[0, 1, 3:7] = matrix_to_quaternion(r)
        model_mean_reverse[0, 1, 0] *= -1

        model_mean_reverse, rescale_factor_reverse, loss_reverse = PGS3D_optimize(model_mean_reverse, 10*(9-t), uncropped_data, processed_matches, iter_num=30, viz=viz, pose_scale=pose_scale)

        if multi_sample and not disable_retry:
            raise HighLossException(min(loss.item(), loss_reverse.item()))
    elif t == 9:
        model_mean, rescale_factor, loss = PGS3D_optimize(model_mean, 10*(9-t), uncropped_data, processed_matches, viz=viz, pose_scale=pose_scale)
        if multi_sample and not disable_retry:
            raise HighLossException(loss.item())
    else:
        model_mean, rescale_factor, loss = PGS3D_optimize(model_mean, 10*(9-t), uncropped_data, processed_matches, viz=viz, pose_scale=pose_scale)
        # print(f"loss: {loss}")

    # pred_cameras = pose_encoding_to_visdom(model_mean, pose_encoding_type)
    # shape = torch.tensor([[224, 224], [224, 224]])
    # gt_pose = convert_data_to_perspective_camera(uncropped_data)
    # pred_R, pred_T, pred_K = opencv_from_visdom_projection(pred_cameras, shape)
    # gt_R, gt_T, gt_K = opencv_from_visdom_projection(gt_pose, shape)

    # out = {}
    # out['t'] = t
    # out['predR0'] = matrix_to_quaternion(pred_R)[0].cpu().numpy()
    # out['predT0'] = pred_T[0].cpu().numpy()
    # out['predK0'] = torch.tensor([pred_K[0][0][0], pred_K[0][0][2], pred_K[0][1][1], pred_K[0][1][2]]).cpu().numpy()
    # out['predR1'] = matrix_to_quaternion(pred_R)[1].cpu().numpy()
    # out['predT1'] = pred_T[1].cpu().numpy()
    # out['predK1'] = torch.tensor([pred_K[1][0][0], pred_K[1][0][2], pred_K[1][1][1], pred_K[1][1][2]]).cpu().numpy()
    # out['gtR0'] = matrix_to_quaternion(gt_R)[0].cpu().numpy()
    # out['gtT0'] = gt_T[0].cpu().numpy()
    # out['gtK0'] = torch.tensor([gt_K[0][0][0], gt_K[0][0][2], gt_K[0][1][1], gt_K[0][1][2]]).cpu().numpy()
    # out['gtR1'] = matrix_to_quaternion(gt_R)[1].cpu().numpy()
    # out['gtT1'] = gt_T[1].cpu().numpy()
    # out['gtK1'] = torch.tensor([gt_K[1][0][0], gt_K[1][0][2], gt_K[1][1][1], gt_K[1][1][2]]).cpu().numpy()
    # out['pgs3dloss'] = loss.item()
    # out['pose_solver_conf'] = pose_solver_conf

    # write_header = not os.path.isfile(f"./{os.environ['out_root']}/pgs3d_losses.csv")
    # with open(f"./{os.environ['out_root']}/pgs3d_losses.csv", 'a') as f:
    #     w = csv.DictWriter(f, out.keys())
    #     if write_header:
    #         w.writeheader()
    #     w.writerow(out)
    
    if model_mean_reverse is not None:
        return (model_mean, loss), (model_mean_reverse, loss_reverse), rescale_factor
    return (model_mean, loss), rescale_factor

def PGS3D_optimize(
    model_mean: torch.Tensor,
    t: int,
    uncropped_data: Dict,
    processed_matches: Dict,
    update_R: bool = True,
    update_T: bool = True,
    update_FL: bool = True,
    pose_encoding_type: str = "absT_quaR_logFL",
    alpha: float = 1e-2,
    learning_rate: float = 1e-2,
    iter_num: int = 20,
    viz = None,
    pose_scale=1,
    **kwargs,
):
    with torch.enable_grad():
        model_mean.requires_grad_(True)
        optimizer = torch.optim.SGD([model_mean], lr=learning_rate, momentum=0.9)
        batch_size = model_mean.shape[1]

        for i in range(iter_num):
            loss, rescale_factor = compute_point_distance(
                i,
                model_mean,
                t,
                uncropped_data,
                processed_matches,
                update_R=update_R,
                update_T=update_T,
                update_FL=update_FL,
                pose_encoding_type=pose_encoding_type,
                viz=viz,
                pose_scale=pose_scale
            )
        
            optimizer.zero_grad()
            loss.backward()

            grads = model_mean.grad
            grad_norm = grads.norm()
            grad_mask = (grads.abs() > 0).detach()
            model_mean_norm = (model_mean * grad_mask).norm()

            max_norm = alpha * model_mean_norm / learning_rate

            total_norm = torch.nn.utils.clip_grad_norm_(model_mean, max_norm)
            optimizer.step()

        model_mean = model_mean.detach()

    # try:
    #     pose_encoding_type = "absT_quaR_logFL"
    #     pred_cameras = pose_encoding_to_visdom(model_mean, pose_encoding_type)
    #     shape = torch.tensor([[224, 224], [224, 224]])
    #     gt_pose = convert_data_to_perspective_camera(uncropped_data)
    #     pred_R, pred_T, pred_K = opencv_from_visdom_projection(pred_cameras, shape)
    #     rel_pred_R = quaternion_multiply(matrix_to_quaternion(pred_R[0]), quaternion_invert(matrix_to_quaternion(pred_R[1])))
    #     gt_R, gt_T, gt_K = opencv_from_visdom_projection(gt_pose, shape)
    #     rel_gt_R = quaternion_multiply(matrix_to_quaternion(gt_R[0]), quaternion_invert(matrix_to_quaternion(gt_R[1])))
    #     print(f"{t}, {loss.item()}, {quat_angle_error(label=rel_pred_R, pred=rel_gt_R, variant='sin')[0, 0]}, {torch.norm(torch.subtract((pred_T[1] / rescale_factor) - pred_T[0], gt_T[1] - gt_T[0]))}")
    # except Exception as e:
    #     print(e)
    #     pass

    return model_mean, rescale_factor, loss


def compute_point_distance(
    i: int,
    model_mean: torch.Tensor,
    t: int,
    uncropped_data: Dict,
    processed_matches: Dict,
    update_R=True,
    update_T=True,
    update_FL=True,
    pose_encoding_type: str = "absT_quaR_logFL",
    viz = None,
    pose_scale = 1,
    try_to_reverse=True,
):
    model_mean = model_mean[0]
    cam1_T = model_mean[:, :3]
    cam1_scale = torch.norm(cam1_T)

    rescale_factor = cam1_scale / pose_scale
    if rescale_factor > 1.:
        rescale_factor = torch.tensor(1., device=rescale_factor.device)

    pred_cameras = pose_encoding_to_visdom(model_mean, pose_encoding_type)
    shape = torch.tensor([[224, 224], [224, 224]])
    gt_pose = convert_data_to_perspective_camera(uncropped_data)
    pred_R, pred_T, pred_K = opencv_from_visdom_projection(pred_cameras, shape)

    # F1 = pred_cameras.focal_length.mean(dim=0).repeat(len(camera1.focal_length), 1)

    if not update_R:
        pred_R = pred_R.detach()
        R0, R1 = pred_R
    else:
        R0, R1 = pred_R
        R0 = R0.detach()

    if not update_T:
        pred_T = pred_T.detach()
        T0, T1 = pred_T
    else:
        T0, T1 = pred_T
        T0 = T0.detach()

    if not update_FL:
        pred_K = pred_K.detach()

    relR, relT = relative_pose(R0, R1, T0, T1)
    kp1 = processed_matches['kp1']
    kp2 = processed_matches['kp2']
    mask = get_mask(uncropped_data, kp1, kp2).ravel() == 1
    loss = emat_solver_loss(i, relR, relT, uncropped_data, kp1, kp2, mask, pred_cameras, viz, gt_pose, rescale_factor)

    return loss, rescale_factor


def emat_solver_loss(i, R, t, data, kp1, kp2, mask, camera, viz, gt_pose, rescale_factor):
    # backproject E-mat inliers at each camera
    ransac_scale_threshold = 0.1

    K0 = data['K_color0'].squeeze(0)
    K1 = data['K_color1'].squeeze(0)

    if type(kp1) is torch.Tensor:
        kp1 = kp1.cpu().numpy()
        kp2 = kp2.cpu().numpy()

    inliers_kp1 = torch.from_numpy(kp1[mask]).to(dtype=torch.int32, device=R.device)
    inliers_kp2 = torch.from_numpy(kp2[mask]).to(dtype=torch.int32, device=R.device)
    depth_inliers_0 = data['depth0'][0, inliers_kp1[:, 1], inliers_kp1[:, 0]]
    depth_inliers_1 = data['depth1'][0, inliers_kp2[:, 1], inliers_kp2[:, 0]]

    valid = (depth_inliers_0 > 0) * (depth_inliers_1 > 0)
    xyz0 = backproject_3d_tensor(inliers_kp1[valid], depth_inliers_0[valid], K0).to(dtype=torch.float32, device=R.device)
    xyz1 = backproject_3d_tensor(inliers_kp2[valid], depth_inliers_1[valid], K1).to(dtype=torch.float32, device=R.device)

    xyz0 = ((R @ xyz0.clone().T).T)# + (t * rescale_factor)
    xyz0 += rescale_factor * t

    pointcloud1 = Pointclouds([xyz0])
    pointcloud2 = Pointclouds([xyz1])

    if viz is not None:
        cams_show = {"gt": gt_pose, "pred": camera, "xyz0": pointcloud1, "xyz1": pointcloud2} #
        fig = plot_scene({f"{2}": cams_show})
        viz.plotlyplot(fig, env="main", win=f"{2}")

    loss = (torch.abs(xyz1 - xyz0) ** 2).mean()
    return loss


def get_mask(data, kp1, kp2):
    ransac_pix_threshold = 2.0
    ransac_confidence = 0.9999

    if type(kp1) is torch.Tensor:
        kp1 = kp1.cpu().numpy()
        kp2 = kp2.cpu().numpy()

    K0 = data['K_color0'].squeeze(0).cpu().numpy()
    K1 = data['K_color1'].squeeze(0).cpu().numpy()

    # normalize keypoints
    kp1 = (kp1 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kp2 = (kp2 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    ransac_thr = ransac_pix_threshold / np.mean([K0[0, 0], K1[1, 1], K0[1, 1], K1[0, 0]])

    # compute pose with OpenCV
    _, mask = cv.findEssentialMat(
        kp1, kp2, np.eye(3),
        threshold=ransac_thr, prob=ransac_confidence, method=cv.USAC_MAGSAC)

    return mask


def relative_pose(R0, R1, T0, T1):
    relativeR_quat = quaternion_multiply(matrix_to_quaternion(R1), quaternion_invert(matrix_to_quaternion(R0)))
    relativeR = quaternion_to_matrix(relativeR_quat)
        
    relativeT = T1 - T0

    return relativeR, relativeT