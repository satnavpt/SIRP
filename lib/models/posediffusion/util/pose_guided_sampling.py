# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from typing import Dict, List, Optional, Union
from .camera_transform import pose_encoding_to_camera, camera_to_pose_encoding, _convert_pixels_to_ndc, transform_to_extrinsics, convert_data_to_perspective_camera, convert_pose_solver_to_perspective_camera, opencv_from_visdom_projection, pose_encoding_to_visdom, get_cropped_images, pose_solver_camera_to_pose_encoding
from .get_fundamental_matrix import get_fundamental_matrices
from pytorch3d.renderer.cameras import CamerasBase, PerspectiveCameras
import os
from pytorch3d.transforms import se3_exp_map, se3_log_map, Transform3d, so3_relative_angle, quaternion_invert, quaternion_multiply, quaternion_apply
from pytorch3d.transforms.rotation_conversions import matrix_to_quaternion, quaternion_to_matrix
import time
from pytorch3d.vis.plotly_vis import plot_scene, AxisArgs
import numpy as np
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

def pose_guided_sampling(model_mean: torch.Tensor, t: int, pose_solver_output: torch.Tensor, r_weight=100, t_weight=10, f_weight=1e-2, uncropped_data: Dict = None, viz = None, disable_retry=None):
    def _to_device(tensor):
        return torch.from_numpy(tensor).to(device)

    def _to_homogeneous(tensor):
        return torch.nn.functional.pad(tensor, [0, 1], value=1)

    # conduct GGS
    # model_mean = PGS_optimize(model_mean, t, pose_solver_output)

    print(f"step: {t}")
    
    pose_encoding_type = "absT_quaR_logFL"

    shape = torch.from_numpy(np.array([224, 224]))
    shape = torch.stack((shape, shape))
    gt_R, gt_T, gt_K = opencv_from_visdom_projection(convert_data_to_perspective_camera(uncropped_data), shape)
    gt_R = matrix_to_quaternion(gt_R)

    # Optimize FL, R, and T separately
    model_mean, rescale_factor = PGS_optimize(
        model_mean, t, pose_solver_output, r_weight=r_weight, t_weight=t_weight, f_weight=f_weight, update_T=False, update_R=False, update_FL=True, uncropped_data=uncropped_data, viz=viz)  # only optimize FL

    # pred_R, pred_T, pred_K = opencv_from_visdom_projection(pose_encoding_to_visdom(model_mean, pose_encoding_type), shape)
    # pred_R = matrix_to_quaternion(pred_R)
    # print(f"actual loss: {torch.abs(gt_R - pred_R).sum() + torch.abs(gt_T - (pred_T / rescale_factor)).sum()}")

    model_mean, rescale_factor = PGS_optimize(
        model_mean, t, pose_solver_output, r_weight=r_weight, t_weight=t_weight, f_weight=f_weight, update_T=False, update_R=True, update_FL=False, uncropped_data=uncropped_data, viz=viz)  # only optimize R

    # pred_R, pred_T, pred_K = opencv_from_visdom_projection(pose_encoding_to_visdom(model_mean, pose_encoding_type), shape)
    # pred_R = matrix_to_quaternion(pred_R)
    # print(f"actual loss: {torch.abs(gt_R - pred_R).sum() + torch.abs(gt_T - (pred_T / rescale_factor)).sum()}")

    model_mean, rescale_factor = PGS_optimize(
        model_mean, t, pose_solver_output, r_weight=r_weight, t_weight=t_weight, f_weight=f_weight, update_T=True, update_R=False, update_FL=False, uncropped_data=uncropped_data, viz=viz)  # only optimize T

    # pred_R, pred_T, pred_K = opencv_from_visdom_projection(pose_encoding_to_visdom(model_mean, pose_encoding_type), shape)
    # pred_R = matrix_to_quaternion(pred_R)
    # print(f"actual loss: {torch.abs(gt_R - pred_R).sum() + torch.abs(gt_T - (pred_T / rescale_factor)).sum()}")

    model_mean, rescale_factor = PGS_optimize(model_mean, t, pose_solver_output, r_weight=r_weight, t_weight=t_weight, f_weight=f_weight, uncropped_data=uncropped_data, viz=viz)

    # pred_R, pred_T, pred_K = opencv_from_visdom_projection(pose_encoding_to_visdom(model_mean, pose_encoding_type), shape)
    # pred_R = matrix_to_quaternion(pred_R)
    # print(f"actual loss: {torch.abs(gt_R - pred_R).sum() + torch.abs(gt_T - (pred_T / rescale_factor)).sum()}")

    return model_mean, rescale_factor


def PGS_optimize(
    model_mean: torch.Tensor,
    t: int,
    pose_solver_output: torch.Tensor,
    uncropped_data: Dict,
    r_weight=100,
    t_weight=10,
    f_weight=1e-2,
    update_R: bool = True,
    update_T: bool = True,
    update_FL: bool = True,
    pose_encoding_type: str = "absT_quaR_logFL",
    alpha: float = 1e-2,
    learning_rate: float = 1e-2,
    iter_num: int = 10,
    viz = None,
    **kwargs,
):
    with torch.enable_grad():
        model_mean.requires_grad_(True)

        # if update_R and update_T and update_FL:
        #     iter_num = iter_num * 2

        optimizer = torch.optim.SGD([model_mean], lr=learning_rate, momentum=0.9)
        batch_size = model_mean.shape[1]

        for _ in range(iter_num):
            loss, rescale_factor = compute_pose_distance(
                model_mean,
                t,
                uncropped_data,
                pose_solver_output,
                r_weight=r_weight,
                t_weight=t_weight,
                f_weight=f_weight,
                update_R=update_R,
                update_T=update_T,
                update_FL=update_FL,
                pose_encoding_type=pose_encoding_type,
                viz=viz,
            )

            if loss.item() > 200:
                break
            
            optimizer.zero_grad()
            loss.backward()

            grads = model_mean.grad
            grad_norm = grads.norm()
            grad_mask = (grads.abs() > 0).detach()
            model_mean_norm = (model_mean * grad_mask).norm()

            max_norm = alpha * model_mean_norm / learning_rate

            total_norm = torch.nn.utils.clip_grad_norm_(model_mean, max_norm)
            optimizer.step()

            # print(f"R: {rescale_factor}")

        model_mean = model_mean.detach()
        # camera1 = pose_encoding_to_camera(model_mean, pose_encoding_type)
        # R, T = camera1.R, camera1.T
        # relR, relT = relative_pose(R,T)
        # print(f"scale 0: {torch.norm(relT[0])}")
        # print(f"S: {torch.norm(relT)}")
    # print(f"L: {loss.item()}")
    pose_encoding_type = "absT_quaR_logFL"
    pred_cameras = pose_encoding_to_visdom(model_mean, pose_encoding_type)
    shape = torch.tensor([[224, 224], [224, 224]])
    gt_pose = convert_data_to_perspective_camera(uncropped_data)
    pred_R, pred_T, pred_K = opencv_from_visdom_projection(pred_cameras, shape)
    rel_pred_R = quaternion_multiply(matrix_to_quaternion(pred_R[0]), quaternion_invert(matrix_to_quaternion(pred_R[1])))
    gt_R, gt_T, gt_K = opencv_from_visdom_projection(gt_pose, shape)
    rel_gt_R = quaternion_multiply(matrix_to_quaternion(gt_R[0]), quaternion_invert(matrix_to_quaternion(gt_R[1])))

    # print(f"r: {quat_angle_error(label=rel_pred_R, pred=rel_gt_R, variant='sin')[0, 0]}")
    print(f"t: {torch.norm(torch.subtract((pred_T[1] / rescale_factor) - pred_T[0], gt_T[1] - gt_T[0]))}")

    return model_mean, rescale_factor


# def compute_relative_pose_distance(
#     model_mean: torch.Tensor,
#     t: int,
#     pose_solver_output: Dict,
#     update_R=True,
#     update_T=True,
#     update_FL=True,
#     pose_encoding_type: str = "absT_quaR_logFL",
# ):
#     camera1 = pose_encoding_to_camera(model_mean, pose_encoding_type)
#     camera2 = pose_encoding_to_camera(pose_solver_output, pose_encoding_type)

#     # pick the mean of the predicted focal length
#     F1 = camera1.focal_length.mean(dim=0).repeat(len(camera1.focal_length), 1)
#     F2 = camera2.focal_length.mean(dim=0).repeat(len(camera2.focal_length), 1)

#     if not update_R:
#         camera1.R = camera1.R.detach()

#     if not update_T:
#         camera1.T = camera1.T.detach()

#     if not update_FL:
#         camera1.focal_length = camera1.focal_length.detach()

#     R1, T1 = camera1.R, camera1.T
#     relR1, relT1 = relative_pose(R1, T1)
#     R2, T2 = camera2.R, camera2.T
#     relR2, relT2 = relative_pose(R2, T2)

#     disR = torch.norm(torch.abs(relR1 - relR2))
#     disT = torch.norm(torch.abs(relT1 - relT2))
#     disF = torch.norm(torch.abs(F1 - F2)[0])

#     loss = disR + disT + disF
#     return loss

def compute_pose_distance(
    model_mean: torch.Tensor,
    t: int,
    uncropped_data: Dict,
    pose_solver_output: Dict,
    r_weight=100,
    t_weight=10,
    f_weight=1e-2,
    update_R=True,
    update_T=True,
    update_FL=True,
    pose_encoding_type: str = "absT_quaR_logFL",
    viz = None,
):
    model_mean = model_mean[0]
    cam1_T = model_mean[:, :3]
    cam1_scale = torch.norm(cam1_T)
    cam1_F = model_mean[:, -2:]

    # print(model_mean)

    camera1 = pose_encoding_to_visdom(model_mean, pose_encoding_type)

    pose_solver_output_c = pose_solver_output.clone()
    cam2_T = pose_solver_output_c[:, :3]
    cam2_scale = torch.norm(cam2_T)
    cam2_F = pose_solver_output_c[:, -2:]

    # print(cam1_T)
    # print(cam1_F)
    # print(cam1_scale)
    # print(cam2_T)
    # print(cam2_F)
    # print(cam2_scale)

    rescale_factor = cam1_scale / cam2_scale

    pose_solver_output_c[:, :3] *= rescale_factor

    camera2 = pose_encoding_to_visdom(pose_solver_output_c, pose_encoding_type)


    gt_pose = convert_data_to_perspective_camera(uncropped_data)


    # pick the mean of the predicted focal length
    F1 = camera1.focal_length.mean(dim=0).repeat(len(camera1.focal_length), 1)
    F2 = camera2.focal_length.mean(dim=0).repeat(len(camera2.focal_length), 1)

    if not update_R:
        camera1.R = camera1.R.detach()

    if not update_T:
        camera1.T = camera1.T.detach()

    if not update_FL:
        camera1.focal_length = camera1.focal_length.detach()

    R1, T1 = camera1.R, camera1.T
    R2, T2 = camera2.R, camera2.T

    disR = torch.norm(torch.abs(R1 - R2)) * r_weight #100
    disT = torch.norm(torch.abs(T1 - T2)) * t_weight #10
    disF = torch.norm(torch.abs(F1 - F2)[0]) * f_weight #1e-3

    loss = (disR) + (disT) + (disF)

    if viz is not None:
        cams_show = {"gt": gt_pose, "pred": camera1, "pose_solver": camera2}
        fig = plot_scene({f"{2}": cams_show})
        viz.plotlyplot(fig, env="main", win=f"{2}")

    return (loss, rescale_factor)


def relative_pose(R, T):
    relativeR_quat = quaternion_multiply(matrix_to_quaternion(R[1]), quaternion_invert(matrix_to_quaternion(R[0])))
    relativeR = quaternion_to_matrix(relativeR_quat)
        
    relativeT = T[1] - T[0]

    return relativeR, relativeT