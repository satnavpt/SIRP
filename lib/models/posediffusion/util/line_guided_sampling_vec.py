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


class HighLossException(Exception):
    def __init__(self, loss):
        self.loss = loss

def line_guided_sampling_vec(model_mean: torch.Tensor, t: int, disable_retry, matches_dict: Dict, uncropped_data: Dict, viz, flip):
    device = model_mean.device

    def _to_device(tensor):
        return torch.from_numpy(np.array(tensor)).to(device)

    kp1 = _to_device(matches_dict["kp1"])
    kp2 = _to_device(matches_dict["kp2"])
    l1 = _to_device(matches_dict["li1"])
    l2 = _to_device(matches_dict["li2"])
    
    processed_matches = {
        "kp1": kp1,
        "kp2": kp2,
        "l1": l1,
        "l2": l2,
    }

    def _to_homogeneous(tensor):
        return torch.nn.functional.pad(tensor, [0, 1], value=1)

    # print(f"step: {t}")
    pose_encoding_type = "absT_quaR_logFL"

    # print(f"points: {processed_matches['kp1'].shape[0]}")
    # print(f"lines: {processed_matches['l1'].shape[0]}")

    # optimise
    model_mean_reverse = None
    if flip and t == 9:
        model_mean, rescale_factor, loss = LGSV_optimize(model_mean, t, uncropped_data, processed_matches, viz=viz)

        model_mean_reverse = model_mean.detach().clone()
        r = quaternion_to_matrix(model_mean_reverse[0, 1, 3:7])
        model_mean_reverse[0, 1, 3:7] = matrix_to_quaternion(r)
        model_mean_reverse[0, 1, 0] *= -1

        model_mean_reverse, rescale_factor_reverse, loss_reverse = LGSV_optimize(model_mean_reverse, t, uncropped_data, processed_matches, viz=viz)
    else:
        model_mean, rescale_factor, loss = LGSV_optimize(model_mean, t, uncropped_data, processed_matches, viz=viz)
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
    # out['lgs3dloss'] = loss.item()
    # out['num_points'] = processed_matches["kp1"].shape[0]
    # out['num_lines'] = processed_matches["l1"].shape[0]

    # write_header = not os.path.isfile(f"./{os.environ['out_root']}/lgs_losses.csv")
    # with open(f"./{os.environ['out_root']}/lgs_losses.csv", 'a') as f:
    #     w = csv.DictWriter(f, out.keys())
    #     if write_header:
    #         w.writeheader()
    #     w.writerow(out)

    if model_mean_reverse is not None:
        return (model_mean, loss), (model_mean_reverse, loss_reverse), rescale_factor
    return (model_mean, loss), rescale_factor

def LGSV_optimize(
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
    iter_num: int = 30,
    viz = None,
    pose_scale=1,
    **kwargs,
):
    with torch.enable_grad():
        model_mean.requires_grad_(True)
        optimizer = torch.optim.SGD([model_mean], lr=learning_rate, momentum=0.9)
        batch_size = model_mean.shape[1]

        for i in range(iter_num):
            ll, pl, rescale_factor = compute_line_distance(
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

            loss = ll + pl
        
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
    print(f"{(ll + pl).item()}, {(ll).item()}, {(pl).item()}")

    return model_mean, rescale_factor, loss


def compute_line_distance(
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
    # rescale_factor = torch.tensor(1., device=model_mean.device)

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
    l1 = processed_matches['l1']
    l2 = processed_matches['l2']
    ll, pl = line_vector_loss(i, relR, relT, uncropped_data, kp1, kp2, l1, l2, pred_cameras, viz, gt_pose, rescale_factor)

    return ll, pl, rescale_factor

def line_vector_loss(i, R, t, data, kp1, kp2, l1, l2, camera, viz, gt_pose, rescale_factor):
    # backproject E-mat inliers at each camera
    ransac_scale_threshold = 0.1

    K0 = data['K_color0'].squeeze(0).to(torch.float32)
    K1 = data['K_color1'].squeeze(0).to(torch.float32)
    # mask = get_mask(data, all_points_1, all_points_2).ravel() == 1

    if type(kp1) is torch.Tensor:
        kp1 = kp1.cpu().numpy()
        kp2 = kp2.cpu().numpy()
        l1 = l1.cpu().numpy()
        l2 = l2.cpu().numpy()

    # inliers_kp1 = torch.from_numpy(kp1[mask]).to(dtype=torch.int32, device=R.device)
    # inliers_kp2 = torch.from_numpy(kp2[mask]).to(dtype=torch.int32, device=R.device)
    # depth_inliers_0 = data['depth0'][0, inliers_kp1[:, 1], inliers_kp1[:, 0]]
    # depth_inliers_1 = data['depth1'][0, inliers_kp2[:, 1], inliers_kp2[:, 0]]

    # valid = (depth_inliers_0 > 0) * (depth_inliers_1 > 0)
    # xyz0 = backproject_3d_tensor(inliers_kp1[valid], depth_inliers_0[valid], K0).to(dtype=torch.float32, device=R.device)
    # xyz1 = backproject_3d_tensor(inliers_kp2[valid], depth_inliers_1[valid], K1).to(dtype=torch.float32, device=R.device)

    # xyz0 = ((R @ xyz0.clone().T).T)# + (t * rescale_factor)
    # xyz0 += rescale_factor * t

    # point_loss = (torch.abs(xyz1 - xyz0) ** 2).mean()

    s = l1.shape
    l1 = l1.reshape(s[0], 2, 2)
    l2 = l2.reshape(s[0], 2, 2)

    l1[:, :, 1] = np.clip(l1[:, :, 1], 0, data['depth0'].shape[1]-1)
    l1[:, :, 0] = np.clip(l1[:, :, 0], 0, data['depth0'].shape[2]-1)

    l2[:, :, 1] = np.clip(l2[:, :, 1], 0, data['depth1'].shape[1]-1)
    l2[:, :, 0] = np.clip(l2[:, :, 0], 0, data['depth1'].shape[2]-1)

    l1_starts = l1[:, 0, :]
    l1_ends = l1[:, 1, :]
    l2_starts = l2[:, 0, :]
    l2_ends = l2[:, 1, :]

    depth_l1_starts = data['depth0'][0, l1_starts[:, 1], l1_starts[:, 0]]

    l1_starts_3d = backproject_3d_tensor(torch.from_numpy(l1_starts).to(dtype=torch.float32, device=K0.device), depth_l1_starts, K0).to(dtype=torch.float32, device=R.device)
    l1_starts_3d = ((R @ l1_starts_3d.T).T)
    l1_starts_3d += (t * rescale_factor)

    depth_l1_ends = data['depth0'][0, l1_ends[:, 1], l1_ends[:, 0]]
    l1_ends_3d = backproject_3d_tensor(torch.from_numpy(l1_ends).to(dtype=torch.float32, device=K0.device), depth_l1_ends, K0).to(dtype=torch.float32, device=R.device)
    l1_ends_3d = ((R @ l1_ends_3d.T).T)
    l1_ends_3d += (t * rescale_factor)

    l1_vecs = l1_ends_3d - l1_starts_3d
    # n = torch.norm(l1_vecs, dim=1)
    # n = torch.stack([n, n, n], dim=-1)
    # l1_vecs = torch.div(l1_vecs, n)

    depth_l2_starts = data['depth1'][0, l2_starts[:, 1], l2_starts[:, 0]]
    l2_starts_3d = backproject_3d_tensor(torch.from_numpy(l2_starts).to(dtype=torch.float32, device=K0.device), depth_l2_starts, K0).to(dtype=torch.float32, device=R.device)

    depth_l2_ends = data['depth1'][0, l2_ends[:, 1], l2_ends[:, 0]]
    l2_ends_3d = backproject_3d_tensor(torch.from_numpy(l2_ends).to(dtype=torch.float32, device=K0.device), depth_l2_ends, K0).to(dtype=torch.float32, device=R.device)

    l2_vecs = l2_ends_3d - l2_starts_3d
    # n = torch.norm(l2_vecs, dim=1)
    # n = torch.stack([n, n, n], dim=-1)
    # l2_vecs = torch.div(l2_vecs, n)

    line_loss = (torch.abs(l2_vecs - l1_vecs) ** 2).mean()

    l1startspc = Pointclouds([l1_starts_3d])
    l1endspc = Pointclouds([l1_ends_3d])
    l2startspc = Pointclouds([l2_starts_3d])
    l2endspc = Pointclouds([l2_ends_3d])

    if viz is not None:
        cams_show = {"gt": gt_pose, "pred": camera, "l1s": l1startspc, "l1e": l1endspc, "l2s": l2startspc, "l2e": l2endspc} #
        fig = plot_scene({f"{2}": cams_show})
        viz.plotlyplot(fig, env="main", win=f"{2}")
    
    pl = (torch.abs(l2_starts_3d - l1_starts_3d) ** 2).mean()  # + (torch.abs(l2_ends_3d - l1_ends_3d) ** 2).mean()

    # print(f"point: {pl.item()}, line: {line_loss.item()}")
    return line_loss, pl

def sample(lines1, lines2):
    l1_points = []
    l2_points = []
    for (point1A, point1B), (point2A, point2B) in zip(lines1, lines2):
        p1 = []
        p2 = []

        vec1 = point1B - point1A
        vec2 = point2B - point2A
        num_samples = np.floor(np.linalg.norm(vec1)).astype(np.int32)
        vec1_n = vec1 / num_samples
        vec2_n = vec2 / num_samples
        p1.append(point1A)
        p2.append(point2A)
        for s in range(1, num_samples):
            p1.append(point1A + (vec1_n * s))
            p2.append(point2A + (vec2_n * s))
        # TODO: code to ensure last sample != pointB
        p1.append(point1B)
        p2.append(point2B)

        l1_points.append(np.round(np.array(p1)))
        l2_points.append(np.round(np.array(p2)))

    return l1_points, l2_points

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