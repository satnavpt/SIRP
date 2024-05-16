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

def resolve_scale(pred_cameras, data, kp1, kp2, shape):
    # distance between means
    pred_R, pred_T, _ = opencv_from_visdom_projection(pred_cameras, shape)

    R0, R1 = pred_R
    T0, T1 = pred_T
    R, t = relative_pose(R0, R1, T0, T1)

    try:
        mask = get_mask(data, kp1, kp2).ravel() == 1
    except Exception as e:
        print("failed to resolve scale")
        return R, t

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

    xyz0 = (R @ xyz0.T).T

    pmean0 = torch.mean(xyz0, axis=0)
    pmean1 = torch.mean(xyz1, axis=0)
    vec = pmean1 - pmean0

    # print(vec)
    # print(torch.norm(vec))
    # print(torch.norm(t))

    scale = torch.norm(vec) / torch.norm(t)

    t_metric = scale * t
    t_metric = t_metric.reshape(3, 1)

    # print(t_metric)

    # print(f"scale: {scale}")

    return R, t_metric


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