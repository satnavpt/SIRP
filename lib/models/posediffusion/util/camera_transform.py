# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from pytorch3d.transforms.rotation_conversions import matrix_to_quaternion, quaternion_to_matrix
from pytorch3d.renderer.cameras import CamerasBase, PerspectiveCameras
import numpy as np
from pytorch3d.transforms import quaternion_invert, quaternion_multiply, RotateAxisAngle
from pytorch3d.utils import cameras_from_opencv_projection

from ..util.normalize_cameras import normalize_cameras
from transforms3d.quaternions import mat2quat
from transforms3d.quaternions import quat2mat
import cv2

def bbox_xyxy_to_xywh(xyxy):
    wh = xyxy[2:] - xyxy[:2]
    xywh = np.concatenate([xyxy[:2], wh])
    return xywh


def adjust_camera_to_bbox_crop_(fl, pp, image_size_wh: torch.Tensor, clamp_bbox_xywh: torch.Tensor):
    focal_length_px, principal_point_px = _convert_ndc_to_pixels(fl, pp, image_size_wh)
    principal_point_px_cropped = principal_point_px - clamp_bbox_xywh[:2]

    focal_length, principal_point_cropped = _convert_pixels_to_ndc(
        focal_length_px, principal_point_px_cropped, clamp_bbox_xywh[2:]
    )

    return focal_length, principal_point_cropped


def adjust_camera_to_image_scale_(fl, pp, original_size_wh: torch.Tensor, new_size_wh: torch.LongTensor):
    focal_length_px, principal_point_px = _convert_ndc_to_pixels(fl, pp, original_size_wh)

    # now scale and convert from pixels to NDC
    image_size_wh_output = new_size_wh.float()
    scale = (image_size_wh_output / original_size_wh).min(dim=-1, keepdim=True).values
    focal_length_px_scaled = focal_length_px * scale
    principal_point_px_scaled = principal_point_px * scale

    focal_length_scaled, principal_point_scaled = _convert_pixels_to_ndc(
        focal_length_px_scaled, principal_point_px_scaled, image_size_wh_output
    )
    return focal_length_scaled, principal_point_scaled

def transform_to_extrinsics(t):
    tr = t[3, :3]
    ro = t[:3, :3]
    return tr, ro

def convert_data_to_camera_intrinsics(data):
    images = torch.cat((data['image0'].clone(), data['image0'].clone()))
    K = torch.cat((data['K_color0'].clone(), data['K_color1'].clone())).to(dtype=torch.float32)
    fl = torch.stack((K[:, 0, 0], K[:, 1, 1])).T.to(dtype=torch.float32)
    pp = torch.stack((K[:, 0, 2], K[:, 1, 2])).T.to(dtype=torch.float32)
    shape = torch.from_numpy(np.array([images.shape[2], images.shape[3]])).to(device=K.device, dtype=torch.float32)

    image_size = shape = torch.cat((shape.unsqueeze(0), shape.unsqueeze(0)))

    shape = shape.flip(dims=(1,))
    
    scale = shape.min(dim=1, keepdim=True)[0] / 2.0
    scale = scale.expand(-1, 2)
    c0 = shape / 2.0

    focal_length = torch.stack([K[:, 0, 0], K[:, 1, 1]], dim=-1)
    principal_point = K[:, :2, 2]
    focal_pytorch3d = focal_length / scale
    p0_pytorch3d = -(principal_point - c0) / scale

    return focal_pytorch3d

    # return PerspectiveCameras(
    #     R=R_pytorch3d,
    #     T=T_pytorch3d,
    #     focal_length=focal_pytorch3d,
    #     principal_point=p0_pytorch3d,
    #     image_size=image_size,
    #     device=K.device,
    # )

def convert_data_to_perspective_camera(data):
    m = data['T_0to1'][0].clone()
    tr = m[:3, 3].permute(*torch.arange(m[:3, 3].ndim - 1, -1, -1)).unsqueeze(0)

    images = torch.cat((data['image0'].clone(), data['image0'].clone()))
    K = torch.cat((data['K_color0'].clone(), data['K_color1'].clone())).to(dtype=torch.float32)
    fl = torch.stack((K[:, 0, 0], K[:, 1, 1])).T.to(dtype=torch.float32)
    pp = torch.stack((K[:, 0, 2], K[:, 1, 2])).T.to(dtype=torch.float32)
    shape = torch.from_numpy(np.array([images.shape[2], images.shape[3]])).to(device=K.device, dtype=torch.float32)

    rotations_world_camera = torch.cat((quaternion_to_matrix(data['abs_q_0'].clone()), quaternion_to_matrix(data['abs_q_1'].clone()))).to(dtype=torch.float32)
    translations_world_camera = torch.cat((torch.zeros((1,3), device=K.device), tr)).to(dtype=torch.float32)

    image_size = shape = torch.cat((shape.unsqueeze(0), shape.unsqueeze(0)))

    shape = shape.to(rotations_world_camera).flip(dims=(1,))
    
    scale = shape.to(rotations_world_camera).min(dim=1, keepdim=True)[0] / 2.0
    scale = scale.expand(-1, 2)
    c0 = shape / 2.0

    focal_length = torch.stack([K[:, 0, 0], K[:, 1, 1]], dim=-1)
    principal_point = K[:, :2, 2]
    focal_pytorch3d = focal_length / scale
    p0_pytorch3d = -(principal_point - c0) / scale

    R_pytorch3d = rotations_world_camera.clone().permute(0, 2, 1)
    T_pytorch3d = translations_world_camera.clone()

    R_pytorch3d[:, :, 2] *= -1
    T_pytorch3d[:, 2] *= -1

    return PerspectiveCameras(
        R=R_pytorch3d,
        T=T_pytorch3d,
        focal_length=focal_pytorch3d,
        principal_point=p0_pytorch3d,
        image_size=image_size,
        device=K.device,
    )

def get_cropped_images(data, resize):
    # for K in ['K_color0', 'K_color1']:
        # new_K = correct_intrinsic_scale(data[K], resize[0] / W, resize[1] / H)
    images = []
    for im in ['image0', 'image1']:
        image = (data[im][0].clone() * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        image = cv2.resize(image, dsize=resize)
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255
        images.append(image.unsqueeze(0).to(data[im][0].device))

    return images
    # (h, w, 3) -> (3, h, w) and normalized


def convert_pose_solver_to_perspective_camera(rotations_world_camera, translations_world_camera, K, image_size):
    rotations_world_camera = rotations_world_camera.to(K.device)
    translations_world_camera = translations_world_camera.to(K.device)
    image_size = shape = torch.cat((image_size.unsqueeze(0), image_size.unsqueeze(0)))
    shape = shape.to(rotations_world_camera).flip(dims=(1,))

    # print(f"shape:  {shape}")
    
    scale = shape.to(rotations_world_camera).min(dim=1, keepdim=True)[0] / 2.0
    scale = scale.expand(-1, 2)
    c0 = shape / 2.0

    focal_length = torch.stack([K[:, 0, 0], K[:, 1, 1]], dim=-1)
    principal_point = K[:, :2, 2]
    focal_pytorch3d = focal_length / scale
    p0_pytorch3d = -(principal_point - c0) / scale



    R_pytorch3d = rotations_world_camera.clone().permute(0, 2, 1)
    T_pytorch3d = translations_world_camera.clone()
    # R_pytorch3d[:, :, :2] *= -1
    # T_pytorch3d[:, :2] *= -1

    R_pytorch3d[:, :, 2] *= -1
    T_pytorch3d[:, 2] *= -1

    # print("111111dlfjhdflhgjvdhf")
    # print(R_pytorch3d)
    # print([matrix_to_quaternion(r) for r in R_pytorch3d])

    return PerspectiveCameras(
        R=R_pytorch3d,
        T=T_pytorch3d,
        focal_length=focal_pytorch3d,
        principal_point=p0_pytorch3d,
        image_size=image_size,
        device=K.device,
    )

def opencv_from_visdom_projection(cameras, image_size):
    R_pytorch3d = cameras.R.clone()
    T_pytorch3d = cameras.T.clone()

    focal_pytorch3d = cameras.focal_length
    p0_pytorch3d = cameras.principal_point
    R_pytorch3d[:, :, 2] *= -1
    T_pytorch3d[:, 2] *= -1
    # T_pytorch3d[:, :2] *= -1
    # R_pytorch3d[:, :, :2] *= -1
    tvec = T_pytorch3d
    R = R_pytorch3d.permute(0, 2, 1)

    # Retype the image_size correctly and flip to width, height.
    image_size_wh = image_size.to(R).flip(dims=(1,))

    # NDC to screen conversion.
    scale = image_size_wh.to(R).min(dim=1, keepdim=True)[0] / 2.0
    scale = scale.expand(-1, 2)
    c0 = image_size_wh / 2.0

    principal_point = -p0_pytorch3d * scale + c0
    focal_length = focal_pytorch3d * scale

    camera_matrix = torch.zeros_like(R)
    camera_matrix[:, :2, 2] = principal_point
    camera_matrix[:, 2, 2] = 1.0
    camera_matrix[:, 0, 0] = focal_length[:, 0]
    camera_matrix[:, 1, 1] = focal_length[:, 1]
    return R, tvec, camera_matrix

# def pred_to_opencv(pred_cams, image_size):
#     # pred_R, pred_T, pred_K = opencv_from_cameras_projection(pred_cameras, shape)
#     R_pytorch3d = pred_cams.R.clone()
#     T_pytorch3d = pred_cams.T.clone()
#     focal_pytorch3d = pred_cams.focal_length
#     p0_pytorch3d = pred_cams.principal_point

#     T_pytorch3d[:, 0] *= -1
#     R_pytorch3d[:, :, 0] *= -1
#     tvec = T_pytorch3d
#     # R = R_pytorch3d.permute(0, 2, 1)

#     # Retype the image_size correctly and flip to width, height.
#     image_size_wh = image_size.to(R).flip(dims=(1,))

#     # NDC to screen conversion.
#     scale = image_size_wh.to(R).min(dim=1, keepdim=True)[0] / 2.0
#     scale = scale.expand(-1, 2)
#     c0 = image_size_wh / 2.0
    
#     principal_point = -p0_pytorch3d * scale + c0
#     focal_length = focal_pytorch3d * scale

#     # camera_matrix = torch.zeros_like(R)
#     # camera_matrix[:, :2, 2] = principal_point
#     # camera_matrix[:, 2, 2] = 1.0
#     # camera_matrix[:, 0, 0] = focal_length[:, 0]
#     # camera_matrix[:, 1, 1] = focal_length[:, 1]
#     # return R, tvec, camera_matrix

#     cameras = PerspectiveCameras(focal_length=focal_length, principal_point=principal_point, R=R, T=tvec, image_size=image_size, in_ndc=False)
#     return cameras
    # return cameras

# def convert_data_to_ndc(data):
#     m = data['T_0to1']
#     # print(m)
#     tr = m[0, :3, 3].T
#     ro = m[0, :3, :3]
#     print(tr)
#     print(matrix_to_quaternion(ro))
#     # print(m)
#     # m[0, 3, :3] = tr
#     # m[0, :3, 3] = 0
#     # m = m
#     # print(data['abs_q_0'])
#     # print(data['abs_c_0'])
#     # print(data['abs_q_1'])
#     # print(data['abs_c_1'])
#     # print(data['K_color0'])
#     # print(data['K_color1'])
#     # for i in [1]:
#     images = torch.cat((data['image0'], data['image0']))
#     rotations_world_camera = torch.cat((quaternion_to_matrix(data['abs_q_0']), quaternion_to_matrix(data['abs_q_1']))).to(dtype=torch.float32)
#     translations_world_camera = torch.cat((data['abs_c_0'], data['abs_c_1'])).to(dtype=torch.float32)
#     K = torch.cat((data['K_color0'], data['K_color1'])).to(dtype=torch.float32)
#     # print(rotations_world_camera)
#     # print(translations_world_camera)
#     fl = torch.stack((K[:, 0, 0], K[:, 1, 1])).T.to(dtype=torch.float32)
#     pp = torch.stack((K[:, 0, 2], K[:, 1, 2])).T.to(dtype=torch.float32)
#     shape = torch.from_numpy(np.array([images.shape[2], images.shape[3]])).to(dtype=torch.float32)
#     # print(shape)
#     # shape = np.array([[image.shape[1], image.shape[2]]])
#     # camera = PerspectiveCameras(focal_length = )
#     # ndc_transform = camera.get_ndc_camera_transform().get_matrix()
#     # print(ndc_transform)
#     # print(fl)
#     # print(pp)
#     # print(shape)
#     fl_ndc, pp_ndc = _convert_pixels_to_ndc(fl, pp, shape)
#     # print(fl_ndc)
#     # print(pp_ndc)
#     # cameras = PerspectiveCameras(focal_length=fl, principal_point=pp, R=rotations_world_camera, T=translations_world_camera, image_size=shape, in_ndc=False)
#     # # print(cameras.get_world_to_view_transform().get_matrix())
#     # normalised_cameras, scale = normalize_cameras(cameras)
#     # print(normalised_cameras.get_world_to_view_transform().get_matrix())
#     # print(scale)
#     # print(normalised_cameras.focal_length)


#     cameras = PerspectiveCameras(focal_length=fl_ndc, principal_point=pp_ndc, R=rotations_world_camera, T=translations_world_camera, image_size=shape)
#     # print(cameras.get_world_to_view_transform().get_matrix())
#     normalised_cameras, scale = normalize_cameras(cameras)
#     m = (normalised_cameras.get_world_to_view_transform().get_matrix())
#     # print(m)
#     # print(scale)

#     tr = m[1, 3, :3] - m[0, 3, :3]
#     ro = quaternion_multiply((matrix_to_quaternion(m[1, :3, :3])), quaternion_invert(matrix_to_quaternion(m[1, :3, :3])))
#     # print(tr)
#     # print(ro)
#     # qmult(q2, qinverse(q1))
#     # print(normalised_cameras.focal_length)

#         # rotation_world_camera_mat = quaternion_to_matrix(rotation_world_camera)
#         # m = torch.zeros((4,4))
#         # m[:3, :3] = rotation_world_camera_mat
#         # m[3, :3] = translation_world_camera
#         # m[3, 3] = 1
#         # print(m)
#         # print(ndc_stransform * m)

#     # and also t 0 to 1
#     # print(normalised_cameras.get_ndc_camera_transform().get_matrix())

#     # exit()

# # def ndc_to_mfl():


def _convert_ndc_to_pixels(focal_length: torch.Tensor, principal_point: torch.Tensor, image_size_wh: torch.Tensor):
    half_image_size = image_size_wh / 2
    rescale = half_image_size.min()
    principal_point_px = half_image_size - principal_point * rescale
    focal_length_px = focal_length * rescale
    return focal_length_px, principal_point_px


def _convert_pixels_to_ndc(
    focal_length_px: torch.Tensor, principal_point_px: torch.Tensor, image_size_wh: torch.Tensor
):
    half_image_size = image_size_wh / 2
    rescale = half_image_size.min()
    principal_point = (half_image_size - principal_point_px) / rescale
    focal_length = focal_length_px / rescale
    return focal_length, principal_point

def pose_encoding_to_camera(
    pose_encoding,
    pose_encoding_type="absT_quaR_logFL",
    # log_focal_length_bias=0.1, 
    log_focal_length_bias=1.8,
    min_focal_length=0.1,
    max_focal_length=20,
    return_dict=False,
):
    """
    Args:
        pose_encoding: A tensor of shape `BxNxC`, containing a batch of
                        `BxN` `C`-dimensional pose encodings.
        pose_encoding_type: The type of pose encoding,
                        only "absT_quaR_logFL" is supported.
    """

    pose_encoding_reshaped = pose_encoding.reshape(-1, pose_encoding.shape[-1])  # Reshape to BNxC

    if pose_encoding_type == "absT_quaR_logFL":
        # forced that 3 for absT, 4 for quaR, 2 logFL
        # TODO: converted to 1 dim for logFL, consistent with our paper
        abs_T = pose_encoding_reshaped[:, :3]
        quaternion_R = pose_encoding_reshaped[:, 3:7]
        R = quaternion_to_matrix(quaternion_R)

        log_focal_length = pose_encoding_reshaped[:, 7:9]

        # log_focal_length_bias was the hyperparameter
        # to ensure the mean of logFL close to 0 during training
        # Now converted back

        focal_length = (log_focal_length + log_focal_length_bias).exp()

        # clamp to avoid weird fl values
        focal_length = torch.clamp(focal_length, min=min_focal_length, max=max_focal_length)
    else:
        raise ValueError(f"Unknown pose encoding {pose_encoding_type}")

    if return_dict:
        return {"focal_length": focal_length, "R": R, "T": abs_T}

    pred_cameras = PerspectiveCameras(focal_length=focal_length, R=R, T=abs_T, device=R.device, in_ndc=False)
    return pred_cameras


def pose_encoding_to_visdom(
    pose_encoding,
    pose_encoding_type="absT_quaR_logFL",
    # log_focal_length_bias=0.1, 
    log_focal_length_bias=1.8,
    min_focal_length=0.1,
    max_focal_length=20,
    return_dict=False,
):
    """
    Args:
        pose_encoding: A tensor of shape `BxNxC`, containing a batch of
                        `BxN` `C`-dimensional pose encodings.
        pose_encoding_type: The type of pose encoding,
                        only "absT_quaR_logFL" is supported.
    """

    pose_encoding_reshaped = pose_encoding.reshape(-1, pose_encoding.shape[-1])  # Reshape to BNxC

    if pose_encoding_type == "absT_quaR_logFL":
        # forced that 3 for absT, 4 for quaR, 2 logFL
        # TODO: converted to 1 dim for logFL, consistent with our paper
        abs_T = pose_encoding_reshaped[:, :3]
        quaternion_R = pose_encoding_reshaped[:, 3:7]
        R = quaternion_to_matrix(quaternion_R)

        ###
        
        abs_T = abs_T.clone()
        abs_T[:, :2] = abs_T[:, :2] * -1
        R[:, :, :2] *= -1
        R = R.permute(0, 2, 1)

        ###

        R[:, :, 2] *= -1
        abs_T[:, 2] *= -1

        ###

        R = R.permute(0, 2, 1)
        R[:, :, :2] *= -1
        # abs_T[:, :2] *= -1

        ###

        # R[:, 2, :] *= -1
        # R[:, :, 1] *= -1
        # abs_T[:, :2] *= -1
        # R[:, :, 0] *= -1

        log_focal_length = pose_encoding_reshaped[:, 7:9]

        # log_focal_length_bias was the hyperparameter
        # to ensure the mean of logFL close to 0 during training
        # Now converted back

        focal_length = (log_focal_length + log_focal_length_bias).exp()


        # clamp to avoid weird fl values
        focal_length = torch.clamp(focal_length, min=min_focal_length, max=max_focal_length)
    else:
        raise ValueError(f"Unknown pose encoding {pose_encoding_type}")

    if return_dict:
        return {"focal_length": focal_length, "R": R, "T": abs_T}

    pred_cameras = PerspectiveCameras(focal_length=focal_length, R=R, T=abs_T, device=R.device, in_ndc=False)
    return pred_cameras


def camera_to_pose_encoding(
    camera, 
    pose_encoding_type="absT_quaR_logFL",
    # log_focal_length_bias=0.2, 
    log_focal_length_bias=1.8, 
    min_focal_length=0.1, 
    max_focal_length=20
):
    """ """

    if pose_encoding_type == "absT_quaR_logFL":
        r = camera.R.clone()
        t = camera.T.clone()

        quaternion_R = matrix_to_quaternion(r)

        # Calculate log_focal_length
        # print(f"test1: {camera.focal_length}")
        log_focal_length = (
            torch.log(torch.clamp(camera.focal_length, min=min_focal_length, max=max_focal_length))
            - log_focal_length_bias
        )
        # print(f"test2: {log_focal_length}")

        # Concatenate to form pose_encoding
        pose_encoding = torch.cat([t, quaternion_R, log_focal_length], dim=-1)

    else:
        raise ValueError(f"Unknown pose encoding {pose_encoding_type}")

    return pose_encoding

def pose_solver_camera_to_pose_encoding(
    camera, 
    pose_encoding_type="absT_quaR_logFL", 
    # log_focal_length_bias=0.2, 
    # log_focal_length_bias=1.8, 
    # min_focal_length=0.1, 
    # max_focal_length=20
):
    """ """

    if pose_encoding_type == "absT_quaR_logFL":
        # Convert rotation matrix to quaternion
        r = camera.R.clone()
        # r[:, 2] *= -1

        t = camera.T.clone()
        # t[:, 2] *= -1

        t[:, :2] *= -1
        r[:, :, :2] *= -1
        r = r.permute(0, 2, 1)

        ###

        r[:, :, 2] *= -1
        t[:, 2] *= -1

        ###

        r = r.permute(0, 2, 1)
        r[:, :, :2] *= -1

        quaternion_R = matrix_to_quaternion(r)

        # Calculate log_focal_length
        log_focal_length = (
            torch.log((camera.focal_length))
        )

        # Concatenate to form pose_encoding
        pose_encoding = torch.cat([t, quaternion_R, log_focal_length], dim=-1)

    else:
        raise ValueError(f"Unknown pose encoding {pose_encoding_type}")

    return pose_encoding