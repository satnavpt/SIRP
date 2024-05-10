# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Standard library imports
import base64
import io
import logging
import math
import os
import pickle
import tempfile
import warnings
from collections import defaultdict
from dataclasses import field, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from omegaconf import OmegaConf, DictConfig
from functools import partial
import time
import csv

# Third-party library imports
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.transforms import se3_exp_map, se3_log_map, Transform3d, so3_relative_angle, quaternion_invert, quaternion_multiply, quaternion_apply
from pytorch3d.vis.plotly_vis import plot_scene, AxisArgs
from pytorch3d.utils import opencv_from_cameras_projection, cameras_from_opencv_projection

from ..util.camera_transform import pose_encoding_to_camera, camera_to_pose_encoding, _convert_pixels_to_ndc, transform_to_extrinsics, convert_data_to_perspective_camera, convert_pose_solver_to_perspective_camera, opencv_from_visdom_projection, pose_encoding_to_visdom, get_cropped_images, pose_solver_camera_to_pose_encoding
from ..util.normalize_cameras import normalize_cameras
from ..util.match_extraction import extract_match_memory
from ..util.load_img_folder import preprocess_images
from ..util.geometry_guided_sampling import geometry_guided_sampling
from ..util.pose_guided_sampling import pose_guided_sampling
from ..util.point_guided_sampling import point_guided_sampling, HighLossException
from ..util.line_guided_sampling import line_guided_sampling
from lib.models.matching.feature_matching import *
from lib.models.matching.pose_solver import *
from lib.models.matching.model import FeatureMatchingModel

from .. import model
from hydra.utils import instantiate
from pytorch3d.renderer.cameras import PerspectiveCameras

from transforms3d.quaternions import mat2quat, quat2mat

from visdom import Visdom

from pytorch3d.ops import corresponding_cameras_alignment 

from typing import NamedTuple


logger = logging.getLogger(__name__)

class PoseDiffusionModel(nn.Module):
    def __init__(self, pose_encoding_type: str, IMAGE_FEATURE_EXTRACTOR: Dict, DIFFUSER: Dict, DENOISER: Dict, cfg: Dict = None):
        """Initializes a PoseDiffusion model.

        Args:
            pose_encoding_type (str):
                Defines the encoding type for extrinsics and intrinsics
                Currently, only `"absT_quaR_logFL"` is supported -
                a concatenation of the translation vector,
                rotation quaternion, and logarithm of focal length.
            image_feature_extractor_cfg (Dict):
                Configuration for the image feature extractor.
            diffuser_cfg (Dict):
                Configuration for the diffuser.
            denoiser_cfg (Dict):
                Configuration for the denoiser.
        """

        super().__init__()

        self.pose_encoding_type = pose_encoding_type

        self.image_feature_extractor = instantiate(IMAGE_FEATURE_EXTRACTOR, _recursive_=False)
        self.diffuser = instantiate(DIFFUSER, _recursive_=False)

        denoiser = instantiate(DENOISER, _recursive_=False)
        self.diffuser.model = denoiser

        self.target_dim = denoiser.target_dim

        self.apply(self._init_weights)

        if cfg.POSE_SOLVER:
            self.pose_solver = FeatureMatchingModel(cfg)
            self.pose_solver_2 = EssentialMatrixMetricSolver(cfg.EMAT_RANSAC)
        else:
            self.pose_solver = None

        self.GGS = cfg.GGS
        self.PGS = cfg.PGS
        self.PGS3D = cfg.PGS3D
        self.LGS = cfg.LGS
        self.GT_ALIGN = cfg.GT_ALIGN
        self.POSE_ALIGN = cfg.POSE_ALIGN
        self.INIT_POSE = cfg.INIT_POSE
        self.DIFF_CONF = cfg.DIFF_CONF

        self.DIFFUSER = DIFFUSER
        self.DENOISER = DENOISER

        self.crop = cfg.image_size

        if torch.cuda.is_available():
            self.viz = None #Visdom()
        else:
            self.viz = Visdom()
        self.i = 0
        self.ransac_scale_threshold = cfg.EMAT_RANSAC.SCALE_THRESHOLD
        self.ransac_pix_threshold = cfg.EMAT_RANSAC.PIX_THRESHOLD
        self.ransac_confidence = cfg.EMAT_RANSAC.CONFIDENCE
        # if self.PGS3D.enable:
        if cfg.FEATURE_MATCHING == 'SIFT':
            self.fm = SIFTMatching(cfg)
        elif cfg.FEATURE_MATCHING == 'Precomputed':
            self.fm = PrecomputedMatching(cfg)
        elif cfg.FEATURE_MATCHING == "HLOC":
            self.fm = HLOCMatching(cfg)
        elif cfg.FEATURE_MATCHING == "GLUE":
            self.fm = GLUEMatching(cfg)
        elif cfg.FEATURE_MATCHING == "DUST3R":
            self.fm = DUST3RMatching(cfg)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, uncropped_data):
        r, q = get_cropped_images(uncropped_data, resize=(self.crop, self.crop))
        images, image_info = preprocess_images(torch.cat((r, q), dim=0))
        original_images, original_image_info = preprocess_images(torch.cat((uncropped_data['image0'], uncropped_data['image1']), dim=0))
        pose_solver_conf = None

        if self.pose_solver is not None:
            K = torch.cat((uncropped_data['K_color0'], uncropped_data['K_color1'])).to(dtype=torch.float32)
            id = torch.tensor([uncropped_data['image0'].shape[1], uncropped_data['image0'].shape[2]])
            R, T = self.pose_solver.forward(uncropped_data)
            R = R.float().squeeze()
            T = T.float().reshape(-1)
            
            I = torch.eye(3)
            Rs = torch.stack((I, R))
            Ts = torch.stack((torch.zeros(3), T))
            if torch.isnan(R).any() or torch.isnan(T).any():
                pose_solver_output_1 = None
                print("pose solver output contains nans")
            else:
                pose_solver_output_1 = convert_pose_solver_to_perspective_camera(Rs, Ts, K, id)
                pose_solver_conf = uncropped_data['inliers']
                print("pose solver output doesnt contain nans")

            pose_solver_output_2 = None #convert_pose_solver_to_perspective_camera(Rs, Ts, K, id)
            if self.INIT_POSE.enable:
                if pose_solver_output_1 is None:
                    init_poses = None
                else:
                    init_poses = pose_solver_camera_to_pose_encoding(pose_solver_output_1).unsqueeze(0)
            else:
                init_poses = None
        else:
            pose_solver_output_1 = None
            pose_solver_output_2 = None
            init_poses = None

        cond_fn = None

        if self.GGS.enable:
            # Optional TODO: remove the keypoints outside the cropped region?
            # kp1, kp2, i12 = extract_match_memory(images=images, image_info=image_info)
            kp1, kp2 = self.fm.get_correspondences(uncropped_data)
            i12 = np.repeat(np.array([[0., 1.]]), len(kp1), axis=0)

            # kp1, kp2, i12 = extract_match_memory(images=images, image_info=image_info)

            # key points in each image and correspondences
            if (kp1 is not None) and len(kp1) != 0:
                keys = ["kp1", "kp2", "i12", "img_shape"]
                values = [kp1, kp2, i12, images.shape]
                matches_dict = dict(zip(keys, values))

                self.GGS.pose_encoding_type = self.pose_encoding_type
                GGS_cfg = OmegaConf.to_container(self.GGS)

                fn = (partial(geometry_guided_sampling, matches_dict=matches_dict, GGS_cfg=GGS_cfg), [10, 0])
                if cond_fn is None:
                    cond_fn = fn
                elif type(cond_fn) is list:
                    cond_fn = cond_fn.append(fn)
                else:
                    cond_fn = [cond_fn, fn]
            else:
                f = open(f"./{os.environ['out_root']}/ggs_results.txt", "a")
                f.write(f"insufficient key points found\n")
                f.close()
        if self.PGS.enable:
            if pose_solver_output_1 is not None:
                print("pgs is enabled")
                self.PGS.pose_encoding_type = self.pose_encoding_type
                fn = (partial(pose_guided_sampling, pose_solver_output=pose_solver_camera_to_pose_encoding(pose_solver_output_1), r_weight=self.PGS.r_weight, t_weight=self.PGS.t_weight, f_weight=self.PGS.f_weight, uncropped_data=uncropped_data, viz=self.viz), [10,0])
                if cond_fn is None:
                    cond_fn = fn
                elif type(cond_fn) is list:
                    cond_fn = cond_fn.append(fn)
                else:
                    cond_fn = [cond_fn, fn]
            else:
                print("pgs is not enabled")
        if self.PGS3D.enable:
            # kp1, kp2, i12 = extract_match_memory(images=original_images, image_info=original_image_info)
            kp1, kp2 = self.fm.get_correspondences(uncropped_data)
            i12 = np.repeat(np.array([[0., 1.]]), len(kp1), axis=0)

            if (kp1 is not None) and (len(kp1) != 0) and (pose_solver_conf is not None):# and (pose_solver_conf > 10):
                if pose_solver_conf <= 5:
                    return None, None, None
                if pose_solver_conf > 10:
                    keys = ["kp1", "kp2", "i12", "img_shape"]
                    values = [kp1, kp2, i12, images.shape]
                    matches_dict = dict(zip(keys, values))

                    self.PGS3D.pose_encoding_type = self.pose_encoding_type
                    if pose_solver_output_1 is not None:
                        cam2_scale = torch.norm(pose_solver_output_1.T)
                        # cam2_scale = cam2_scale.detach()
                        fn = (partial(point_guided_sampling, matches_dict=matches_dict, uncropped_data=uncropped_data, viz=self.viz, pose_scale=cam2_scale, pose_solver_conf=pose_solver_conf), [10, 0])
                        if cond_fn is None:
                            cond_fn = fn
                        elif type(cond_fn) is list:
                            cond_fn = cond_fn.append(fn)
                        else:
                            cond_fn = [cond_fn, fn]
        if self.LGS.enable:
            # kp1, kp2, i12 = extract_match_memory(images=original_images, image_info=original_image_info)
            kp1, kp2, li1, li2 = self.fm.get_correspondences(uncropped_data)

            keys = ["kp1", "kp2", "li1", "li2", "img_shape"]
            values = [kp1, kp2, li1, li2, images.shape]
            matches_dict = dict(zip(keys, values))

            self.LGS.pose_encoding_type = self.pose_encoding_type
            fn = (partial(line_guided_sampling, matches_dict=matches_dict, uncropped_data=uncropped_data, viz=self.viz), [10, 0])
            if cond_fn is None:
                cond_fn = fn
            elif type(cond_fn) is list:
                cond_fn = cond_fn.append(fn)
            else:
                cond_fn = [cond_fn, fn]


        training = False
        images = images.unsqueeze(0)
        
        result = self._forward(image=images.to(device=r.device), cond_fn=cond_fn, training=False, denoise_init=init_poses, data=uncropped_data, pose_solver_output=(pose_solver_output_1, pose_solver_output_2), pose_solver_conf=pose_solver_conf)
        if result is None:
            return None, None, None
        pred_cameras, conf = result
        shape = torch.from_numpy(np.array(images.shape[-2:]))
        shape = torch.stack((shape, shape))
        uncropped_data['inliers'] = 0

        pred_R, pred_T, pred_K = opencv_from_visdom_projection(pred_cameras, shape)

        relativeR_quat = quaternion_multiply(torch.from_numpy(mat2quat(pred_R[1].cpu().numpy())), quaternion_invert(torch.from_numpy(mat2quat(pred_R[0].cpu().numpy()))))
        relativeR = torch.from_numpy(quat2mat(relativeR_quat.cpu())).unsqueeze(0)
        
        relativeT = pred_T[1] - pred_T[0]

        return relativeR, relativeT, conf


    def _forward(
        self,
        image: torch.Tensor,
        gt_cameras: Optional[CamerasBase] = None,
        sequence_name: Optional[List[str]] = None,
        cond_fn=None,
        training=True,
        batch_repeat=-1,
        denoise_init=None,
        data=None,
        pose_solver_output = None,
        pose_solver_conf = None,
    ):
        """
        Forward pass of the PoseDiffusionModel.

        Args:
            image (torch.Tensor):
                Input image tensor, Bx3xHxW.
            gt_cameras (Optional[CamerasBase], optional):
                Camera object. Defaults to None.
            sequence_name (Optional[List[str]], optional):
                List of sequence names. Defaults to None.
            cond_fn ([type], optional):
                Conditional function. Wrapper for GGS or other functions.
            cond_start_step (int, optional):
                The sampling step to start using conditional function.
            denoise_init

        Returns:
            PerspectiveCameras: PyTorch3D camera object.
        """

        shapelist = list(image.shape)
        batch_num = shapelist[0]
        frame_num = shapelist[1]

        reshaped_image = image.reshape(batch_num * frame_num, *shapelist[2:])
        z = self.image_feature_extractor(reshaped_image).reshape(batch_num, frame_num, -1)
        if training:
            pose_encoding = camera_to_pose_encoding(gt_cameras, pose_encoding_type=self.pose_encoding_type)

            if batch_repeat > 0:
                pose_encoding = pose_encoding.reshape(batch_num * batch_repeat, -1, self.target_dim)
                z = z.repeat(batch_repeat, 1, 1)
            else:
                pose_encoding = pose_encoding.reshape(batch_num, -1, self.target_dim)

            diffusion_results = self.diffuser(pose_encoding, z=z)

            diffusion_results["pred_cameras"] = pose_encoding_to_camera(
                diffusion_results["x_0_pred"], pose_encoding_type=self.pose_encoding_type
            )

            return diffusion_results
        else:
            B, N, _ = z.shape

            target_shape = [B, N, self.target_dim]

            gt_pose = convert_data_to_perspective_camera(data)
            gt_logfl = torch.log(gt_pose.focal_length) - 1.8

            if self.DIFF_CONF.enable:
                pose_encodings = []
                pose_encoding_samples = []
                for i in range(5):
                    pose_encoding, pose_encoding_sample = self.diffuser.sample(
                        shape=target_shape, z=z, cond_fn=cond_fn, init_pose=denoise_init
                    )
                    pose_encodings.append(pose_encoding)
                    pose_encoding_samples.append(pose_encoding_sample)

                print(len(pose_encoding_samples))

                pose_encodings_10 = [p[-12] for p in pose_encoding_samples]
                pose_var = torch.var(torch.stack(pose_encodings, dim=0), dim=0)
                pose_var_10 = torch.var(torch.stack(pose_encodings_10, dim=0), dim=0)
                print(f"mean variance: {pose_var.mean()}")
                print(f"mean variance 10: {pose_var_10.mean()}")
                conf = 1 / pose_var.mean()
                conf_10 = 1 / pose_var_10.mean()
                print(f"conf: {conf}")
                print(f"conf 10: {conf_10}")


                out = {'scene_id': data['scene_id'][0], 'pair_names': '-'.join([n[0].split('_')[1].split('.')[0] for n in data['pair_names']])}

                shape = torch.from_numpy(np.array(image.shape[-2:]))
                shape = torch.stack((shape, shape))

                for i in range(len(pose_encodings)):
                    pred = pose_encoding_to_visdom(pose_encodings[i], pose_encoding_type=self.pose_encoding_type, return_dict=False)
                    pred_R, pred_T, pred_K = opencv_from_visdom_projection(pred, shape)

                    out[f'pose_{i}_R0'] = matrix_to_quaternion(pred_R)[0].cpu().numpy()
                    out[f'pose_{i}_T0'] = pred_T[0].cpu().numpy()
                    out[f'pose_{i}_K0'] = torch.tensor([pred_K[0][0][0], pred_K[0][0][2], pred_K[0][1][1], pred_K[0][1][2]]).cpu().numpy()


                    out[f'pose_{i}_R1'] = matrix_to_quaternion(pred_R)[1].cpu().numpy()
                    out[f'pose_{i}_T1'] = pred_T[1].cpu().numpy()
                    out[f'pose_{i}_K1'] = torch.tensor([pred_K[1][0][0], pred_K[1][0][2], pred_K[1][1][1], pred_K[1][1][2]]).cpu().numpy()

                for i in range(len(pose_encodings_10)):
                    pred = pose_encoding_to_visdom(pose_encodings_10[i], pose_encoding_type=self.pose_encoding_type, return_dict=False)
                    pred_R, pred_T, pred_K = opencv_from_visdom_projection(pred, shape)

                    out[f'pose10_{i}_R0'] = matrix_to_quaternion(pred_R)[0].cpu().numpy()
                    out[f'pose10_{i}_T0'] = pred_T[0].cpu().numpy()
                    out[f'pose10_{i}_K0'] = torch.tensor([pred_K[0][0][0], pred_K[0][0][2], pred_K[0][1][1], pred_K[0][1][2]]).cpu().numpy()

                    out[f'pose10_{i}_R1'] = matrix_to_quaternion(pred_R)[1].cpu().numpy()
                    out[f'pose10_{i}_T1'] = pred_T[1].cpu().numpy()
                    out[f'pose10_{i}_K1'] = torch.tensor([pred_K[1][0][0], pred_K[1][0][2], pred_K[1][1][1], pred_K[1][1][2]]).cpu().numpy()

                pred_R, pred_T, pred_K = opencv_from_visdom_projection(gt_pose, shape)

                out[f'gt_R0'] = matrix_to_quaternion(pred_R)[0].cpu().numpy()
                out[f'gt_T0'] = pred_T[0].cpu().numpy()
                out[f'gt_K0'] = torch.tensor([pred_K[0][0][0], pred_K[0][0][2], pred_K[0][1][1], pred_K[0][1][2]]).cpu().numpy()

                out[f'gt_R1'] = matrix_to_quaternion(pred_R)[1].cpu().numpy()
                out[f'gt_T1'] = pred_T[1].cpu().numpy()
                out[f'gt_K1'] = torch.tensor([pred_K[1][0][0], pred_K[1][0][2], pred_K[1][1][1], pred_K[1][1][2]]).cpu().numpy()

                if pose_solver_output[0]:
                    pred_R, pred_T, pred_K = opencv_from_visdom_projection(pose_solver_output[0], shape)

                    out[f'emat_R0'] = matrix_to_quaternion(pred_R)[0].cpu().numpy()
                    out[f'emat_T0'] = pred_T[0].cpu().numpy()
                    out[f'emat_K0'] = torch.tensor([pred_K[0][0][0], pred_K[0][0][2], pred_K[0][1][1], pred_K[0][1][2]]).cpu().numpy()

                    out[f'emat_R1'] = matrix_to_quaternion(pred_R)[1].cpu().numpy()
                    out[f'emat_T1'] = pred_T[1].cpu().numpy()
                    out[f'emat_K1'] = torch.tensor([pred_K[1][0][0], pred_K[1][0][2], pred_K[1][1][1], pred_K[1][1][2]]).cpu().numpy()

                    out[f"emat_conf"] = pose_solver_conf

                else:
                    out[f'emat_R0'] = np.nan
                    out[f'emat_T0'] = np.nan
                    out[f'emat_K0'] = np.nan

                    out[f'emat_R1'] = np.nan
                    out[f'emat_T1'] = np.nan
                    out[f'emat_K1'] = np.nan

                    out[f"emat_conf"] = np.nan

                write_header = not os.path.isfile(f"./{os.environ['out_root']}/confidence.csv")

                with open(f"./{os.environ['out_root']}/confidence.csv", 'a') as f:
                    w = csv.DictWriter(f, out.keys())
                    if write_header:
                        w.writeheader()
                    w.writerow(out)

                if self.viz is not None:
                    cams_show_1 = {"gt": gt_pose, "emat": pose_solver_output[0]}
                    for i in range(len(pose_encodings)):
                        cams_show_1[f"pose_{i}"] = pose_encoding_to_visdom(pose_encodings[i], pose_encoding_type=self.pose_encoding_type, return_dict=False)
                    fig = plot_scene({f"{conf}": cams_show_1}, axis_args=AxisArgs(showgrid=True))
                    self.viz.plotlyplot(fig, env="main", win=f"{0}")

                    cams_show_2 = {"gt": gt_pose}
                    for i in range(len(pose_encodings)):
                        cams_show_2[f"pose_10_{i}"] = pose_encoding_to_visdom(pose_encodings_10[i], pose_encoding_type=self.pose_encoding_type, return_dict=False)
                    fig = plot_scene({f"{conf_10}": cams_show_2}, axis_args=AxisArgs(showgrid=True))
                    self.viz.plotlyplot(fig, env="main", win=f"{1}")


                return pose_encoding_to_visdom(pose_encodings[0], pose_encoding_type=self.pose_encoding_type, return_dict=False), conf
            elif self.PGS3D.enable:
                attempts_remaining = 3
                best_diffuser_loss = float('inf')
                best_diffuser_start = None
                best_diffuser_pose_process = None
                print(f"PGS3D: attempts remaining: {attempts_remaining}!!!")
                while attempts_remaining >= 0:
                    print(f"PGS3D: attempt {4 - attempts_remaining}!!!")
                    try:
                        (pose_encoding, pose_encoding_diffusion_samples) = self.diffuser.sample(
                            shape=target_shape, z=z, cond_fn=cond_fn, init_pose=denoise_init
                        )
                        if pose_solver_conf is not None:
                            print(f"confidence in pose_solver: {pose_solver_conf}")
                        pred_cams = pose_encoding_to_visdom(pose_encoding, pose_encoding_type=self.pose_encoding_type, return_dict=False)
                        break
                    except HighLossException as e:
                        print(f"PGS3D doesn't agree with diffusion output!!! loss is: {e.loss}")
                        if e.loss < best_diffuser_loss:
                            print("This was the best sample so far though - saving...")
                            best_diffuser_loss = e.loss
                            best_diffuser_start = self.diffuser.start
                            best_diffuser_pose_process = self.diffuser.pose_process
                        if attempts_remaining > 0:
                            print("Going to generate another diffusion sample!!!")
                            attempts_remaining -= 1
                        else:
                            print("Out of attempts, going to use best sample so far!!!")
                            (pose_encoding, pose_encoding_diffusion_samples) = self.diffuser.continue_sample(best_diffuser_pose_process, best_diffuser_start,
                                shape=target_shape, z=z, cond_fn=cond_fn, init_pose=denoise_init
                            )
                            if pose_solver_conf is not None:
                                print(f"confidence in pose_solver: {pose_solver_conf}")
                            pred_cams = pose_encoding_to_visdom(pose_encoding, pose_encoding_type=self.pose_encoding_type, return_dict=False)
                            break
            else:
                (pose_encoding, pose_encoding_diffusion_samples) = self.diffuser.sample(
                    shape=target_shape, z=z, cond_fn=cond_fn, init_pose=denoise_init, gt_fl=gt_logfl
                )
                log_focal_length = pose_encoding[:,:, 7:9]
                if pose_solver_conf is not None:
                    print(f"confidence in pose_solver: {pose_solver_conf}")
                pred_cams = pose_encoding_to_visdom(pose_encoding, pose_encoding_type=self.pose_encoding_type, return_dict=False)


            # pred_cams_process = [pose_encoding_to_visdom(p, pose_encoding_type=self.pose_encoding_type, return_dict=False) for p in pose_encoding_diffusion_samples]

            if self.GT_ALIGN.enable:
                pred_cams_aligned = corresponding_cameras_alignment(
                cameras_src=pred_cams, cameras_tgt=gt_pose, estimate_scale=True, mode="extrinsics", eps=1e-9
                )

                return pred_cams_aligned, 0
            elif self.POSE_ALIGN.enable:
                if pose_solver_output[0]:
                    pred_cams_aligned_p_1 = corresponding_cameras_alignment(
                        cameras_src=pred_cams, cameras_tgt=pose_solver_output[0], estimate_scale=True, mode="extrinsics", eps=1e-9
                    )

                    if self.viz is not None:
                        cams_show = {"gt": gt_pose, "pred": pred_cams, "pose_1": pose_solver_output[0], "a1": pred_cams_aligned_p_1}#, "pose_2": pose_solver_output[1], , "a2": pred_cams_aligned_p_2}
                        fig = plot_scene({f"{self.i}": cams_show}, axis_args=AxisArgs(showgrid=True))
                        self.viz.plotlyplot(fig, env="main", win=f"{self.i}")
                    return pred_cams_aligned_p_1, 0
                else:
                    return pred_cams, 0
            elif self.INIT_POSE.enable:
                if self.viz is not None:
                    d = {}
                    i = 1
                    d["gt"] = gt_pose
                    d["pred"] = pred_cams
                    d["pose_1"] = pose_solver_output[0]
                    # d[0] = pred_cams_process[0]
                    # for p in pred_cams_process[-15:]:
                    #     d[i] = p
                    #     i += 1
                    cams_show = d
                    fig = plot_scene({f"{self.i}": cams_show}, axis_args=AxisArgs(showgrid=True))
                    self.viz.plotlyplot(fig, env="main", win=f"{self.i}")
                return pred_cams, 0
            elif self.PGS.enable:
                if self.viz is not None:
                    d = {}
                    i = 0
                    d["gt"] = gt_pose
                    d["pred"] = pred_cams
                    d["pose_1"] = pose_solver_output[0]
                    for p in pred_cams_process[-15:]:
                        d[i] = p
                        i += 1
                    cams_show = d
                    fig = plot_scene({f"{1}": cams_show}, axis_args=AxisArgs(showgrid=True))
                    self.viz.plotlyplot(fig, env="main", win=f"{1}")
                    # self.i += 1
                return pred_cams, 0
            elif self.PGS3D.enable:
                if self.viz is not None:
                    d = {}
                    i = 0
                    d["gt"] = gt_pose
                    d["pred"] = pred_cams
                    if pose_solver_output[0] is not None:
                        d["pose_solver"] = pose_solver_output[0]
                    for p in pred_cams_process[-15:]:
                        d[i] = p
                        i += 1
                    cams_show = d
                    fig = plot_scene({f"{1}": cams_show}, axis_args=AxisArgs(showgrid=True))
                    self.viz.plotlyplot(fig, env="main", win=f"{1}")
                if pose_solver_conf is not None:
                    if pose_solver_conf <= 10:
                        pose_solver_conf = -1
                    return pred_cams, pose_solver_conf
            else:
                if self.viz is not None:
                    cams_show = {"gt": gt_pose, "pred": pred_cams}
                    fig = plot_scene({f"{self.i}": cams_show}, axis_args=AxisArgs(showgrid=True))
                    self.viz.plotlyplot(fig, env="main", win=f"{self.i}")
                return pred_cams, 0
