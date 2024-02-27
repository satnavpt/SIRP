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

# Third-party library imports
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.transforms import se3_exp_map, se3_log_map, Transform3d, so3_relative_angle, quaternion_invert, quaternion_multiply, quaternion_apply
from pytorch3d.vis.plotly_vis import plot_scene
from pytorch3d.utils import opencv_from_cameras_projection, cameras_from_opencv_projection

from ..util.camera_transform import pose_encoding_to_camera, camera_to_pose_encoding, _convert_pixels_to_ndc, transform_to_extrinsics, convert_data_to_perspective_camera, pred_to_opencv
from ..util.normalize_cameras import normalize_cameras
from ..util.match_extraction import extract_match_memory
from ..util.load_img_folder import preprocess_images
from ..util.geometry_guided_sampling import geometry_guided_sampling
from lib.models.matching.feature_matching import *
from lib.models.matching.pose_solver import *

from .. import model
from hydra.utils import instantiate
from pytorch3d.renderer.cameras import PerspectiveCameras

from transforms3d.quaternions import mat2quat
from transforms3d.quaternions import quat2mat

from visdom import Visdom

from pytorch3d.ops import corresponding_cameras_alignment       


logger = logging.getLogger(__name__)

class PoseDiffusionModel(nn.Module):
    def __init__(self, pose_encoding_type: str, IMAGE_FEATURE_EXTRACTOR: Dict, DIFFUSER: Dict, DENOISER: Dict, POSE_SOLVER: str, PROCRUSTES: Dict, MATCHING: Dict = None, GGS: Dict = None):
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

        if POSE_SOLVER == "Procrustes":
            self.pose_solver = ProcrustesSolver(PROCRUSTES)
            self.feature_matching = PrecomputedMatching(MATCHING)
        else:
            self.pose_solver = None
            self.feature_matching = None

        self.GGS = GGS

        self.viz = None#Visdom()
        self.i = 0

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, data):
        # convert_data_to_ndc(data)
        r = data['image0']
        q = data['image1']
        images = torch.cat((r, q), dim=0)
        images, image_info = preprocess_images(images)

        # size = torch.tensor(image_info['size']) * image_info['resized_scales']
        # h = size[0]
        # K = torch.cat((data['K_color0'], data['K_color1'])).to(dtype=torch.float32)
        # fl = K[:, 0, 0]
        # fx = fl[0]
        # print(f"h: {h}")
        # print(f"fx: {fx}")
        # fov_x = torch.rad2deg(2 * torch.arctan(h / (2 * fx)))
        # print(f"fov_x (deg): {fov_x}")

        # exit()

        if self.pose_solver is not None:
            try:
                K_ref = data['K_color0'][0]
                # fl = torch.tensor([K[0][0], K[1][1]])
                # pp = torch.tensor([K[0][2], K[1][2]])
                id = torch.tensor([224, 224])
                # fl, pp = _convert_pixels_to_ndc(fl, pp, id)

                # reference = torch.zeros((9))
                # reference[7:9] = torch.log(torch.tensor(flmm))

                pts1, pts2 = self.feature_matching.get_correspondences(data)
                R_matcher, t_matcher, inliers = self.pose_solver.estimate_pose(pts1, pts2, data)
                R_matcher = torch.from_numpy(R_matcher.copy()).float()
                t_matcher = torch.from_numpy(t_matcher.copy()).float().reshape(-1)
                # R_matcher = mat2quat(R_matcher.numpy())
                K_que = data['K_color1'][0]
                # fl = torch.tensor([K[0][0], K[1][1]])
                # pp = torch.tensor([K[0][2], K[1][2]])
                # fl, pp = _convert_pixels_to_ndc(fl, pp, id)
                # print(fl)
                # print(pp)
                # print(R_matcher)
                # print(t_matcher)
                # print(id)

                Rs = torch.stack((torch.eye(3), R_matcher))
                # print(Rs)

                Ts = torch.stack((torch.zeros(3), t_matcher))
                # print(Ts)

                Ks = torch.stack((K_ref, K_que))
                # print(Ks)

                sizes = torch.stack((id, id))
                # print(sizes)

                init_poses = camera_to_pose_encoding(cameras_from_opencv_projection(Rs.to(K_ref), Ts.to(K_ref), Ks.to(K_ref), sizes.to(K_ref))).unsqueeze(0).float()
                if torch.isnan(init_poses).any():
                    init_poses = None
            except:
                init_poses = None

                # camera = PerspectiveCameras(focal_length=fl, principal_point=pp, R=R_matcher, T=t_matcher, image_size=id)
                # print(camera.get_world_to_view_transform().get_matrix())

                # cameras, s = normalize_cameras(convert_data_to_perspective_camera(data))
                # m = (cameras.get_world_to_view_transform().get_matrix())
                # fl = cameras.focal_length
                # print(fl)
                # r = m[0]
                # q = m[1]
                # print(r)
                # print(q)

                # reference = torch.zeros((9))
                # reference[3:7] = torch.from_numpy(mat2quat(r[:3, :3].numpy()))
                # reference[7:9] = torch.log(torch.tensor(fl[0]))
                # query = torch.zeros((9))
                # query[:3] = q[3, :3]
                # query[3:7] = torch.from_numpy(mat2quat(q[:3, :3].numpy()))
                # query[7:9] = torch.log(torch.tensor(fl[1]))
                
                # query = torch.zeros((9))
                # query[:3] = t_matcher
                # query[3:7] = torch.from_numpy(R_matcher)
                # query[7:9] = torch.log(torch.tensor(flmm))

                # init_poses = torch.stack((reference, query)).unsqueeze(0).to(device=data['K_color0'][0].device)

                # print(f"T procrustes: {t_matcher}")
                # print(f"R procrustes: {R_matcher}")
                # print(f"fl procrustes: {flmm}")
        #         print(f"init: {init_poses}")
        #     except Exception as e:
        #         print(e)
        #         init_poses = None
        else:
            init_poses = None

        if self.GGS.enable:
            # Optional TODO: remove the keypoints outside the cropped region?

            # key points in each image and correspondences
            kp1, kp2, i12 = extract_match_memory(images=images, image_info=image_info)

            if kp1 is not None:
                keys = ["kp1", "kp2", "i12", "img_shape"]
                values = [kp1, kp2, i12, images.shape]
                matches_dict = dict(zip(keys, values))

                # R_matcher = torch.from_numpy(R.copy()).unsqueeze(0).float()
                # t_matcher = torch.from_numpy(t.copy()).view(3).unsqueeze(0).float()
                # print(R_matcher)
                # print(t_matcher)
                # r = 0.9671299251693237 0.1159496343168482 0.2091477721901659 0.0864441989474035 
                # t = -0.8317242955761917 -0.1680682894548832 0.3226619586918621
                # cameras_matcher = PerspectiveCameras(R=R_matcher, T=t_matcher, device=R_matcher.device)
                # pose_matcher = camera_to_pose_encoding(cameras_matcher, pose_encoding_type=self.pose_encoding_type)

                self.GGS.pose_encoding_type = self.pose_encoding_type
                GGS_cfg = OmegaConf.to_container(self.GGS)

                cond_fn = partial(geometry_guided_sampling, matches_dict=matches_dict, GGS_cfg=GGS_cfg)
            else:
                f = open(f"./{os.environ['out_root']}/ggs_results.txt", "a")
                f.write(f"insufficient key points found\n")
                f.close()
                cond_fn = None
        else:
            cond_fn = None

        training = False
        images = images.unsqueeze(0)
        pred_cameras = self._forward(image=images.to(device=r.device), cond_fn=cond_fn, cond_start_step=self.GGS.start_step, training=False, denoise_init=init_poses, data=data)
        # pred_cameras = predictions["pred_cameras"]
        shape = torch.from_numpy(np.array(images.shape[-2:]))
        shape = torch.stack((shape, shape))
        data['inliers'] = 0

        pred_R, pred_T, pred_K = opencv_from_cameras_projection(pred_cameras, shape)
        print("unaligned output converted to opencv")
        print(mat2quat(pred_R[0].cpu().numpy()))
        print(mat2quat(pred_R[1].cpu().numpy()))
        print(pred_T)
        print(pred_K)


        # relativeR = 
        relativeR_quat = quaternion_multiply(torch.from_numpy(mat2quat(pred_R[1].cpu().numpy())), quaternion_invert(torch.from_numpy(mat2quat(pred_R[0].cpu().numpy()))))
        relativeR = torch.from_numpy(quat2mat(relativeR_quat.cpu())).unsqueeze(0)
        
        relativeT = pred_T[1] - pred_T[0]


        # print(relativeT)
        # print(relativeR)
        # npcToCamera(relative)
        # print(relativeR)

        # R_r, R_q = pred_cameras.R[0], pred_cameras.R[1]
        # R = torch.matmul(R_r.T, R_q)

        # T_r, T_q = pred_cameras.T[0], pred_cameras.T[1]
        # T = T_q - T_r

        # print(R_matcher)
        # print(t_matcher)

        # print(data['T_0to1'])

        # print(R)
        # print(T)
        # query_img = data['pair_names'][1][0]
        # estimated_pose = Pose(image_name=query_img,
        #                       q=mat2quat(R.detach().cpu().numpy()).reshape(-1),
        #                       t=T.reshape(-1).detach().cpu().numpy().reshape(-1),
        #                       inliers=0)

        # print(estimated_pose)

        return relativeR, relativeT


    def _forward(
        self,
        image: torch.Tensor,
        gt_cameras: Optional[CamerasBase] = None,
        sequence_name: Optional[List[str]] = None,
        cond_fn=None,
        cond_start_step=0,
        training=True,
        batch_repeat=-1,
        denoise_init=None,
        data=None,
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
            # print(target_shape)

            # sampling
            (pose_encoding, pose_encoding_diffusion_samples) = self.diffuser.sample(
                shape=target_shape, z=z, cond_fn=cond_fn, cond_start_step=cond_start_step, init_pose=denoise_init
            )

            print("raw diffusion output")
            print(pose_encoding)
            # print(mat2quat(gt_pose.R[0].cpu().numpy()))
            # print(mat2quat(gt_pose.R[1].cpu().numpy()))
            # print(gt_pose.T)
            # print(gt_pose.focal_length)

            # print(f"process: {pose_encoding_diffusion_samples}")

            # print(pose_encoding)

            # # print(pose_encoding.shape)
            # absT = pose_encoding[:, :, :3]
            # quaR = pose_encoding[:, :, 3:7]
            # fl = pose_encoding[:, :, 7:]
            # # print(absT.shape)
            # # print(absT)
            # relativeT = absT[:, 1] - absT[:, 0]
            # # print(quaR.shape)
            # # print(quaR)
            # relativeQuaR = quaternion_multiply(quaternion_invert(quaR[:, 0]), quaR[:, 1])

            # print("before cameraing")
            # print(f"translation: {relativeT}")
            # print(f"rotation: {relativeQuaR}")
            # print(f"fl: {fl}")
            # print(relativeT[0], relativeQuaR[0])
            # return relativeT[0], relativeQuaR[0]

            # convert the encoded representation to PyTorch3D cameras
            # pose_encoding[:, 1, :3] *= -1 
            # pose_encoding[:, :, 3] *= -1
            # print(pose_encoding)
            # preds_chain = {}
            # for i, item in enumerate(pose_encoding_diffusion_samples):
            #     pred_c = pose_encoding_to_camera(item, pose_encoding_type=self.pose_encoding_type, return_dict=False)
            #     preds_chain[i] = pred_c
            size = torch.from_numpy(np.array([400,400])).unsqueeze(0)
            print("pred to perspective")
            pred_cams = pose_encoding_to_camera(pose_encoding, pose_encoding_type=self.pose_encoding_type, return_dict=False)
            print(mat2quat(pred_cams.R[0].cpu().numpy()))
            print(mat2quat(pred_cams.R[1].cpu().numpy()))
            print(pred_cams.T)
            print(pred_cams.focal_length)

            # return pred_cams


            # return pred_cams
            # pred_cams2 = pred_to_opencv(pred_cams, torch.cat((size, size)))
            # sizes = torch.cat((size, size))
            # for transform in pred_cams.get_world_to_view_transform().get_matrix():
            #     t, r = transform_to_extrinsics(transform)
            #     print(mat2quat(r.cpu().numpy()))
            #     print(t)
            # R, t, K = opencv_from_cameras_projection(normalize_cameras(pred_cams)[0], sizes)
            # print("rotation matrices")
            # print(R)
            # print("rotation quaternions")
            # for m in R:
            #     print(mat2quat(m.cpu().numpy()))
            # print("translations")
            # print(t)
            # print("intrinsics")
            # print(K)
            # # for r in R:
                # print(mat2quat(r.cpu().numpy()))
                
            # print(R)
            # print(t)
            # print(K)
            # print("pose diff output converted")
            # print(pred_cameras.get_ndc_camera_transform().get_matrix())
            # print(pred_cameras.get_projection_transform().get_matrix())
            # print(pred_cameras.get_world_to_view_transform().get_matrix())
            # print(pred_cameras.get_full_projection_transform().get_matrix())

            # transforms = pred_cameras.get_world_to_view_transform().get_matrix()
            # r = transforms[0]
            # q = transforms[1]

            # R_r = R[0]
            # R_q = R[1]
            # T_r = t[0]
            # T_q = t[1]

            # T_r = r[3, :3]
            # R_r = r[:3, :3]
            # # print(f"ref rot: {mat2quat(R_r.cpu().numpy())}")
            # # print(f"ref tran: {T_r}")

            # T_q = q[3, :3]
            # R_q = q[:3, :3]
            # # print(f"que rot: {mat2quat(R_q.cpu().numpy())}")
            # # print(f"que tran: {T_q}")

            # relativeR = quaternion_multiply(torch.from_numpy(mat2quat(R_q.cpu().numpy())), quaternion_invert(torch.from_numpy(mat2quat(R_r.cpu().numpy()))))
            # relativeT = T_q - quaternion_apply(relativeR, T_r)

            # print("after cameraing")
            # print(f"translation: {relativeT}")
            # print(f"rotation: {relativeR}")
            # print(f"fl: {pred_cameras.focal_length}")

            # exit()
            print("gt to perspective!")
            gt_pose = convert_data_to_perspective_camera(data)
            print(mat2quat(gt_pose.R[0].cpu().numpy()))
            print(mat2quat(gt_pose.R[1].cpu().numpy()))
            print(gt_pose.T)
            print(gt_pose.focal_length)

            # exit()
            # for transform in gt_pose.get_world_to_view_transform().get_matrix():
            #     t, r = transform_to_extrinsics(transform)
            #     print(mat2quat(r.cpu().numpy()))
            #     print(t)
            # print(gt_pose.focal_length)
            # print(gt_pose.principal_point)
            # exit()

            # return pred_cams

            pred_cams_aligned = corresponding_cameras_alignment(
                cameras_src=pred_cams, cameras_tgt=gt_pose, estimate_scale=True, mode="extrinsics", eps=1e-9
            )
            print("aligned")
            print(mat2quat(pred_cams_aligned.R[0].cpu().numpy()))
            print(mat2quat(pred_cams_aligned.R[1].cpu().numpy()))
            print(pred_cams_aligned.T)
            print(pred_cams_aligned.focal_length)


            # print(mat2quat(pred_cams_aligned.R[0].cpu().numpy()))
            # print(mat2quat(pred_cams_aligned.R[1].cpu().numpy()))
            # print(pred_cams_aligned.T)
            # print(pred_cams_aligned.focal_length)

            # gt_pose = normalize_cameras(gt_pose)[0]
            # pred_cams = normalize_cameras(pred_cams)[0]

            # print("\n\nrotation matrices")
            # print(gt_pose.R)
            # print(pred_cams.R)

            # print("\n\nrotation quaternions")
            # for r in (gt_pose.R):
                # print(mat2quat(r.cpu().numpy()))
            # for r in (pred_cams.R):
                # print(mat2quat(r.cpu().numpy()))

            # print("\n\ntranslations")
            # print(gt_pose.T)
            # print(pred_cams.T)
            if self.viz is not None:
                cams_show = {"gt": gt_pose, "pred": pred_cams,  "pred_aligned": pred_cams_aligned} # "pred": pred_cams, , "pred": pred_cams,
                # # preds_chain["gt"] = gt_pose
                fig = plot_scene({f"0": cams_show})
                self.viz.plotlyplot(fig, env="main", win=f"{self.i}")
                # self.i += 1
                # time.sleep(10)
            return pred_cams_aligned


            # return pred_cams

            # return relativeT, relativeR