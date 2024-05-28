import numpy as np
import cv2 as cv
import open3d as o3d
import torch

# from dependency.dust3r.dust3r.inference import inference
# from dependency.dust3r.dust3r.model import AsymmetricCroCo3DStereo
# from dependency.dust3r.dust3r.utils.image import load_images
# from dependency.dust3r.dust3r.image_pairs import make_pairs
# from dependency.dust3r.dust3r.cloud_opt import global_aligner, GlobalAlignerMode
# from dependency.dust3r.dust3r.utils.geometry import find_reciprocal_matches, xy_grid

from pytorch3d.transforms import quaternion_invert, quaternion_multiply, RotateAxisAngle
from pytorch3d.transforms.rotation_conversions import matrix_to_quaternion, quaternion_to_matrix

def backproject_3d(uv, depth, K):
    '''
    Backprojects 2d points given by uv coordinates into 3D using their depth values and intrinsic K
    :param uv: array [N,2]
    :param depth: array [N]
    :param K: array [3,3]
    :return: xyz: array [N,3]
    '''

    uv1 = np.concatenate([uv, np.ones((uv.shape[0], 1))], axis=1)
    xyz = depth.reshape(-1, 1) * (np.linalg.inv(K) @ uv1.T).T
    return xyz

def backproject_3d_tensor(uv, depth, K):
    '''
    Backprojects 2d points given by uv coordinates into 3D using their depth values and intrinsic K
    :param uv: array [N,2]
    :param depth: array [N]
    :param K: array [3,3]
    :return: xyz: array [N,3]
    '''

    uv1 = torch.concatenate([uv, torch.ones((uv.shape[0], 1), device=uv.device)], axis=1)
    xyz = depth.reshape(-1, 1) * (torch.linalg.inv(K.to(torch.float32)) @ uv1.T).T
    return xyz

# class DUST3RMatching:
#     def __init__(self, cfg):
#         self.batch_size = 1
#         schedule = 'cosine'
#         lr = 0.01
#         niter = 300
#         model_name = "./dependency/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth"
#         self.model = AsymmetricCroCo3DStereo.from_pretrained(model_name)

#     def estimate_pose(self, data):

#         self.device = data['image0'].device
#         self.model = self.model.to(self.device)

#         # r = data['image0']
#         # q = data['image1']

#         scene_root = data["scene_root"][0]
#         paths = [p[0] for p in data["pair_names"]]

#         print(scene_root)
#         print(paths)

#         r = scene_root + '/' + paths[0]
#         q = scene_root + '/' + paths[1]

#         images = load_images([r, q], size=224)
#         pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
#         output = inference(pairs, self.model, self.device, batch_size=self.batch_size)

#         scene = global_aligner(output, device=self.device, mode=GlobalAlignerMode.PairViewer)

#         imgs = scene.imgs
#         focals = scene.get_focals()
#         poses = scene.get_im_poses()
#         pts3d = scene.get_pts3d()
#         confidence_masks = scene.get_masks()


#         pts2d_list, pts3d_list = [], []
#         for i in range(2):
#             conf_i = confidence_masks[i].cpu().numpy()
#             pts2d_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
#             pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])
#         reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(*pts3d_list)
#         print(f'found {num_matches} matches')
#         matches_im1 = pts2d_list[1][reciprocal_in_P2]
#         matches_im0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]

#         print(poses)
#         pose_r_R, pose_r_T = transform_to_extrinsics(poses[0])
#         print(pose_r_R, pose_r_T)
#         pose_q_R, pose_q_T = transform_to_extrinsics(poses[1])
#         print(pose_q_R, pose_q_T)


#         return relative_pose(pose_r_R, pose_q_R, pose_r_T, pose_q_T)

def relative_pose(R0, R1, T0, T1):
    relativeR_quat = quaternion_multiply(matrix_to_quaternion(R1), quaternion_invert(matrix_to_quaternion(R0)))
    relativeR = quaternion_to_matrix(relativeR_quat)
        
    relativeT = T1 - T0

    return relativeR, relativeT

def transform_to_extrinsics(t):
    tr = t[:3, 3].T
    ro = t[:3, :3]
    return ro, tr


class EssentialMatrixSolver:
    '''Obtain relative pose (up to scale) given a set of 2D-2D correspondences'''

    def __init__(self, EMAT):

        # EMat RANSAC parameters
        self.ransac_pix_threshold = EMAT.PIX_THRESHOLD
        self.ransac_confidence = EMAT.CONFIDENCE

    def estimate_pose(self, kpts0, kpts1, data):
        R = np.full((3, 3), np.nan)
        t = np.full((3, 1), np.nan)
        if (kpts0 is None) or (len(kpts0) < 5):
            return R, t, 0

        K0 = data['K_color0'].squeeze(0).cpu().numpy()
        K1 = data['K_color1'].squeeze(0).cpu().numpy()

        # normalize keypoints
        kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
        kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

        # normalize ransac threshold
        ransac_thr = self.ransac_pix_threshold / np.mean([K0[0, 0], K1[1, 1], K0[1, 1], K1[0, 0]])

        # compute pose with OpenCV
        E, mask = cv.findEssentialMat(
            kpts0, kpts1, np.eye(3),
            threshold=ransac_thr, prob=self.ransac_confidence, method=cv.USAC_MAGSAC)
        self.mask = mask
        if E is None:
            return R, t, 0

        # recover pose from E
        best_num_inliers = 0
        ret = R, t, 0
        for _E in np.split(E, len(E) / 3):
            n, R, t, _ = cv.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
            if n > best_num_inliers:
                best_num_inliers = n
                ret = (R, t[:, 0], n)
        return ret


class EssentialMatrixMetricSolverMEAN(EssentialMatrixSolver):
    '''Obtains relative pose with scale using E-Mat decomposition and depth values at inlier correspondences'''

    def __init__(self, EMAT):
        super().__init__(EMAT)

    def estimate_pose(self, kpts0, kpts1, data):
        '''Estimates metric translation vector using by back-projecting E-mat inliers to 3D using depthmaps.
        The metric translation vector can be obtained by looking at the residual vector (projected to the translation vector direction).
        In this version, each 3D-3D correspondence gives an optimal scale for the translation vector. 
        We simply aggregate them by averaging them.
        '''

        # get pose up to scale
        R, t, inliers = super().estimate_pose(kpts0, kpts1, data)
        if inliers == 0:
            return R, t, inliers

        # backproject E-mat inliers at each camera
        K0 = data['K_color0'].squeeze(0)
        K1 = data['K_color1'].squeeze(0)
        mask = self.mask.ravel() == 1        # get E-mat inlier mask from super class
        inliers_kpts0 = np.int32(kpts0[mask])
        inliers_kpts1 = np.int32(kpts1[mask])
        depth_inliers_0 = data['depth0'][0, inliers_kpts0[:, 1], inliers_kpts0[:, 0]].cpu().numpy()
        depth_inliers_1 = data['depth1'][0, inliers_kpts1[:, 1], inliers_kpts1[:, 0]].cpu().numpy()
        # check for valid depth
        valid = (depth_inliers_0 > 0) * (depth_inliers_1 > 0)
        if valid.sum() < 1:
            R = np.full((3, 3), np.nan)
            t = np.full((3, 1), np.nan)
            inliers = 0
            return R, t, inliers
        xyz0 = backproject_3d(inliers_kpts0[valid], depth_inliers_0[valid], K0.cpu())
        xyz1 = backproject_3d(inliers_kpts1[valid], depth_inliers_1[valid], K1.cpu())

        # rotate xyz0 to xyz1 CS (so that axes are parallel)
        xyz0 = (R @ xyz0.T).T

        # get average point for each camera
        pmean0 = np.mean(xyz0, axis=0)
        pmean1 = np.mean(xyz1, axis=0)

        # find scale as the 'length' of the translation vector that minimises the 3D distance between projected points from 0 and the corresponding points in 1
        scale = np.dot(pmean1 - pmean0, t)
        t_metric = scale * t
        t_metric = t_metric.reshape(3, 1)

        return R, t_metric, inliers


class EssentialMatrixMetricSolver(EssentialMatrixSolver):
    '''
        Obtains relative pose with scale using E-Mat decomposition and RANSAC for scale based on depth values at inlier correspondences.
        The scale of the translation vector is obtained using RANSAC over the possible scales recovered from 3D-3D correspondences.
    '''

    def __init__(self, EMAT):
        super().__init__(EMAT)
        self.ransac_scale_threshold = EMAT.SCALE_THRESHOLD

    def estimate_pose(self, kpts0, kpts1, data):
        '''Estimates metric translation vector using by back-projecting E-mat inliers to 3D using depthmaps.
        '''

        # get pose up to scale
        R, t, inliers = super().estimate_pose(kpts0, kpts1, data)
        if inliers == 0:
            return R, t, inliers

        # backproject E-mat inliers at each camera
        K0 = data['K_color0'].squeeze(0)
        K1 = data['K_color1'].squeeze(0)
        mask = self.mask.ravel() == 1        # get E-mat inlier mask from super class
        inliers_kpts0 = np.int32(kpts0[mask])
        inliers_kpts1 = np.int32(kpts1[mask])
        depth_inliers_0 = data['depth0'][0, inliers_kpts0[:, 1], inliers_kpts0[:, 0]].cpu().numpy()
        depth_inliers_1 = data['depth1'][0, inliers_kpts1[:, 1], inliers_kpts1[:, 0]].cpu().numpy()

        # check for valid depth
        valid = (depth_inliers_0 > 0) * (depth_inliers_1 > 0)
        if valid.sum() < 1:
            R = np.full((3, 3), np.nan)
            t = np.full((3, 1), np.nan)
            inliers = 0
            return R, t, inliers
        xyz0 = backproject_3d(inliers_kpts0[valid], depth_inliers_0[valid], K0.cpu())
        xyz1 = backproject_3d(inliers_kpts1[valid], depth_inliers_1[valid], K1.cpu())

        # rotate xyz0 to xyz1 CS (so that axes are parallel)
        xyz0 = (R @ xyz0.T).T

        # get individual scales (for each 3D-3D correspondence)
        scale = np.dot(xyz1 - xyz0, t.reshape(3, 1))  # [N, 1]

        # RANSAC loop
        best_inliers = 0
        best_scale = None
        for scale_hyp in scale:
            inliers_hyp = (np.abs(scale - scale_hyp) < self.ransac_scale_threshold).sum().item()
            if inliers_hyp > best_inliers:
                best_scale = scale_hyp
                best_inliers = inliers_hyp

        # Output results
        t_metric = best_scale * t
        t_metric = t_metric.reshape(3, 1)

        return R, t_metric, best_inliers


class PnPSolver:
    '''Estimate relative pose (metric) using Perspective-n-Point algorithm (2D-3D) correspondences'''

    def __init__(self, cfg):
        # PnP RANSAC parameters
        self.ransac_iterations = cfg.RANSAC_ITER
        self.reprojection_inlier_threshold = cfg.REPROJECTION_INLIER_THRESHOLD
        self.confidence = cfg.CONFIDENCE

    def estimate_pose(self, pts0, pts1, data):
        # uses nearest neighbour
        pts0 = np.int32(pts0)

        if len(pts0) < 4:
            return np.full((3, 3), np.nan), np.full((3, 1), np.nan), 0

        # get depth at correspondence points
        depth_0 = data['depth0'].squeeze(0).cpu().numpy()
        depth_pts0 = depth_0[pts0[:, 1], pts0[:, 0]]

        # remove invalid pts (depth == 0)
        valid = depth_pts0 > depth_0.min()
        if valid.sum() < 4:
            return np.full((3, 3), np.nan), np.full((3, 1), np.nan), 0
        pts0 = pts0[valid]
        pts1 = pts1[valid]
        depth_pts0 = depth_pts0[valid]

        # backproject points to 3D in each sensors' local coordinates
        K0 = data['K_color0'].squeeze(0).cpu().numpy()
        K1 = data['K_color1'].squeeze(0).cpu().numpy()
        xyz_0 = backproject_3d(pts0, depth_pts0, K0)

        # get relative pose using PnP + RANSAC
        succ, rvec, tvec, inliers = cv.solvePnPRansac(
            xyz_0, pts1, K1,
            None, iterationsCount=self.ransac_iterations,
            reprojectionError=self.reprojection_inlier_threshold, confidence=self.confidence,
            flags=cv.SOLVEPNP_P3P)

        # refine with iterative PnP using inliers only
        if succ and len(inliers) >= 6:
            succ, rvec, tvec, _ = cv.solvePnPGeneric(xyz_0[inliers], pts1[inliers], K1,
             None, useExtrinsicGuess=True, rvec=rvec, tvec=tvec, flags=cv.SOLVEPNP_ITERATIVE)
            rvec = rvec[0]
            tvec = tvec[0]

        # avoid degenerate solutions
        if succ:
            if np.linalg.norm(tvec) > 1000:
                succ = False

        if succ:
            R, _ = cv.Rodrigues(rvec)
            t = tvec.reshape(3, 1)
        else:
            R = np.full((3, 3), np.nan)
            t = np.full((3, 1), np.nan)
            inliers = []

        return R, t, len(inliers)


class ProcrustesSolver:
    '''Estimate relative pose (metric) using 3D-3D correspondences'''

    def __init__(self, PROCRUSTES):

        # Procrustes RANSAC parameters
        self.ransac_max_corr_distance = PROCRUSTES.MAX_CORR_DIST
        self.refine = PROCRUSTES.REFINE

    def estimate_pose(self, pts0, pts1, data):
        # uses nearest neighbour
        pts0 = np.int32(pts0)
        pts1 = np.int32(pts1)

        # print(pts0.shape)

        if len(pts0) < 3:
            return np.full((3, 3), np.nan), np.full((3, 1), np.nan), 0

        # get depth at correspondence points
        depth_0, depth_1 = data['depth0'].squeeze(0).cpu(), data['depth1'].squeeze(0).cpu()
        # print(depth_0.shape)
        depth_pts0 = depth_0[pts0[:, 1], pts0[:, 0]]
        depth_pts1 = depth_1[pts1[:, 1], pts1[:, 0]]

        # remove invalid pts (depth == 0)
        valid = (depth_pts0 > depth_0.min()) * (depth_pts1 > depth_1.min())
        if valid.sum() < 3:
            return np.full((3, 3), np.nan), np.full((3, 1), np.nan), 0
        pts0 = pts0[valid]
        pts1 = pts1[valid]
        depth_pts0 = depth_pts0[valid]
        depth_pts1 = depth_pts1[valid]

        # backproject points to 3D in each sensors' local coordinates
        K0 = data['K_color0'].squeeze(0).cpu()
        K1 = data['K_color1'].squeeze(0).cpu()
        xyz_0 = backproject_3d(pts0, depth_pts0, K0)
        xyz_1 = backproject_3d(pts1, depth_pts1, K1)

        # create open3d point cloud objects and correspondences idxs
        pcl_0 = o3d.geometry.PointCloud()
        pcl_0.points = o3d.utility.Vector3dVector(xyz_0)
        pcl_1 = o3d.geometry.PointCloud()
        pcl_1.points = o3d.utility.Vector3dVector(xyz_1)
        corr_idx = np.arange(pts0.shape[0])
        corr_idx = np.tile(corr_idx.reshape(-1, 1), (1, 2))
        corr_idx = o3d.utility.Vector2iVector(corr_idx)

        # obtain relative pose using procrustes
        ransac_criteria = o3d.pipelines.registration.RANSACConvergenceCriteria()
        res = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
            pcl_0, pcl_1, corr_idx, self.ransac_max_corr_distance, criteria=ransac_criteria)
        inliers = int(res.fitness * np.asarray(pcl_1.points).shape[0])

        # refine with ICP
        if self.refine:
            # first, backproject both (whole) point clouds
            vv, uu = np.mgrid[0:depth_0.shape[0], 0:depth_1.shape[1]]
            uv_coords = np.concatenate([uu.reshape(-1, 1), vv.reshape(-1, 1)], axis=1)

            valid = depth_0.reshape(-1) > 0
            xyz_0 = backproject_3d(uv_coords[valid], depth_0.reshape(-1)[valid], K0)

            valid = depth_1.reshape(-1) > 0
            xyz_1 = backproject_3d(uv_coords[valid], depth_1.reshape(-1)[valid], K1)

            pcl_0 = o3d.geometry.PointCloud()
            pcl_0.points = o3d.utility.Vector3dVector(xyz_0)
            pcl_1 = o3d.geometry.PointCloud()
            pcl_1.points = o3d.utility.Vector3dVector(xyz_1)

            icp_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-4,
                                                                             relative_rmse=1e-4,
                                                                             max_iteration=30)

            res = o3d.pipelines.registration.registration_icp(pcl_0,
                                                              pcl_1,
                                                              self.ransac_max_corr_distance,
                                                              init=res.transformation,
                                                              criteria=icp_criteria)

        R = res.transformation[:3, :3]
        t = res.transformation[:3, -1].reshape(3, 1)
        inliers = int(res.fitness * np.asarray(pcl_1.points).shape[0])
        return R, t, inliers
