import torch

from lib.models.matching.feature_matching import *
from lib.models.matching.pose_solver import *


class FeatureMatchingModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        if cfg.FEATURE_MATCHING == 'SIFT':
            self.feature_matching = SIFTMatching(cfg)
        elif cfg.FEATURE_MATCHING == 'Precomputed':
            self.feature_matching = PrecomputedMatching(cfg)
        elif cfg.FEATURE_MATCHING == 'PrecomputedGLUE':
            self.feature_matching = PrecomputedGLUEMatching(cfg)
        elif cfg.FEATURE_MATCHING == "HLOC":
            self.feature_matching = HLOCMatching(cfg)
        elif cfg.FEATURE_MATCHING == "MultiplePrecomputed":
            self.feature_matching = MultiplePrecomputedMatching(cfg)
        elif cfg.FEATURE_MATCHING == "GLUE":
            self.feature_matching = GLUEMatching(cfg)
        # elif cfg.FEATURE_MATCHING == "DUST3R":
        #     self.feature_matching = None
        else:
            raise NotImplementedError('Invalid feature matching')

        if cfg.POSE_SOLVER == 'EssentialMatrix':
            self.pose_solver = EssentialMatrixSolver(cfg)
        elif cfg.POSE_SOLVER == 'EssentialMatrixMetric':
            self.pose_solver = EssentialMatrixMetricSolver(cfg.EMAT_RANSAC)
        elif cfg.POSE_SOLVER == 'EssentialMatrixMetricMean':
            self.pose_solver = EssentialMatrixMetricSolverMEAN(cfg.EMAT_RANSAC)
        elif cfg.POSE_SOLVER == 'Procrustes':
            self.pose_solver = ProcrustesSolver(cfg.PROCRUSTES)
        elif cfg.POSE_SOLVER == 'PNP':
            self.pose_solver = PnPSolver(cfg.PNP)
        # elif cfg.FEATURE_MATCHING == "DUST3R":
        #     self.pose_solver = DUST3RMatching(cfg)
        else:
            raise NotImplementedError('Invalid pose solver')

    def forward(self, data):
        assert data['depth0'].shape[0] == 1, 'Baseline models require batch size of 1'

        res = self.feature_matching.get_correspondences(data)
        # get 2D-2D correspondences
        if len(res) == 4:
            pts1, pts2, lin1, lin2 = res
            data['inliers'] = (len(pts1) // 2) + (len(lin1) // 2)
            return None, None
        elif len(res) == 2:
            pts1, pts2 = res

            # get relative pose
            R, t, inliers = self.pose_solver.estimate_pose(pts1, pts2, data)
            data['inliers'] = inliers
            R = torch.from_numpy(R.copy()).unsqueeze(0).float()
            t = torch.from_numpy(t.copy()).view(1, 3).unsqueeze(0).float()
            return R, t
        else:
            poses = self.pose_solver.estimate_pose(data)
            return poses