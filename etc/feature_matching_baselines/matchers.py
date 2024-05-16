import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
from typing import Optional, List, Dict, Any
import os
import cv2
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

from LoFTR.src.loftr import LoFTR, default_cfg

from SuperGlue.models.utils import read_image
from SuperGlue.models.matching import Matching

import pycolmap
from hloc import extract_features, logger, match_features, match_dense, pairs_from_exhaustive, visualization, reconstruction
from hloc.triangulation import (
    import_features,
    import_matches,
    estimation_and_geometric_verification,
    parse_option_args,
    OutputCapture,
)
from hloc.utils.database import COLMAPDatabase, image_ids_to_pair_id, pair_id_to_image_ids
from hloc.reconstruction import create_empty_db, import_images, get_image_ids
from hloc.utils.viz import (
        plot_images, plot_keypoints, plot_matches, cm_RdGn, add_text)
# from hloc.utils.io import read_image
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster
from torchvision.utils import save_image

from gluefactory.eval.io import parse_config_path, load_model
from gluefactory.utils.export_predictions import export_predictions_from_memory

from lightglue import viz2d
from gluestick.drawing import plot_color_line_matches

torch.set_grad_enabled(False)

# class ImageDataset(torch.utils.data.Dataset):
#     def __init__(self, images, scales, names=['ref.png', 'query.png']):
#         self.images = images
#         self.scales = scales
#         self.names = names

#     def __getitem__(self, idx):
#         image = self.images[idx]
#         orig_size = image.shape[-2:][::-1] / np.array(self.scales[idx])

#         data = {
#             "image": image,
#             "original_size": torch.from_numpy(np.array(orig_size)).to(torch.int64),
#         }
#         return data

#     def __len__(self):
#         return len(self.images)

class LoFTR_matcher:
    def __init__(self, resize, outdoor=False):
        # Initialize LoFTR
        print("started loading model")
        matcher = LoFTR(config=default_cfg)
        weights_path = "LoFTR/weights/outdoor_ot.ckpt" if outdoor else "LoFTR/weights/indoor_ot.ckpt"
        matcher.load_state_dict(torch.load(weights_path)['state_dict'], strict=False)
        matcher = matcher.eval().cuda()
        self.matcher = matcher
        print("model loaded")
        self.resize = resize

    def match(self, pair_path):
        '''retrurn correspondences between images (w/ path pair_path)'''

        input_path0, input_path1 = pair_path
        resize = self.resize
        resize_float = True
        rot0, rot1 = 0, 0
        device = 'cuda'

        # using resolution [640, 480] (default for 7Scenes, re-scale Scannet)
        image0, inp0, scales0 = read_image(
            input_path0, device, resize, rot0, resize_float)

        image1, inp1, scales1 = read_image(
            input_path1, device, resize, rot1, resize_float)

        # LoFTR needs resolution multiple of 8. If that is not the case, we pad 0's to get to a multiple of 8
        if inp0.size(2) % 8 != 0 or inp0.size(1) % 8 != 0:
            pad_bottom = inp0.size(2) % 8
            pad_right = inp0.size(3) % 8
            pad_fn = torch.nn.ConstantPad2d((0, pad_right, 0, pad_bottom), 0)
            inp0 = pad_fn(inp0)
            inp1 = pad_fn(inp1)

        with torch.no_grad():
            batch = {'image0': inp0, 'image1': inp1}
            self.matcher(batch)
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()

        if mkpts0.shape[0] > 0:
            pts = np.concatenate([mkpts0, mkpts1], axis=1)
            return pts
        else:
            print("no correspondences")
            return np.full((1, 4), np.nan)


class SuperGlue_matcher:
    def __init__(self, resize, outdoor=True):
        # copied default values
        nms_radius = 4
        keypoint_threshold = 0.005
        max_keypoints = 1024

        superglue_weights = 'outdoor' if outdoor else 'indoor'  # indoor trained on scannet
        sinkhorn_iterations = 20
        match_threshold = 0.2

        # Load the SuperPoint and SuperGlue models.
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Running inference on device \"{}\"'.format(device))
        config = {
            'superpoint': {
                'nms_radius': nms_radius,
                'keypoint_threshold': keypoint_threshold,
                'max_keypoints': max_keypoints
            },
            'superglue': {
                'weights': superglue_weights,
                'sinkhorn_iterations': sinkhorn_iterations,
                'match_threshold': match_threshold,
            }
        }
        self.matching = Matching(config).eval().to(device)
        self.device = device
        print('SuperGlue model loaded')
        self.resize = resize

    def match(self, pair_path):
        '''retrurn correspondences between images (w/ path pair_path)'''

        input_path0, input_path1 = pair_path
        resize = self.resize
        resize_float = True
        rot0, rot1 = 0, 0

        image0, inp0, scales0 = read_image(
            input_path0, self.device, resize, rot0, resize_float)
        image1, inp1, scales1 = read_image(
            input_path1, self.device, resize, rot1, resize_float)
        pred = self.matching({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']

        # Keep the matching keypoints.
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]

        if mkpts0.shape[0] > 0:
            pts = np.concatenate([mkpts0, mkpts1], axis=1)



            # axes = viz2d.plot_images([image0, image1])
            # viz2d.plot_matches(mkpts0[:20], mkpts1[:20], color="lime", lw=0.2)
            # plt.savefig('matched sg.png')
            # viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)

            # kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
            # viz2d.plot_images([image0, image1])
            # viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)


            return pts
        else:
            print("no correspondences")
            return np.full((1, 4), np.nan)


class SIFT_matcher:
    def __init__(self, resize, outdoor=False):
        self.resize = resize

    def root_sift(self, descs):
        '''Apply the Hellinger kernel by first L1-normalizing, taking the square-root, and then L2-normalizing'''

        eps = 1e-7
        descs /= (descs.sum(axis=1, keepdims=True) + eps)
        descs = np.sqrt(descs)
        return descs

    def match(self, pair_path):
        '''
        Given path to im1, im2, extract correspondences using OpenCV SIFT.
        Returns: pts (N x 4) array containing (x1, y1, x2, y2) correspondences; returns nan array if no correspondences.
        '''

        im1_path, im2_path = pair_path

        # hyper-parameters
        ratio_test_threshold = 0.8
        n_features = 2048
        sift = cv2.SIFT_create(n_features)

        # Read images in grayscale
        img0 = cv2.imread(im1_path, 0)
        img1 = cv2.imread(im2_path, 0)

        # Resize
        img0 = cv2.resize(img0, self.resize)
        img1 = cv2.resize(img1, self.resize)

        # get SIFT key points and descriptors
        kp0, des0 = sift.detectAndCompute(img0, None)
        kp1, des1 = sift.detectAndCompute(img1, None)

        # Apply normalisation (rootSIFT)
        des0, des1 = self.root_sift(des0), self.root_sift(des1)

        # Get matches using FLANN
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des0, des1, k=2)

        pts1 = []
        pts2 = []
        good_matches = []
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < ratio_test_threshold * n.distance:
                pts2.append(kp1[m.trainIdx].pt)
                pts1.append(kp0[m.queryIdx].pt)
                good_matches.append(m)

        pts1 = np.float32(pts1).reshape(-1, 2)
        pts2 = np.float32(pts2).reshape(-1, 2)

        if pts1.shape[0] > 0:
            pts = np.concatenate([pts1, pts2], axis=1)
            return pts
        else:
            print("no correspondences")
            return np.full((1, 4), np.nan)


class HLOC_matcher:
    def __init__(self, feature_conf="superpoint_aachen", matcher_conf="superglue", resize=(540, 720), outdoor=False):
        device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        print(feature_conf)
        print(matcher_conf)

        self.feature_conf = extract_features.confs[feature_conf]
        self.matcher_conf = match_features.confs[matcher_conf]

        self.feature_conf["preprocessing"]['device'] = device
        self.feature_conf["preprocessing"]['resize'] = resize
        self.feature_conf["preprocessing"]['rotation'] = 0
        self.feature_conf["preprocessing"]['resize_float'] = True

        self.resize = resize

    def match(self, pair_path):
        tmpdir = tempfile.TemporaryDirectory()
        tmp_mapping = os.path.join(tmpdir.name, "mapping")
        os.makedirs(tmp_mapping)
        shutil.copyfile(pair_path[0], os.path.join(tmp_mapping, f"ref.png"))
        shutil.copyfile(pair_path[1], os.path.join(tmp_mapping, f"query.png"))

        matches, keypoints = self.run_hloc(tmpdir.name)

        kp2, kp1, _ = self.colmap_keypoint_to_pytorch3d(matches, keypoints)

        if kp1 is not None:
            if kp1.shape[0] > 0:
                pts = np.concatenate([kp1, kp2], axis=1)

                # input_path0, input_path1 = pair_path
                # resize_float = True
                # rot0, rot1 = 0, 0

                # image0, inp0, scales0 = read_image(
                #     input_path0, self.device, self.resize, rot0, resize_float)
                # image1, inp1, scales1 = read_image(
                #     input_path1, self.device, self.resize, rot1, resize_float)

                # axes = viz2d.plot_images([image0, image1])
                # viz2d.plot_matches(kp1[:20], kp2[:20], color="lime", lw=0.2)
                # plt.savefig('matched hloc max.png')
                # exit()

                return pts
            else:
                print("no correspondences")
                return np.full((1, 4), np.nan)
        else:
            print("no correspondences")
            return np.full((1, 4), np.nan)

    def run_hloc(self, output_dir: str):
        images = Path(output_dir)
        outputs = Path(os.path.join(output_dir, "output"))
        sfm_pairs = outputs / "pairs-sfm.txt"
        sfm_dir = outputs / "sfm"
        features = outputs / "features.h5"
        matches = outputs / "matches.h5"

        references = [p.relative_to(images).as_posix() for p in (images / "mapping/").iterdir()]

        feature_path = extract_features.main(self.feature_conf, images, image_list=references, feature_path=features)
        pairs_from_exhaustive.main(sfm_pairs, image_list=references)
        match_path = match_features.main(self.matcher_conf, sfm_pairs, features=features, matches=matches)

        matches, keypoints = self.compute_matches_and_keypoints(
            sfm_dir, images, sfm_pairs, features, matches, image_list=references
        )

        return matches, keypoints

    def compute_matches_and_keypoints(
        self,
        sfm_dir: Path,
        image_dir: Path,
        pairs: Path,
        features: Path,
        matches: Path,
        camera_mode: pycolmap.CameraMode = pycolmap.CameraMode.AUTO,
        verbose: bool = False,
        min_match_score: Optional[float] = None,
        image_list: Optional[List[str]] = None,
        image_options: Optional[Dict[str, Any]] = None,
    ) -> pycolmap.Reconstruction:

        sfm_dir.mkdir(parents=True, exist_ok=True)
        database = sfm_dir / "database.db"

        create_empty_db(database)
        import_images(image_dir, database, camera_mode, image_list, image_options)
        image_ids = get_image_ids(database)
        import_features(image_ids, database, features)
        import_matches(image_ids, database, pairs, matches, min_match_score)
        estimation_and_geometric_verification(database, pairs, verbose)
        db = COLMAPDatabase.connect(database)

        matches = dict(
            (pair_id_to_image_ids(pair_id), _blob_to_array_safe(data, np.uint32, (-1, 2)))
            for pair_id, data in db.execute("SELECT pair_id, data FROM matches")
        )

        keypoints = dict(
            (image_id, _blob_to_array_safe(data, np.float32, (-1, 2)))
            for image_id, data in db.execute("SELECT image_id, data FROM keypoints")
        )

        db.close()

        return matches, keypoints

    def colmap_keypoint_to_pytorch3d(self, matches, keypoints):
        kp1, kp2, i12 = [], [], []

        for idx in keypoints:
            cur_keypoint = keypoints[idx] - 0.5

            keypoints[idx] = cur_keypoint

        for (r_idx, q_idx), pair_match in matches.items():
            if pair_match is not None:
                kp1.append(keypoints[r_idx][pair_match[:, 0]])
                kp2.append(keypoints[q_idx][pair_match[:, 1]])

                i12_pair = np.array([[r_idx - 1, q_idx - 1]])
                i12.append(np.repeat(i12_pair, len(pair_match), axis=0))

        if kp1:
            kp1, kp2, i12 = map(np.concatenate, (kp1, kp2, i12), (0, 0, 0))
        else:
            kp1 = kp2 = i12 = None

        return kp1, kp2, i12

def _blob_to_array_safe(blob, dtype, shape=(-1,)):
    if blob is not None:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return blob

class GLUE_matcher:
    def __init__(self, resize, outdoor=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.resize = resize

    def match(self, pair_path):
        tmpdir = tempfile.TemporaryDirectory()
        tmp_mapping = os.path.join(tmpdir.name, "mapping")
        os.makedirs(tmp_mapping)

        input_path0, input_path1 = pair_path
        resize = self.resize
        resize_float = True
        rot0, rot1 = 0, 0

        image0, inp0, scales0 = read_image(
            input_path0, self.device, resize, rot0, resize_float)
        image1, inp1, scales1 = read_image(
            input_path1, self.device, resize, rot1, resize_float)

        matcher_data = {
            "name": ["ref_query"],
            "view0": {
                "name": "ref",
                "image": inp0,
                "scales": torch.from_numpy(np.array(scales0))
            },
            "view1": {
                "name":"query",
                "image": inp1,
                "scales": torch.from_numpy(np.array(scales1))
            }
        }

        kp1, kp2, li1, li2 = [np.array(i) for i in self.run_gluefactory(matcher_data, tmpdir.name)]

        if li1 is not None:
            if li1.shape[0] > 0:
                lines = np.concatenate([li1, li2], axis=2)
            else:
                print("no lines")
                lines = np.full((1, 8), np.nan)
        else:
            print("no lines")
            lines = np.full((1, 8), np.nan)

        if kp1 is not None:
            if kp1.shape[0] > 0:
                pts = np.concatenate([kp1, kp2], axis=1)
            else:
                print("no correspondences")
                pts = np.full((1, 4), np.nan)
        else:
            print("no correspondences")
            pts = np.full((1, 4), np.nan)

        del kp1
        del kp2
        del li1
        del li2
        
        return pts, lines

        # axes = viz2d.plot_images([image0, image1])
        # viz2d.plot_matches(kp1, kp2, color="lime", lw=0.2)
        # plot_color_line_matches([li1, li2], lw=1)
        # plt.savefig('matched glue.png')

    def get_gluestick_conf(self, cfg="superpoint+lsd+gluestick"):
        configs_path = "configs/"
        conf_path = parse_config_path(cfg, configs_path)
        conf = OmegaConf.load(conf_path)
        return conf

    def run_gluefactory(self, data, output_dir: str):
        outputs = Path(os.path.join(output_dir, "output"))
        pred_file = outputs / "predictions.h5"
        conf = self.get_gluestick_conf()
        model = load_model(conf.model, None)
        lines0, lines1, line_matches0, keypoints0, keypoints1, matches0 = export_predictions_from_memory(
            data,
            model,
            pred_file,
        )

        matched_lines0 = []
        matched_lines1 = []

        matched_points0 = []
        matched_points1 = []

        for i, line in enumerate(lines0):
            if line_matches0[i] > -1:
                matched_lines0.append(line)
                matched_lines1.append(lines1[line_matches0[i]])

        for i, point in enumerate(keypoints0):
            if matches0[i] > -1:
                matched_points0.append(point)
                matched_points1.append(keypoints1[matches0[i]])

        return matched_points0, matched_points1, matched_lines0, matched_lines1