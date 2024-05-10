# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import tempfile
from pathlib import Path
from omegaconf import OmegaConf

import numpy as np
import pycolmap
from typing import Optional, List, Dict, Any
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
from hloc.utils.io import read_image
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster

from matplotlib import cm
import random
import numpy as np
import pickle
import pycolmap
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from gluefactory.eval.io import parse_config_path, load_model
from gluefactory.utils.export_predictions import export_predictions_from_memory


def extract_match_memory(images = None, image_info = None):
    # Now only supports SPSG

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_mapping = os.path.join(tmpdir, "mapping")
        os.makedirs(tmp_mapping)

        for i, image in enumerate(images):
            save_image(image, os.path.join(tmp_mapping, f"{i}.png"))
        matches, keypoints = run_hloc(tmpdir)

    # print(matches)
    # print(len(list(matches.values())[0]))
    # print(keypoints)
    # print(len(list(keypoints.values())[0]))
    # print(len(list(keypoints.values())[1]))
    # From the format of colmap to PyTorch3D
    kp1, kp2, i12 = colmap_keypoint_to_pytorch3d(matches, keypoints, image_info)

    return kp1, kp2, i12

def extract_match_memory_glue(data=None):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_mapping = os.path.join(tmpdir, "mapping")
        os.makedirs(tmp_mapping)
    matched_points0, matched_points1, matched_lines0, matched_lines1 = run_gluefactory(data, tmpdir)

    return matched_points0, matched_points1, matched_lines0, matched_lines1

def extract_match(image_paths = None, image_folder_path = None, image_info = None):
    # Now only supports SPSG
        
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_mapping = os.path.join(tmpdir, "mapping")
        os.makedirs(tmp_mapping)

        if image_paths is None:
            for filename in os.listdir(image_folder_path):
                if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff")):
                    shutil.copy(os.path.join(image_folder_path, filename), os.path.join(tmp_mapping, filename))
        else:
            for filename in image_paths: 
                if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff")):
                    shutil.copy(filename, os.path.join(tmp_mapping, os.path.basename(filename)))
        matches, keypoints = run_hloc(tmpdir)

    # From the format of colmap to PyTorch3D
    kp1, kp2, i12 = colmap_keypoint_to_pytorch3d(matches, keypoints, image_info)

    return kp1, kp2, i12


def colmap_keypoint_to_pytorch3d(matches, keypoints, image_info):
    kp1, kp2, i12 = [], [], []
    bbox_xyxy, scale = image_info["bboxes_xyxy"], image_info["resized_scales"]

    for idx in keypoints:
        # coordinate change from COLMAP to OpenCV
        cur_keypoint = keypoints[idx] - 0.5

        # go to the coordiante after cropping
        # use idx - 1 here because the COLMAP format starts from 1 instead of 0
        cur_keypoint = cur_keypoint - [bbox_xyxy[idx - 1][0], bbox_xyxy[idx - 1][1]]
        cur_keypoint = cur_keypoint * scale[idx - 1]
        keypoints[idx] = cur_keypoint

    for (r_idx, q_idx), pair_match in matches.items():
        if pair_match is not None:
            kp1.append(keypoints[r_idx][pair_match[:, 0]])
            kp2.append(keypoints[q_idx][pair_match[:, 1]])
            # print(keypoints[r_idx][pair_match[:, 0]])
            # print(keypoints[q_idx][pair_match[:, 1]])

            i12_pair = np.array([[r_idx - 1, q_idx - 1]])
            i12.append(np.repeat(i12_pair, len(pair_match), axis=0))

    if kp1:
        kp1, kp2, i12 = map(np.concatenate, (kp1, kp2, i12), (0, 0, 0))
    else:
        kp1 = kp2 = i12 = None

    return kp1, kp2, i12

# def colmap_keypoint_to_opencv(matches, keypoints, image_info):
#     kp1, kp2, i12 = [], [], []
#     bbox_xyxy, scale = image_info["bboxes_xyxy"], image_info["resized_scales"]

#     for idx in keypoints:
#         # coordinate change from COLMAP to OpenCV
#         cur_keypoint = keypoints[idx] - 0.5

#         # go to the coordiante after cropping
#         # use idx - 1 here because the COLMAP format starts from 1 instead of 0
#         # cur_keypoint = cur_keypoint - [bbox_xyxy[idx - 1][0], bbox_xyxy[idx - 1][1]]
#         # cur_keypoint = cur_keypoint * scale[idx - 1]
#         # keypoints[idx] = cur_keypoint

#     for (r_idx, q_idx), pair_match in matches.items():
#         if pair_match is not None:
#             kp1.append(keypoints[r_idx][pair_match[:, 0]])
#             kp2.append(keypoints[q_idx][pair_match[:, 1]])
#             # print(keypoints[r_idx][pair_match[:, 0]])
#             # print(keypoints[q_idx][pair_match[:, 1]])

#             i12_pair = np.array([[r_idx - 1, q_idx - 1]])
#             i12.append(np.repeat(i12_pair, len(pair_match), axis=0))

#     if kp1:
#         kp1, kp2, i12 = map(np.concatenate, (kp1, kp2, i12), (0, 0, 0))
#     else:
#         kp1 = kp2 = i12 = None

#     return kp1, kp2, i12


def run_hloc(output_dir: str):
    # learned from
    # https://github.com/cvg/Hierarchical-Localization/blob/master/pipeline_SfM.ipynb

    images = Path(output_dir)
    outputs = Path(os.path.join(output_dir, "output"))
    sfm_pairs = outputs / "pairs-sfm.txt"
    sfm_dir = outputs / "sfm"
    features = outputs / "features.h5"
    matches = outputs / "matches.h5"

    feature_conf = extract_features.confs["superpoint_aachen"]  # or superpoint_inloc
    matcher_conf = match_features.confs["superpoint+lightglue"]

    references = [p.relative_to(images).as_posix() for p in (images / "mapping/").iterdir()]

    feature_path = extract_features.main(feature_conf, images, image_list=references, feature_path=features)
    pairs_from_exhaustive.main(sfm_pairs, image_list=references)
    match_path = match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)
    # match_path = match_dense.main(matcher_conf, sfm_pairs, images, features=features, matches=matches)

    matches, keypoints = compute_matches_and_keypoints(
        sfm_dir, images, sfm_pairs, features, matches, image_list=references
    )

    # model = reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, match_path)
    # if model is None:
    # visualization.visualize_sfm_2d(model, images, color_by='visibility', n=5)
    # visualise_sfm_2d(model, images, color_by='visibility', n=5)

    return matches, keypoints

def get_gluestick_conf():
    configs_path = "configs/"
    conf_path = parse_config_path("superpoint+lsd+gluestick", configs_path)
    conf = OmegaConf.load(conf_path)
    return conf

def run_gluefactory(data, output_dir: str):
    outputs = Path(os.path.join(output_dir, "output"))
    pred_file = outputs / "predictions.h5"
    # default_conf = OmegaConf.create(default_conf)
    conf = get_gluestick_conf()
    # conf = OmegaConf.merge(default_conf, selected_conf)
    model = load_model(conf.model, None)
    lines0, lines1, line_matches0, keypoints0, keypoints1, matches0 = export_predictions_from_memory(
        data,
        model,
        pred_file,
        # keys=self.export_keys,
        # optional_keys=self.optional_export_keys,
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


# def visualise_sfm_2d(reconstruction, image_dir, color_by='visibility', selected=[], n=1, seed=0, dpi=75):
#     assert image_dir.exists()
#     if not isinstance(reconstruction, pycolmap.Reconstruction):
#         reconstruction = pycolmap.Reconstruction(reconstruction)

#     if not selected:
#         image_ids = reconstruction.reg_image_ids()
#         selected = random.Random(seed).sample(
#                 image_ids, min(n, len(image_ids)))

#     c = 0
#     for i in selected:
#         image = reconstruction.images[i]
#         keypoints = np.array([p.xy for p in image.points2D])
#         visible = np.array([p.has_point3D() for p in image.points2D])

#         if color_by == 'visibility':
#             color = [(0, 0, 1) if v else (1, 0, 0) for v in visible]
#             text = f'visible: {np.count_nonzero(visible)}/{len(visible)}'
#         elif color_by == 'track_length':
#             tl = np.array([reconstruction.points3D[p.point3D_id].track.length()
#                            if p.has_point3D() else 1 for p in image.points2D])
#             max_, med_ = np.max(tl), np.median(tl[tl > 1])
#             tl = np.log(tl)
#             color = cm.jet(tl / tl.max()).tolist()
#             text = f'max/median track length: {max_}/{med_}'
#         elif color_by == 'depth':
#             p3ids = [p.point3D_id for p in image.points2D if p.has_point3D()]
#             z = np.array([image.transform_to_image(
#                 reconstruction.points3D[j].xyz)[-1] for j in p3ids])
#             z -= z.min()
#             color = cm.jet(z / np.percentile(z, 99.9))
#             text = f'visible: {np.count_nonzero(visible)}/{len(visible)}'
#             keypoints = keypoints[visible]
#         else:
#             raise NotImplementedError(f'Coloring not implemented: {color_by}.')

#         name = image.name
#         plot_images([read_image(image_dir / name)], dpi=dpi)
#         plot_keypoints([keypoints], colors=[color], ps=4)
#         add_text(0, text)
#         add_text(0, name, pos=(0.01, 0.01), fs=5, lcolor=None, va='bottom')

#         # fig = plt.gcf()
#         # print(os.getcwd())
#         plt.savefig(f'fig_{c}.png')
#         c += 1


def compute_matches_and_keypoints(
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
    # learned from
    # https://github.com/cvg/Hierarchical-Localization/blob/master/hloc/reconstruction.py

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


def _blob_to_array_safe(blob, dtype, shape=(-1,)):
    if blob is not None:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return blob
