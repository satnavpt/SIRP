import argparse
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile
from io import TextIOWrapper
import json
import logging

import numpy as np

from benchmark.utils import load_poses, subsample_poses, load_K, precision_recall
from benchmark.metrics import MetricManager, Inputs
import benchmark.config as config
from config.default import cfg

from benchmark.utils import VARIANTS_ANGLE_SIN, quat_angle_error

def compute_difficulty(q_gt, t_gt, variant=VARIANTS_ANGLE_SIN):
    q0 = np.array([ 1., 0., 0., 0.])
    t0 = np.array([ 0., 0., 0.])
    
    rotation_difficulty = quat_angle_error(label=q_gt, pred=q0, variant=variant)[0, 0]
    translation_difficulty = np.linalg.norm(t0 - t_gt)
    
    return (rotation_difficulty, translation_difficulty)

def compute_scene_metrics(dataset_path: Path, submission_zip_gs: ZipFile, submission_zip_nogs: ZipFile, scene: str, rot_diff_cutoff=0, trans_diff_cutoff=0, conf_cutoff_est=0, conf_cutoff_gs=0):
    metric_manager = MetricManager()

    # load intrinsics and poses
    try:
        K, W, H = load_K(dataset_path / scene / 'intrinsics.txt')
        with (dataset_path / scene / 'poses.txt').open('r', encoding='utf-8') as gt_poses_file:
            gt_poses = load_poses(gt_poses_file, load_confidence=False)

    except FileNotFoundError as e:
        logging.error(f'Could not find ground-truth dataset files: {e}')
        raise
    else:
        logging.info(
            f'Loaded ground-truth intrinsics and poses for scene {scene}')

    # try to load estimated poses from submission
    try:
        with submission_zip_gs.open(f'pose_{scene}.txt') as estimated_poses_file:
            estimated_poses_file_wrapper = TextIOWrapper(
                estimated_poses_file, encoding='utf-8')
            estimated_poses_gs = load_poses(
                estimated_poses_file_wrapper, load_confidence=True)
        if submission_zip_nogs is not None:
            with submission_zip_nogs.open(f'pose_{scene}.txt') as estimated_poses_file:
                estimated_poses_file_wrapper = TextIOWrapper(
                    estimated_poses_file, encoding='utf-8')
                estimated_poses_nogs = load_poses(
                    estimated_poses_file_wrapper, load_confidence=True)
        else:
            estimated_poses_nogs = {}
    except KeyError as e:
        logging.warning(
            f'Submission does not have estimates for scene {scene}.')
        return dict(), len(gt_poses)
    except UnicodeDecodeError as e:
        logging.error('Unsupported file encoding: please use UTF-8')
        raise
    else:
        logging.info(f'Loaded estimated poses for scene {scene}')

    # The val/test set is subsampled by a factor of 5
    gt_poses = subsample_poses(gt_poses, subsample=5)

    # failures encode how many frames did not have an estimate
    # e.g. user/method did not provide an estimate for that frame
    # it's different from when an estimate is provided with low confidence!
    failures = 0

    # excluded frames are not considered as they do not meet difficulty requirements
    excluded_frames = 0

    # Results encoded as dict
    # key: metric name; value: list of values (one per frame).
    # e.g. results['t_err'] = [1.2, 0.3, 0.5, ...]
    results = defaultdict(list)

    # compute metrics per frame
    for frame_num, (q_gt, t_gt, _) in gt_poses.items():
        rdiff, tdiff = compute_difficulty(q_gt, t_gt)
        if (rdiff < rot_diff_cutoff):# or (rdiff > rot_diff_cutoff + 10):
            # not in range
            excluded_frames += 1
            continue
        if (tdiff < trans_diff_cutoff):
            # not in range
            excluded_frames += 1
            continue

        failed = False
        if frame_num not in estimated_poses_gs and frame_num not in estimated_poses_nogs:
            # no estimate for this frame - failure!
            failures += 1
            continue
        # estimate from at least one of the methods
        elif frame_num not in estimated_poses_gs:
            estimated_poses_gs[frame_num] = estimated_poses_nogs[frame_num]
        elif (estimated_poses_nogs != {}) and (frame_num not in estimated_poses_nogs):
            estimated_poses_nogs[frame_num] = estimated_poses_gs[frame_num]

        q_est, t_est, confidence = estimated_poses_gs[frame_num]
        if estimated_poses_nogs != {}:
            q_est_nogs, t_est_nogs, confidence_nogs = estimated_poses_nogs[frame_num]

        if confidence < conf_cutoff_est:
            # no estimate for this frame - failure!
            failures += 1
            continue

        if confidence < conf_cutoff_gs and estimated_poses_nogs != {}:
            # confidence is low, use non-guidance method
            inputs = Inputs(q_gt=q_gt, t_gt=t_gt, q_est=q_est_nogs, t_est=t_est_nogs,
                confidence=confidence, K=K[frame_num], W=W, H=H)
        else:
            inputs = Inputs(q_gt=q_gt, t_gt=t_gt, q_est=q_est, t_est=t_est,
                confidence=confidence, K=K[frame_num], W=W, H=H)

        metric_manager(inputs, results)

    return results, failures

def argmedian(x):
  return np.argpartition(x, len(x) // 2)[len(x) // 2]

def aggregate_results(all_results, all_failures):
    # aggregate metrics
    median_metrics = defaultdict(list)
    all_metrics = defaultdict(list)
    for scene_results in all_results.values():
        for metric, values in scene_results.items():
            median_metrics[metric].append(np.median(values))
            all_metrics[metric].extend(values)
    all_metrics = {k: np.array(v) for k, v in all_metrics.items()}
    assert all([v.ndim == 1 for v in all_metrics.values()]
               ), 'invalid metrics shape'

    # compute avg median metrics
    avg_median_metrics = {metric: np.mean(
        values) for metric, values in median_metrics.items()}

    # compute precision/AUC for pose error and reprojection errors
    accepted_poses = (all_metrics['trans_err'] < config.t_threshold) * \
        (all_metrics['rot_err'] < config.R_threshold)
    accepted_vcre = all_metrics['reproj_err'] < config.vcre_threshold
    total_samples = len(next(iter(all_metrics.values()))) + all_failures

    prec_pose = np.sum(accepted_poses) / total_samples
    prec_vcre = np.sum(accepted_vcre) / total_samples

    # compute AUC for pose and VCRE
    _, _, auc_pose = precision_recall(
        inliers=all_metrics['confidence'], tp=accepted_poses, failures=all_failures)
    _, _, auc_vcre = precision_recall(
        inliers=all_metrics['confidence'], tp=accepted_vcre, failures=all_failures)

    # output metrics
    output_metrics = dict()
    output_metrics['Average Median Translation Error'] = avg_median_metrics['trans_err']
    output_metrics['Average Median Rotation Error'] = avg_median_metrics['rot_err']
    output_metrics['Average Median Reprojection Error'] = avg_median_metrics['reproj_err']
    output_metrics[f'Precision @ Pose Error < ({config.t_threshold*100}cm, {config.R_threshold}deg)'] = prec_pose
    output_metrics[f'AUC @ Pose Error < ({config.t_threshold*100}cm, {config.R_threshold}deg)'] = auc_pose
    output_metrics[f'Precision @ VCRE < {config.vcre_threshold}px'] = prec_vcre
    output_metrics[f'AUC @ VCRE < {config.vcre_threshold}px'] = auc_vcre
    output_metrics[f'Estimates for % of frames'] = len(all_metrics['trans_err']) / total_samples
    return output_metrics

# def aggregate_results(all_results_gs, all_results_nogs, all_failures):
#     # aggregate metrics

#     median_metrics = defaultdict(list)
#     all_metrics = defaultdict(list)
#     # count_successes = 0
#     # not_attempted = 0
#     for (scene_gs, scene_results_gs), (scene_nogs, scene_results_nogs) in zip(all_results_gs.items(), all_results_nogs.items()):
#         # print(scene_gs)
#         # print(scene_nogs)
#         # rot_mask = np.array(scene_results_gs["difficulty_rot"]) > rot_diff_cutoff
#         # trans_mask = np.array(scene_results_gs["difficulty_trans"]) > trans_diff_cutoff
#         conf_noest_mask = np.array(scene_results_gs["confidence"]) > conf_cutoff_est
#         conf_nogs_mask = np.array(scene_results_gs["confidence"]) > conf_cutoff_gs
#         # print(scene_results_nogs["confidence"])
#         # mask_gs_conf = rot_mask & trans_mask & conf_noest_mask
#         mask_gs = conf_noest_mask & conf_nogs_mask #rot_mask & trans_mask & 
#         mask_nogs = conf_noest_mask & np.logical_not(conf_nogs_mask) #rot_mask & trans_mask & 
#         # count_successes += np.sum(rot_mask & trans_mask & conf_noest_mask)
#         # not_attempted += np.sum(np.logical_not(rot_mask & trans_mask & conf_noest_mask))
#         for (metric_gs, values_gs), (metric_nogs, values_nogs) in zip(scene_results_gs.items(), scene_results_nogs.items()):
#             values_gs = np.array(values_gs)[mask_gs]
#             values_nogs = np.array(values_nogs)[mask_nogs]
#             values = np.concatenate([values_gs, values_nogs])
#             if len(values) > 0:
#                 median_metrics[metric_gs].append(np.median(values))
#                 all_metrics[metric_gs].extend(values)
#     all_metrics = {k: np.array(v) for k, v in all_metrics.items()}
#     assert all([v.ndim == 1 for v in all_metrics.values()]
#                ), 'invalid metrics shape'

#     # compute avg median metrics
#     avg_median_metrics = {metric: np.mean(
#         values) for metric, values in median_metrics.items()}

#     # compute precision/AUC for pose error and reprojection errors
#     accepted_poses = (all_metrics['trans_err'] < config.t_threshold) * \
#         (all_metrics['rot_err'] < config.R_threshold)
#     accepted_vcre = all_metrics['reproj_err'] < config.vcre_threshold
#     total_samples = len(next(iter(all_metrics.values()))) + all_failures

#     prec_pose = np.sum(accepted_poses) / total_samples
#     prec_vcre = np.sum(accepted_vcre) / total_samples

#     # compute AUC for pose and VCRE
#     _, _, auc_pose = precision_recall(
#         inliers=all_metrics['confidence'], tp=accepted_poses, failures=all_failures)
#     _, _, auc_vcre = precision_recall(
#         inliers=all_metrics['confidence'], tp=accepted_vcre, failures=all_failures)

#     if all_failures > 0:
#         logging.warning(
#             f'Submission is missing pose estimates for {all_failures} frames')
#         logging.warning(
#             f'Excludes {not_attempted} frames')

#     # output metrics
#     output_metrics = dict()
#     output_metrics['Average Median Translation Error'] = avg_median_metrics['trans_err']
#     output_metrics['Average Median Rotation Error'] = avg_median_metrics['rot_err']
#     output_metrics['Average Median Reprojection Error'] = avg_median_metrics['reproj_err']
#     output_metrics[f'Precision @ Pose Error < ({config.t_threshold*100}cm, {config.R_threshold}deg)'] = prec_pose
#     output_metrics[f'AUC @ Pose Error < ({config.t_threshold*100}cm, {config.R_threshold}deg)'] = auc_pose
#     output_metrics[f'Precision @ VCRE < {config.vcre_threshold}px'] = prec_vcre
#     output_metrics[f'AUC @ VCRE < {config.vcre_threshold}px'] = auc_vcre
#     output_metrics[f'Estimates for % of frames'] = len(all_metrics['trans_err']) / total_samples
#     return output_metrics

def count_unexpected_scenes(scenes: tuple, submission_zip: ZipFile):
    submission_scenes = [fname[5:-4]
                         for fname in submission_zip.namelist() if fname.startswith("pose_")]
    return len(set(submission_scenes) - set(scenes))


def main(args):
    dataset_path = args.dataset_path / args.split
    scenes = tuple(f.name for f in dataset_path.iterdir() if f.is_dir())

    submission_zip_gs = ZipFile(args.submission_path_gs, 'r')
    if args.nogs is not None:
        submission_zip_nogs = ZipFile(args.nogs, 'r')
    else:
        submission_zip_nogs = None

    rot_diff_cutoff = int(args.rotcutoff)
    trans_diff_cutoff = int(args.transcutoff)
    conf_cutoff_est = int(args.confcutoffest)
    conf_cutoff_gs = int(args.confcutoffgs)

    all_results = dict()
    all_failures = 0
    for scene in scenes:
        metrics, failures = compute_scene_metrics(
            dataset_path, submission_zip_gs, submission_zip_nogs, scene, rot_diff_cutoff, trans_diff_cutoff, conf_cutoff_est, conf_cutoff_gs)

        all_results[scene] = metrics
        all_failures += failures

    # unexpected_scene_count = count_unexpected_scenes(scenes, submission_zip)
    # if unexpected_scene_count > 0:
    #     logging.warning(
    #         f'Submission contains estimates for {unexpected_scene_count} scenes outside the {args.split} set')

    if all((len(metrics) == 0 for metrics in all_results.values())):
        logging.error( 
            f'Submission does not have any valid pose estimates')
        return

    output_metrics = aggregate_results(all_results, all_failures)

    name = "results"
    if rot_diff_cutoff:
        name += ("_r" + str(rot_diff_cutoff))
    else:
        name += ("_r" + str(0))
    if trans_diff_cutoff:
        name += ("_t" + str(trans_diff_cutoff))
    else:
        name += ("_t" + str(0))
    if conf_cutoff_est:
        name += ("_e" + str(conf_cutoff_est))
    else:
        name += ("_e" + str(0))
    if conf_cutoff_gs:
        name += ("_g" + str(conf_cutoff_gs))
    else:
        name += ("_g" + str(0))
        
    with open(f"results/{str(args.submission_path_gs).split('/')[1]}/{name}.txt", mode='w', encoding='utf-8') as f:
        json.dump(output_metrics, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'eval', description='Evaluate submissions for the MapFree dataset benchmark')
    parser.add_argument('submission_path_gs', type=Path,
                        help='Path to the guided submission ZIP file')
    parser.add_argument('--nogs', type=Path, default=None,
                        help='Path to the non-guided submission ZIP file')
    parser.add_argument('--split', choices=('val', 'test'), default='test',
                        help='Dataset split to use for evaluation. Default: test')
    parser.add_argument('--log', choices=('warning', 'info', 'error'),
                        default='warning', help='Logging level. Default: warning')
    parser.add_argument('--dataset_path', type=Path, default=None,
                        help='Path to the dataset folder')
    parser.add_argument('--rotcutoff', type=float, default=0)
    parser.add_argument('--transcutoff', type=float, default=0)
    parser.add_argument('--confcutoffest', type=float, default=0)
    parser.add_argument('--confcutoffgs', type=float, default=0)

    args = parser.parse_args()

    if args.dataset_path is None:
        cfg.merge_from_file('config/mapfree.yaml')
        args.dataset_path = Path(cfg.DATASET.DATA_ROOT)

    logging.basicConfig(level=args.log.upper())
    main(args)