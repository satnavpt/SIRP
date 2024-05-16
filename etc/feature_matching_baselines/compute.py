import os
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

from utils import parse_7scenes_matching_pairs, parse_mapfree_query_frames, stack_pts, stack_lis, load_scannet_imgpaths
from matchers import LoFTR_matcher, SuperGlue_matcher, SIFT_matcher, HLOC_matcher, GLUE_matcher

MATCHERS = {'LoFTR': LoFTR_matcher, 'SG': SuperGlue_matcher, 'SIFT': SIFT_matcher, "HLOC": HLOC_matcher, "GLUE": GLUE_matcher}


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-ds', type=str, default='7Scenes',
                        choices=['Scannet', '7Scenes', 'Mapfree'])
    parser.add_argument('--matcher', '-m', type=str, default='SIFT',
                        choices=MATCHERS.keys())
    parser.add_argument('--scenes', '-sc', type=str, nargs='*', default=None)
    parser.add_argument('--pair_txt', type=str,
                        default='test_pairs.5nn.5cm10m.vlad.minmax.txt')  # 7Scenes
    parser.add_argument('--pair_npz', type=str,
                        default='../../data/scannet_indices/scene_data/test/test.npz')  # Scannet
    parser.add_argument('--outdoor', action='store_false',
                        help='use outdoor SG/LoFTR model. If not specified, use outdoor models')
    parser.add_argument('--feature_conf', "-fe")
    parser.add_argument('--matcher_conf', "-ma")
    args = parser.parse_args()

    dataset = args.dataset
    if dataset == '7Scenes':
        args.data_root = '../../data/sevenscenes'
        scenes = ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']
        args.scenes = scenes if not args.scenes else args.scenes
        resize = 640, 480
    elif dataset == 'Scannet':
        args.data_root = '../../data/scannet/scans_test'
        resize = 640, 480
    elif dataset == 'Mapfree':
        args.data_root = Path('../../../SIRP/datasets/mfl/')
        test_scenes = [folder for folder in (args.data_root / 'test').iterdir() if folder.is_dir()]
        val_scenes = [folder for folder in (args.data_root / 'val').iterdir() if folder.is_dir()]
        # args.scenes = test_scenes + val_scenes
        args.scenes = val_scenes
        resize = 540, 720

    if args.matcher == "HLOC":
        matcher = MATCHERS[args.matcher](args.feature_conf, args.matcher_conf, resize, args.outdoor)
        args.matcher += ("_" + args.feature_conf + "_" + args.matcher_conf)
        return args, matcher
    return args, MATCHERS[args.matcher](resize, args.outdoor)


if __name__ == '__main__':
    args, matcher = get_parser()

    print(args.matcher)

    if args.dataset == '7Scenes':
        for scene in args.scenes:
            scene_dir = Path(args.data_root) / scene
            im_pairs = parse_7scenes_matching_pairs(
                str(scene_dir / args.pair_txt))  # {(im1, im2) : (q, t, ess_mat)}
            pair_names = list(im_pairs.keys())
            im_pairs_path = [(str(scene_dir / train_im),
                              str(scene_dir / test_im)) for (train_im, test_im) in pair_names]

            pts_stack = list()
            print(f'Started {scene}')
            for pair in tqdm(im_pairs_path):
                pts = matcher.match(pair)
                pts_stack.append(pts)
            pts_stack = stack_pts(pts_stack)
            results = {'correspondences': pts_stack}
            np.savez_compressed(os.path.join(
                scene_dir,
                f'correspondences_{args.matcher}_{args.pair_txt}.npz'),
                **results)
            print(f'Finished {scene}')

            del results

    elif args.dataset == 'Mapfree':
        i = 0
        for scene_dir in tqdm(args.scenes):
            print(f'Started {scene_dir.name}')
            if args.matcher == "GLUE":
                if scene_dir.name in ["s00497", "s00509", "s00475", "s00492", "s00505", "s00508", "s00470", "s00489", "s00491", "s00517", "s00471", "s00474", "s00518", "s00515", "s00520", "s00464", "s00469", "s00499", "s00488", "s00482", "s00473", "s00523", "s00495", "s00500", "s00512", "s00483", "s00516", "s00462", "s00503", "s00504", "s00465", "s00522", "s00498", "s00507", "s00524", "s00490", "s00502", "s00463", "s00479", "s00513", "s00501"]:
                    continue
            query_frames_paths = parse_mapfree_query_frames(scene_dir / 'poses.txt')
            im_pairs_path = [(str(scene_dir / 'seq0' / 'frame_00000.jpg'),
                              str(scene_dir / qpath)) for qpath in query_frames_paths]

            pts_stack = list()
            lis_stack = list()
            print(f'Started {scene_dir.name}')
            for pair in tqdm(im_pairs_path):
                res = matcher.match(pair)
                if type(res) is tuple:
                    pts, lis = res
                    pts_stack.append(pts)
                    lis_stack.append(lis)
                else:
                    pts = res
                    pts_stack.append(pts)
            pts_stack = stack_pts(pts_stack)

            if len(lis_stack) > 0:
                lis_stack = stack_lis(lis_stack)
                results = {'correspondences': pts_stack, 'line_correspondences': lis_stack}
            else:
                results = {'correspondences': pts_stack}
            np.savez_compressed(scene_dir / f'correspondences_{args.matcher}.npz', **results)
            print(f'Finished {scene_dir.name}')

    elif args.dataset == 'Scannet':
        im_pairs_path = load_scannet_imgpaths(args.pair_npz, args.data_root)
        pts_stack = list()
        print(f'Started Scannet')
        for pair in tqdm(im_pairs_path):
            pts = matcher.match(pair)
            pts_stack.append(pts)
        pts_stack = stack_pts(pts_stack)
        results = {'correspondences': pts_stack}
        np.savez_compressed(
            f'../../data/scannet_misc/correspondences_{args.matcher}_scannet_test.npz',
            **results)
        print(f'Finished Scannet')
    else:
        raise NotImplementedError('Invalid dataset')
