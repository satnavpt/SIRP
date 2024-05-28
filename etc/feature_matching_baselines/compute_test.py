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
        args.scenes = test_scenes
        # args.scenes = val_scenes
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
                if scene_dir.name in ['s00620', 's00551', 's00560', 's00571', 's00554', 's00589', 's00651', 's00647', 's00559', 's00562', 's00530', 's00644', 's00638', 's00607', 's00553', 's00525', 's00624', 's00548', 's00629', 's00539', 's00634', 's00538', 's00612', 's00583', 's00611', 's00570', 's00541', 's00641', 's00606', 's00584', 's00582', 's00637', 's00615', 's00642', 's00542', 's00565', 's00529', 's00630', 's00613', 's00622', 's00618', 's00545', 's00627', 's00555', 's00597', 's00588', 's00646', 's00600', 's00631', 's00533', 's00639', 's00566', 's00609', 's00572', 's00568', 's00563', 's00633', 's00598', 's00527', 's00578', 's00614', 's00537', 's00636', 's00552', 's00564', 's00592', 's00586', 's00567', 's00587', 's00648', 's00531', 's00608', 's00532', 's00652', 's00649', 's00540', 's00593', 's00576', 's00621', 's00536', 's00616', 's00557', 's00601', 's00558', 's00645', 's00534', 's00626', 's00643', 's00528', 's00574', 's00580', 's00549', 's00543', 's00653', 's00628', 's00546', 's00575', 's00650', 's00603', 's00547', 's00602', 's00579', 's00585', 's00561', 's00623', 's00526', 's00610', 's00590', 's00596', 's00535', 's00573', 's00550', 's00654', 's00640', 's00635', 's00569', 's00577', 's00556', 's00595', 's00581', 's00605', 's00544']:
                    continue
            query_frames_paths = parse_mapfree_query_frames(scene_dir / 'poses.txt')
            # dirs = os.listdir(scene_dir / 'seq1')
            # print(dirs)
            # dirs = list('seq1/' + s for s in dirs if not ("dpt" in s))
            # print(dirs)
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
