import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from zipfile import ZipFile
import os

import torch
import numpy as np
from tqdm import tqdm

from config.default import cfg
from lib.datasets.datamodules import DataModule
from lib.models.builder import build_model
from lib.utils.data import data_to_model_device
from transforms3d.quaternions import mat2quat

import time


@dataclass
class Pose:
    image_name: str
    q: np.ndarray
    t: np.ndarray
    inliers: float

    def __str__(self) -> str:
        formatter = {'float': lambda v: f'{v:.6f}'}
        max_line_width = 1000
        q_str = np.array2string(self.q, formatter=formatter, max_line_width=max_line_width)[1:-1]
        t_str = np.array2string(self.t, formatter=formatter, max_line_width=max_line_width)[1:-1]
        return f'{self.image_name} {q_str} {t_str} {self.inliers}'


def predict(loader, model, output_root):
    results_dict = defaultdict(list)
    os.environ["out_root"] = str(output_root)

    for data in tqdm(loader):
        # run inference
        f = open(f"./{output_root}/ggs_results.txt", "a")
        f.write(f"{data['scene_id'][0]} ")
        f.write(f"{data['pair_names'][0][0]} ")
        f.write(f"{data['pair_names'][1][0]}\n")
        f.close()
        scene_id = data['scene_id'][0]
        print(scene_id)
        # print(data['pair_names'][0][0])
        q_frame = data['pair_names'][1][0]
        print(q_frame)
        """selected_frames = [('s00523', 'seq1/frame_00000.jpg'),
            ('s00485', 'seq1/frame_00055.jpg'),
            ('s00505', 'seq1/frame_00305.jpg'),
            ('s00485', 'seq1/frame_00015.jpg'),
            ('s00485', 'seq1/frame_00005.jpg'),
            ('s00505', 'seq1/frame_00495.jpg'),
            ('s00514', 'seq1/frame_00455.jpg'),
            ('s00517', 'seq1/frame_00335.jpg'),
            ('s00492', 'seq1/frame_00200.jpg'),
            ('s00504', 'seq1/frame_00185.jpg'),
            ('s00520', 'seq1/frame_00055.jpg'),
            ('s00492', 'seq1/frame_00415.jpg'),
            ('s00472', 'seq1/frame_00520.jpg'),
            ('s00461', 'seq1/frame_00415.jpg'),
            ('s00515', 'seq1/frame_00455.jpg'),
            ('s00484', 'seq1/frame_00390.jpg'),
            ('s00485', 'seq1/frame_00045.jpg'),
            ('s00514', 'seq1/frame_00460.jpg'),
            ('s00515', 'seq1/frame_00520.jpg'),
            ('s00504', 'seq1/frame_00125.jpg'),
            ('s00504', 'seq1/frame_00015.jpg'),
            ('s00503', 'seq1/frame_00065.jpg'),
            ('s00506', 'seq1/frame_00320.jpg'),
            ('s00462', 'seq1/frame_00325.jpg'),
            ('s00492', 'seq1/frame_00145.jpg'),
            ('s00505', 'seq1/frame_00320.jpg'),
            ('s00509', 'seq1/frame_00075.jpg'),
            ('s00517', 'seq1/frame_00110.jpg'),
            ('s00505', 'seq1/frame_00350.jpg'),
            ('s00484', 'seq1/frame_00245.jpg'),
            ('s00515', 'seq1/frame_00480.jpg'),
            ('s00492', 'seq1/frame_00105.jpg'),
            ('s00465', 'seq1/frame_00555.jpg'),
            ('s00465', 'seq1/frame_00260.jpg'),
            ('s00509', 'seq1/frame_00030.jpg'),
            ('s00484', 'seq1/frame_00525.jpg'),
            ('s00472', 'seq1/frame_00425.jpg'),
            ('s00504', 'seq1/frame_00200.jpg'),
            ('s00491', 'seq1/frame_00405.jpg'),
            ('s00519', 'seq1/frame_00100.jpg'),
            ('s00480', 'seq1/frame_00035.jpg'),
            ('s00515', 'seq1/frame_00535.jpg'),
            ('s00461', 'seq1/frame_00390.jpg'),
            ('s00492', 'seq1/frame_00315.jpg'),
            ('s00515', 'seq1/frame_00490.jpg'),
            ('s00515', 'seq1/frame_00485.jpg'),
            ('s00504', 'seq1/frame_00005.jpg'),
            ('s00472', 'seq1/frame_00310.jpg'),
            ('s00491', 'seq1/frame_00355.jpg'),
            ('s00517', 'seq1/frame_00410.jpg'),
            ('s00479', 'seq1/frame_00040.jpg'),
            ('s00461', 'seq1/frame_00450.jpg'),
            ('s00492', 'seq1/frame_00365.jpg'),
            ('s00492', 'seq1/frame_00280.jpg'),
            ('s00484', 'seq1/frame_00295.jpg'),
            ('s00472', 'seq1/frame_00450.jpg'),
            ('s00515', 'seq1/frame_00255.jpg'),
            ('s00520', 'seq1/frame_00070.jpg'),
            ('s00492', 'seq1/frame_00255.jpg'),
            ('s00479', 'seq1/frame_00045.jpg'),
            ('s00479', 'seq1/frame_00055.jpg'),
            ('s00510', 'seq1/frame_00315.jpg'),
            ('s00463', 'seq1/frame_00010.jpg'),
            ('s00511', 'seq1/frame_00525.jpg'),
            ('s00505', 'seq1/frame_00360.jpg'),
            ('s00472', 'seq1/frame_00365.jpg'),
            ('s00472', 'seq1/frame_00420.jpg'),
            ('s00485', 'seq1/frame_00065.jpg'),
            ('s00461', 'seq1/frame_00380.jpg'),
            ('s00479', 'seq1/frame_00035.jpg'),
            ('s00480', 'seq1/frame_00070.jpg'),
            ('s00465', 'seq1/frame_00475.jpg'),
            ('s00503', 'seq1/frame_00060.jpg'),
            ('s00466', 'seq1/frame_00340.jpg'),
            ('s00515', 'seq1/frame_00525.jpg'),
            ('s00484', 'seq1/frame_00545.jpg'),
            ('s00468', 'seq1/frame_00375.jpg'),
            ('s00492', 'seq1/frame_00140.jpg'),
            ('s00510', 'seq1/frame_00350.jpg'),
            ('s00504', 'seq1/frame_00065.jpg'),
            ('s00504', 'seq1/frame_00000.jpg'),
            ('s00510', 'seq1/frame_00235.jpg'),
            ('s00517', 'seq1/frame_00320.jpg'),
            ('s00479', 'seq1/frame_00015.jpg'),
            ('s00484', 'seq1/frame_00500.jpg'),
            ('s00474', 'seq1/frame_00405.jpg'),
            ('s00479', 'seq1/frame_00070.jpg'),
            ('s00465', 'seq1/frame_00160.jpg'),
            ('s00499', 'seq1/frame_00065.jpg'),
            ('s00523', 'seq1/frame_00025.jpg'),
            ('s00494', 'seq1/frame_00370.jpg'),
            ('s00510', 'seq1/frame_00280.jpg'),
            ('s00515', 'seq1/frame_00460.jpg'),
            ('s00514', 'seq1/frame_00415.jpg'),
            ('s00504', 'seq1/frame_00230.jpg'),
            ('s00462', 'seq1/frame_00565.jpg'),
            ('s00461', 'seq1/frame_00485.jpg'),
            ('s00474', 'seq1/frame_00385.jpg'),
            ('s00517', 'seq1/frame_00290.jpg'),
            ('s00497', 'seq1/frame_00465.jpg')]
        """
        selected_frames = [('s00523', 'seq1/frame_00000.jpg')]
        # if (scene_id, q_frame) not in selected_frames:
        #     continue
        data = data_to_model_device(data, model)
        with torch.no_grad():
            results = model(data)
            if len(results) == 3:
                R, t, c = results
            else:
                R, t = results
                c = None

        # ignore frames without poses (e.g. not enough feature matches)
        if (R is None) or (t is None):
            print("NO ESTIMATE!")
            continue

        R = R.detach().cpu().numpy()
        t = t.reshape(-1).detach().cpu().numpy()
        inliers = data['inliers']
        scene = data['scene_id'][0]
        query_img = data['pair_names'][1][0]
        if c is None and inliers is not None:
            c = inliers

        # ignore frames without poses (e.g. not enough feature matches)
        if np.isnan(R).any() or np.isnan(t).any() or np.isinf(t).any() or c==-1:
            print("NO ESTIMATE!")
            continue

        # populate results_dict
        estimated_pose = Pose(image_name=query_img,
                              q=mat2quat(R).reshape(-1),
                              t=t.reshape(-1),
                              inliers=c)
        results_dict[scene].append(estimated_pose)
        # print("final output")
        # print(estimated_pose)

    return results_dict


def save_submission(results_dict: dict, output_path: Path):
    with ZipFile(output_path, 'w') as zip:
        for scene, poses in results_dict.items():
            poses_str = '\n'.join((str(pose) for pose in poses))
            zip.writestr(f'pose_{scene}.txt', poses_str.encode('utf-8'))


def eval(args):
    # Load configs
    cfg.merge_from_file('config/mapfree.yaml')
    cfg.merge_from_file(args.config)

    # Create dataloader
    if args.split == 'test':
        dataloader = DataModule(cfg).test_dataloader()
    elif args.split == 'val':
        cfg.TRAINING.BATCH_SIZE = 1
        cfg.TRAINING.NUM_WORKERS = 1
        dataloader = DataModule(cfg).val_dataloader()
    else:
        raise NotImplemented(f'Invalid split: {args.split}')

    # Create model
    model = build_model(cfg, args.checkpoint)

    # Get predictions from model

    # Save predictions to txt per scene within zip
    args.output_root.mkdir(parents=True, exist_ok=True)
    results_dict = predict(dataloader, model, args.output_root)
    save_submission(results_dict, args.output_root / 'submission.zip')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='path to config file')
    parser.add_argument(
        '--checkpoint', help='path to model checkpoint (models with learned parameters)',
        default='')
    parser.add_argument('--output_root', '-o', type=Path, default=Path('results/'))
    parser.add_argument(
        '--split', choices=('val', 'test'),
        default='test',
        help='Dataset split to use for evaluation. Choose from test or val. Default: test')

    args = parser.parse_args()
    eval(args)