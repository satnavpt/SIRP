import hydra
import os
import torch
from hydra.utils import instantiate, get_original_cwd

from ..util.match_extraction import extract_match


def getPoseDiffusionModel(cfg, checkpoint=None):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)

    model = instantiate(cfg.MODEL_PARAMS, POSE_SOLVER=cfg.POSE_SOLVER, PROCRUSTES=cfg.PROCRUSTES, MATCHING=cfg, GGS=cfg.GGS, _recursive_=False)

    # original_cwd = get_original_cwd()
    # ckpt_path = os.path.join(original_cwd, checkpoint)
    if os.path.isfile(cfg.CHKPT):
        checkpoint = torch.load(cfg.CHKPT, map_location=device)
        model.load_state_dict(checkpoint, strict=True)
        print(f"Loaded checkpoint from: {cfg.CHKPT}")
    else:
        raise ValueError(f"No checkpoint found at: {cfg.CHKPT}")

    model = model.to(device)

    return model