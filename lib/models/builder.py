import torch.cuda

from lib.models.regression.model import RegressionModel
from lib.models.matching.model import FeatureMatchingModel
from lib.models.posediffusion.model.get_model import getPoseDiffusionModel


def build_model(cfg, checkpoint=''):
    if cfg.MODEL == 'FeatureMatching':
        return FeatureMatchingModel(cfg)
    elif cfg.MODEL == 'Regression':
        model = RegressionModel.load_from_checkpoint(checkpoint, cfg=cfg) if \
            checkpoint != '' else RegressionModel(cfg)
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        return model
    elif cfg.MODEL == 'PoseDiffusion':
        model = getPoseDiffusionModel(cfg)
        model.eval()
        return model
    else:
        raise NotImplementedError()
