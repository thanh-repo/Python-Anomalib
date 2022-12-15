import numpy as np
import matplotlib.pyplot as plt
import os, pprint, warnings, math, glob, cv2, random, logging

from omegaconf import DictConfig, ListConfig


def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.filterwarnings('ignore')
logger = logging.getLogger("anomalib")

import torch
import anomalib
from pytorch_lightning import Trainer, seed_everything
from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.models import get_model
from anomalib.utils.callbacks import LoadModelCallback, get_callbacks
from anomalib.utils.loggers import configure_logger, get_experiment_logger




def tourch_gpu():
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.backends.cudnn.version())
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        print(torch.cuda.device_count())
        print(torch.cuda.current_device())
        print(torch.cuda.device(0))
        print(torch.cuda.get_device_name(0))


def model_fit():
    yaml_config = './datasets/surface_crack/config.yaml'
    yaml_config = '../anomalib/models/ganomaly/config.yaml'
    config: DictConfig | ListConfig = get_configurable_parameters(model_name='ganomaly',
                                                                  config_path=yaml_config)
    model = get_model(config)
    experiment_logger = get_experiment_logger(config)
    callbacks = get_callbacks(config)
    datamodule = get_datamodule(config)
    trainer = Trainer(**config.trainer, logger=experiment_logger, callbacks=callbacks)
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    tourch_gpu()
    model_fit()
