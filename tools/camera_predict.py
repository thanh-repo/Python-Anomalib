import numpy as np
import matplotlib.pyplot as plt
import os, pprint, yaml, warnings, math, glob, cv2, random, logging
import anomalib
from pytorch_lightning import Trainer, seed_everything
from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.models import get_model
from anomalib.utils.callbacks import LoadModelCallback, get_callbacks
from anomalib.utils.loggers import configure_logger, get_experiment_logger
import torch


def warn(*args, **kwargs):
    pass


# update yaml key's value
def update_yaml(old_yaml, new_yaml, new_update):
    # load yaml
    with open(old_yaml) as f:
        old = yaml.safe_load(f)

    temp = []

    def set_state(old, key, value):
        if isinstance(old, dict):
            for k, v in old.items():
                if k == 'project':
                    temp.append(k)
                if k == key:
                    if temp and k == 'path':
                        # right now, we don't wanna change `project.path`
                        continue
                    old[k] = value
                elif isinstance(v, dict):
                    set_state(v, key, value)

    # iterate over the new update key-value pari
    for key, value in new_update.items():
        set_state(old, key, value)

    # save the updated / modified yaml file
    with open(new_yaml, 'w') as f:
        yaml.safe_dump(old, f, default_flow_style=False)


warnings.warn = warn
warnings.filterwarnings('ignore')
logger = logging.getLogger("anomalib")


def check_torch():
    torch_cuda = torch.cuda
    print("Torch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    print("GPU availability:", torch_cuda.is_available())
    print("Number of GPU devices:", torch.cuda.device_count())
    if torch_cuda.is_available():
        print("Name of current GPU:", torch.cuda.get_device_name(0))


def test():
    yaml_path = "datasets/surface_crack/config.yaml"
    with open(yaml_path) as f:
        updated_config = yaml.safe_load(f)
    if updated_config['project']['seed'] != 0:
        print(updated_config['project']['seed'])
        seed_everything(updated_config['project']['seed'])
    config = get_configurable_parameters(
        model_name=updated_config['model']['name'],
        config_path=yaml_path
    )
    model = get_model(config)
    experiment_logger = get_experiment_logger(config)
    callbacks = get_callbacks(config)
    datamodule = get_datamodule(config)
    trainer = Trainer(**config.trainer, logger=experiment_logger, callbacks=callbacks)
    load_model_callback = LoadModelCallback(
        weights_path=trainer.checkpoint_callback.best_model_path
    )
    trainer.callbacks.insert(0, load_model_callback)
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    check_torch()
    test()