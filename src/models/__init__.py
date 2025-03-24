import os 
os.environ['WANDB_DIR'] = '/vision/u/emilyjin/.cache/wandb'
os.environ['WANDB_CACHE_DIR'] = '/vision/u/emilyjin/.cache/wandb'
os.environ['TORCH_HOME'] = '/vision/u/emilyjin/abstractions/tmp'
os.environ['TMPDIR'] = '/vision/u/emilyjin/abstractions/tmp'

from .base_model import EndToEndModel
from .our_model import ObjectCentricModel
from .film_model import FiLMModel


MODEL_CLASSES = {
    'EndToEnd': EndToEndModel,
    'ObjectCentric': ObjectCentricModel,
    'FiLM': FiLMModel
}