# python 3.7
"""Helper functions."""

from .model_settings import MODEL_POOL
from .stylegan_generator import StyleGANGenerator
from .stylegan_encoder import StyleGANEncoder
from .perceptual_model import PerceptualModel

__all__ = ['build_generator', 'build_encoder', 'build_perceptual']


def build_generator(model_name, logger=None):
  """Builds generator module by model name."""
  if model_name not in MODEL_POOL:
    raise ValueError(f'Model `{model_name}` is not registered in '
                     f'`MODEL_POOL` in `model_settings.py`!')

  gan_type = model_name.split('_')[0]
  if gan_type in ['styleganinv']:
    return StyleGANGenerator(model_name, logger=logger)
  raise NotImplementedError(f'Unsupported GAN type `{gan_type}`!')


def build_encoder(model_name, logger=None):
  """Builds encoder module by model name."""
  if model_name not in MODEL_POOL:
    raise ValueError(f'Model `{model_name}` is not registered in '
                     f'`MODEL_POOL` in `model_settings.py`!')

  gan_type = model_name.split('_')[0]
  if gan_type == 'styleganinv':
    return StyleGANEncoder(model_name, logger=logger)
  raise NotImplementedError(f'Unsupported GAN type `{gan_type}`!')


build_perceptual = PerceptualModel
