# python 3.7
"""Contains the encoder class of StyleGAN inversion.

This class is derived from the `BaseEncoder` class defined in `base_encoder.py`.
"""

import numpy as np

import torch

from .base_encoder import BaseEncoder
from .stylegan_encoder_network import StyleGANEncoderNet

__all__ = ['StyleGANEncoder']


class StyleGANEncoder(BaseEncoder):
  """Defines the encoder class of StyleGAN inversion."""

  def __init__(self, model_name, logger=None):
    self.gan_type = 'stylegan'
    super().__init__(model_name, logger)

  def build(self):
    self.w_space_dim = getattr(self, 'w_space_dim', 512)
    self.encoder_channels_base = getattr(self, 'encoder_channels_base', 64)
    self.encoder_channels_max = getattr(self, 'encoder_channels_max', 1024)
    self.use_wscale = getattr(self, 'use_wscale', False)
    self.use_bn = getattr(self, 'use_bn', False)
    self.net = StyleGANEncoderNet(
        resolution=self.resolution,
        w_space_dim=self.w_space_dim,
        image_channels=self.image_channels,
        encoder_channels_base=self.encoder_channels_base,
        encoder_channels_max=self.encoder_channels_max,
        use_wscale=self.use_wscale,
        use_bn=self.use_bn)
    self.num_layers = self.net.num_layers
    self.encode_dim = [self.num_layers, self.w_space_dim]

  def _encode(self, images):
    if not isinstance(images, np.ndarray):
      raise ValueError(f'Latent codes should be with type `numpy.ndarray`!')
    if (images.ndim != 4 or images.shape[0] <= 0 or
        images.shape[0] > self.batch_size or images.shape[1:] != (
            self.image_channels, self.resolution, self.resolution)):
      raise ValueError(f'Input images should be with shape [batch_size, '
                       f'channel, height, width], where '
                       f'`batch_size` no larger than {self.batch_size}, '
                       f'`channel` equals to {self.image_channels}, '
                       f'`height` and `width` equal to {self.resolution}!\n'
                       f'But {images.shape} is received!')

    xs = self.to_tensor(images.astype(np.float32))
    codes = self.net(xs)
    assert codes.shape == (images.shape[0], np.prod(self.encode_dim))
    codes = codes.view(codes.shape[0], *self.encode_dim)
    results = {
        'image': images,
        'code': self.get_value(codes),
    }

    if self.use_cuda:
      torch.cuda.empty_cache()

    return results

  def encode(self, images, **kwargs):
    return self.batch_run(images, self._encode)
