# python 3.7
"""Contains the base class for generator in a GAN model.

This class is derived from the `BaseModule` class defined in `base_module.py`.
"""

import numpy as np

from .base_module import BaseModule

__all__ = ['BaseGenerator']


class BaseGenerator(BaseModule):
  """Base class for generator used in GAN variants."""

  def __init__(self, model_name, logger=None):
    """Initializes the generator with model name."""
    super().__init__(model_name, 'generator', logger)

  def sample(self, num, **kwargs):
    """Samples latent codes randomly.

    Args:
      num: Number of latent codes to sample. Should be positive.

    Returns:
      A `numpy.ndarray` as sampled latend codes.
    """
    raise NotImplementedError(f'Should be implemented in derived class!')

  def preprocess(self, latent_codes, **kwargs):
    """Preprocesses the input latent codes if needed.

    Args:
      latent_codes: The input latent codes for preprocessing.

    Returns:
      The preprocessed latent codes which can be used as final inputs to the
        generator.
    """
    raise NotImplementedError(f'Should be implemented in derived class!')

  def easy_sample(self, num, **kwargs):
    """Wraps functions `sample()` and `preprocess()` together."""
    return self.preprocess(self.sample(num, **kwargs), **kwargs)

  def synthesize(self, latent_codes, **kwargs):
    """Synthesizes images with given latent codes.

    NOTE: The latent codes are assumed to have already been preprocessed.

    Args:
      latent_codes: Input latent codes for image synthesis.

    Returns:
      A dictionary whose values are raw outputs from the generator. Keys of the
        dictionary usually include `z` and `image`.
    """
    raise NotImplementedError(f'Should be implemented in derived class!')

  def postprocess(self, images):
    """Postprocesses the output images if needed.

    This function assumes the input numpy array is with shape [batch_size,
    channel, height, width]. Here, `channel = 3` for color image and
    `channel = 1` for grayscale image. The returned images are with shape
    [batch_size, height, width, channel].

    NOTE: The channel order of output images will always be `RGB`.

    Args:
      images: The raw outputs from the generator.

    Returns:
      The postprocessed images with dtype `numpy.uint8` and range [0, 255].

    Raises:
      ValueError: If the input `images` are not with type `numpy.ndarray` or not
        with shape [batch_size, channel, height, width].
    """
    if not isinstance(images, np.ndarray):
      raise ValueError(f'Images should be with type `numpy.ndarray`!')

    if images.ndim != 4 or images.shape[1] != self.image_channels:
      raise ValueError(f'Input should be with shape [batch_size, channel, '
                       f'height, width], where channel equals to '
                       f'{self.image_channels}!\n'
                       f'But {images.shape} is received!')
    images = (images - self.min_val) * 255 / (self.max_val - self.min_val)
    images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
    images = images.transpose(0, 2, 3, 1)
    if self.image_channels == 3 and self.channel_order == 'BGR':
      images = images[:, :, :, ::-1]

    return images

  def easy_synthesize(self, latent_codes, **kwargs):
    """Wraps functions `synthesize()` and `postprocess()` together."""
    outputs = self.synthesize(latent_codes, **kwargs)
    if 'image' in outputs:
      outputs['image'] = self.postprocess(outputs['image'])
    return outputs
