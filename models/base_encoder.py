# python 3.7
"""Contains the base class for encoder in a GAN model.

In addition to the two-player game between generator and discriminator in GANs,
some work introduce an extra encoder to facilitate real image inference. The
encoder can be view as the reverse of the generator, which projects a given
image back to the latent space. This kind of work is also widely known as `GAN
Inversion`.

This class is derived from the `BaseModule` class defined in `base_module.py`.
"""

import numpy as np

from .base_module import BaseModule

__all__ = ['BaseEncoder']


class BaseEncoder(BaseModule):
  """Base class for encoder used in GAN inversion."""

  def __init__(self, model_name, logger=None):
    """Initializes the encoder with model name."""
    self.encode_dim = None  # Target shape of the encoded code.
    super().__init__(model_name, 'encoder', logger)
    assert self.encode_dim is not None
    assert isinstance(self.encode_dim, (list, tuple))

  def preprocess(self, images):
    """Preprocesses the input images if needed.

    This function assumes the input numpy array is with shape [batch_size,
    height, width, channel]. Here, `channel = 3` for color image and
    `channel = 1` for grayscale image. The returned images are with shape
    [batch_size, channel, height, width].

    NOTE: The channel order of input images is always assumed as `RGB`.

    Args:
      images: The raw inputs with dtype `numpy.uint8` and range [0, 255].

    Returns:
      The preprocessed images with dtype `numpy.float32` and range
        [self.min_val, self.max_val].

    Raises:
      ValueError: If the input `images` are not with type `numpy.ndarray` or not
        with dtype `numpy.uint8` or not with shape [batch_size, height, width,
        channel].
    """
    if not isinstance(images, np.ndarray):
      raise ValueError(f'Images should be with type `numpy.ndarray`!')
    if images.dtype != np.uint8:
      raise ValueError(f'Images should be with dtype `numpy.uint8`!')

    if images.ndim != 4 or images.shape[3] not in [1, 3]:
      raise ValueError(f'Input should be with shape [batch_size, height, width '
                       f'channel], where channel equals to 1 or 3!\n'
                       f'But {images.shape} is received!')
    if images.shape[3] == 1 and self.image_channels == 3:
      images = np.tile(images, (1, 1, 1, 3))
    if images.shape[3] != self.image_channels:
      raise ValueError(f'Number of channels of input image, which is '
                       f'{images.shape[3]}, is not supported by the current '
                       f'encoder, which requires {self.image_channels} '
                       f'channels!')
    if self.image_channels == 3 and self.channel_order == 'BGR':
      images = images[:, :, :, ::-1]
    images = images.astype(np.float32)
    images = images / 255.0 * (self.max_val - self.min_val) + self.min_val
    images = images.astype(np.float32).transpose(0, 3, 1, 2)

    return images

  def encode(self, images, **kwargs):
    """Encodes the input images to latent codes.

    NOTE: The images are assumed to have already been preprocessed.

    Args:
      images: Input images to encode.

    Returns:
      A dictionary whose values are raw outputs from the encoder. Keys of the
        dictionary usually include `image` and `code`.
    """
    raise NotImplementedError(f'Should be implemented in derived class!')

  def easy_encode(self, images, **kwargs):
    """Wraps functions `preprocess()` and `encode()` together."""
    return self.encode(self.preprocess(images), **kwargs)
