# python 3.7
"""Contains the VGG16 model for perceptual feature extraction."""

import os
from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn

from . import model_settings

__all__ = ['VGG16', 'PerceptualModel']

_WEIGHT_PATH = os.path.join(model_settings.MODEL_DIR, 'vgg16.pth')

_MEAN_STATS = (103.939, 116.779, 123.68)


class VGG16(nn.Sequential):
  """Defines the VGG16 structure as the perceptual network.

  This models takes `RGB` images with pixel range [-1, 1] and data format `NCHW`
  as raw inputs. This following operations will be performed to preprocess the
  inputs (as defined in `keras.applications.imagenet_utils.preprocess_input`):
  (1) Shift pixel range to [0, 255].
  (3) Change channel order to `BGR`.
  (4) Subtract the statistical mean.

  NOTE: The three fully connected layers on top of the model are dropped.
  """

  def __init__(self, output_layer_idx=23, min_val=-1.0, max_val=1.0):
    """Defines the network structure.

    Args:
      output_layer_idx: Index of layer whose output will be used as perceptual
        feature. (default: 23, which is the `block4_conv3` layer activated by
        `ReLU` function)
      min_val: Minimum value of the raw input. (default: -1.0)
      max_val: Maximum value of the raw input. (default: 1.0)
    """
    sequence = OrderedDict({
        'layer0': nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        'layer1': nn.ReLU(inplace=True),
        'layer2': nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        'layer3': nn.ReLU(inplace=True),
        'layer4': nn.MaxPool2d(kernel_size=2, stride=2),
        'layer5': nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        'layer6': nn.ReLU(inplace=True),
        'layer7': nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        'layer8': nn.ReLU(inplace=True),
        'layer9': nn.MaxPool2d(kernel_size=2, stride=2),
        'layer10': nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        'layer11': nn.ReLU(inplace=True),
        'layer12': nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        'layer13': nn.ReLU(inplace=True),
        'layer14': nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        'layer15': nn.ReLU(inplace=True),
        'layer16': nn.MaxPool2d(kernel_size=2, stride=2),
        'layer17': nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
        'layer18': nn.ReLU(inplace=True),
        'layer19': nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        'layer20': nn.ReLU(inplace=True),
        'layer21': nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        'layer22': nn.ReLU(inplace=True),
        'layer23': nn.MaxPool2d(kernel_size=2, stride=2),
        'layer24': nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        'layer25': nn.ReLU(inplace=True),
        'layer26': nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        'layer27': nn.ReLU(inplace=True),
        'layer28': nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        'layer29': nn.ReLU(inplace=True),
        'layer30': nn.MaxPool2d(kernel_size=2, stride=2),
    })
    self.output_layer_idx = output_layer_idx
    self.min_val = min_val
    self.max_val = max_val
    self.mean = torch.from_numpy(np.array(_MEAN_STATS)).view(1, 3, 1, 1)
    self.mean = self.mean.type(torch.FloatTensor)
    super().__init__(sequence)

  def forward(self, x):
    x = (x - self.min_val) * 255.0 / (self.max_val - self.min_val)
    x = x[:, [2, 1, 0], :, :]
    x = x - self.mean.to(x.device)
    for i in range(self.output_layer_idx):
      x = self.__getattr__(f'layer{i}')(x)
    return x


class PerceptualModel(object):
  """Defines the perceptual model class."""

  def __init__(self, output_layer_idx=23, min_val=-1.0, max_val=1.0):
    """Initializes."""
    self.use_cuda = model_settings.USE_CUDA and torch.cuda.is_available()
    self.batch_size = model_settings.MAX_IMAGES_ON_DEVICE
    self.ram_size = model_settings.MAX_IMAGES_ON_RAM
    self.run_device = 'cuda' if self.use_cuda else 'cpu'
    self.cpu_device = 'cpu'

    self.output_layer_idx = output_layer_idx
    self.image_channels = 3
    self.min_val = min_val
    self.max_val = max_val
    self.net = VGG16(output_layer_idx=self.output_layer_idx,
                     min_val=self.min_val,
                     max_val=self.max_val)

    self.weight_path = _WEIGHT_PATH

    if not os.path.isfile(self.weight_path):
      raise IOError('No pre-trained weights found for perceptual model!')
    self.net.load_state_dict(torch.load(self.weight_path))
    self.net.eval().to(self.run_device)

  def get_batch_inputs(self, inputs, batch_size=None):
    """Gets inputs within mini-batch.

    This function yields at most `self.batch_size` inputs at a time.

    Args:
      inputs: Input data to form mini-batch.
      batch_size: Batch size. If not specified, `self.batch_size` will be used.
        (default: None)
    """
    total_num = inputs.shape[0]
    batch_size = batch_size or self.batch_size
    for i in range(0, total_num, batch_size):
      yield inputs[i:i + batch_size]

  def _extract(self, images):
    """Extracts perceptual feature within mini-batch."""
    if (images.ndim != 4 or images.shape[0] <= 0 or
        images.shape[0] > self.batch_size or images.shape[1] not in [1, 3]):
      raise ValueError(f'Input images should be with shape [batch_size, '
                       f'channel, height, width], where '
                       f'`batch_size` no larger than {self.batch_size}, '
                       f'`channel` equals to 1 or 3!\n'
                       f'But {images.shape} is received!')
    if images.shape[1] == 1:
      images = np.tile(images, (1, 1, 1, 3))
    if images.shape[1] != self.image_channels:
      raise ValueError(f'Number of channels of input image, which is '
                       f'{images.shape[1]}, is not supported by the current '
                       f'perceptual model, which requires '
                       f'{self.image_channels} channels!')
    x = torch.from_numpy(images).type(torch.FloatTensor).to(self.run_device)
    f = self.net(x)
    return f.to(self.cpu_device).detach().numpy()

  def extract(self, images):
    """Extracts perceptual feature from input images."""
    if images.shape[0] > self.ram_size:
      self.logger.warning(f'Number of inputs on RAM is larger than '
                          f'{self.ram_size}. Please use '
                          f'`self.get_batch_inputs()` to split the inputs! '
                          f'Otherwise, it may encounter OOM problem!')

    results = []
    for batch_images in self.get_batch_inputs(images):
      results.append(self._extract(batch_images))

    return np.concatenate(results, axis=0)
