# python 3.7
"""Contains the implementation of generator described in StyleGAN.

For more details, please check the original paper:
https://arxiv.org/pdf/1812.04948.pdf
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['StyleGANGeneratorNet']

# Resolutions allowed.
_RESOLUTIONS_ALLOWED = [8, 16, 32, 64, 128, 256, 512, 1024]

# Initial resolution.
_INIT_RES = 4

# Fused-scale options allowed.
_FUSED_SCALE_OPTIONS_ALLOWED = [True, False, 'auto']

# Minimal resolution for `auto` fused-scale strategy.
_AUTO_FUSED_SCALE_MIN_RES = 128


class StyleGANGeneratorNet(nn.Module):
  """Defines the generator network in StyleGAN.

  NOTE: the generated images are with `RGB` color channels and range [-1, 1].
  """

  def __init__(self,
               resolution,
               z_space_dim=512,
               w_space_dim=512,
               num_mapping_layers=8,
               repeat_w=True,
               image_channels=3,
               final_tanh=False,
               label_size=0,
               fused_scale='auto',
               truncation_psi=0.7,
               truncation_layers=8,
               randomize_noise=False,
               fmaps_base=16 << 10,
               fmaps_max=512):
    """Initializes the generator with basic settings.

    Args:
      resolution: The resolution of the output image.
      z_space_dim: Dimension of the initial latent space. (default: 512)
      w_space_dim: Dimension of the disentangled latent space. (default: 512)
      num_mapping_layers: Number of fully-connected layers to map Z space to W
        space. (default: 8)
      repeat_w: Whether to use same w for different layers. (default: True)
      image_channels: Number of channels of output image. (default: 3)
      final_tanh: Whether to use tanh to control pixel range. (default: False)
      label_size: Size of additional labels. (default: 0)
      fused_scale: If set as `True`, `conv2d_transpose` is used for upscaling.
        If set as `False`, `upsample + conv2d` is used for upscaling. If set as
        `auto`, `upsample + conv2d` is used for bottom layers until resolution
        reaches `_AUTO_FUSED_SCALE_MIN_RES`. (default: `auto`)
      truncation_psi: Style strength multiplier for the truncation trick.
        `None` or `1.0` indicates no truncation. (default: 0.7)
      truncation_layers: Number of layers for which to apply the truncation
        trick. `None` or `0` indicates no truncation. (default: 8)
      randomize_noise: Whether to add random noise for each convolutional layer.
        (default: False)
      fmaps_base: Base factor to compute number of feature maps for each layer.
        (default: 16 << 10)
      fmaps_max: Maximum number of feature maps in each layer. (default: 512)

    Raises:
      ValueError: If the input `resolution` is not supported, or `fused_scale`
        is not supported.
    """
    super().__init__()

    if resolution not in _RESOLUTIONS_ALLOWED:
      raise ValueError(f'Invalid resolution: {resolution}!\n'
                       f'Resolutions allowed: {_RESOLUTIONS_ALLOWED}.')
    if fused_scale not in _FUSED_SCALE_OPTIONS_ALLOWED:
      raise ValueError(f'Invalid fused-scale option: {fused_scale}!\n'
                       f'Options allowed: {_FUSED_SCALE_OPTIONS_ALLOWED}.')

    self.init_res = _INIT_RES
    self.resolution = resolution
    self.z_space_dim = z_space_dim
    self.w_space_dim = w_space_dim
    self.num_mapping_layers = num_mapping_layers
    self.repeat_w = repeat_w
    self.image_channels = image_channels
    self.final_tanh = final_tanh
    self.label_size = label_size
    self.fused_scale = fused_scale
    self.truncation_psi = truncation_psi
    self.truncation_layers = truncation_layers
    self.randomize_noise = randomize_noise
    self.fmaps_base = fmaps_base
    self.fmaps_max = fmaps_max

    self.num_layers = int(np.log2(self.resolution // self.init_res * 2)) * 2

    mapping_space_dim = self.w_space_dim * (1 if repeat_w else self.num_layers)
    self.mapping = MappingModule(input_space_dim=self.z_space_dim,
                                 hidden_space_dim=self.fmaps_max,
                                 final_space_dim=mapping_space_dim,
                                 label_size=self.label_size,
                                 num_layers=self.num_mapping_layers)
    self.truncation = TruncationModule(num_layers=self.num_layers,
                                       w_space_dim=self.w_space_dim,
                                       repeat_w=self.repeat_w,
                                       truncation_psi=self.truncation_psi,
                                       truncation_layers=self.truncation_layers)
    self.synthesis = SynthesisModule(init_resolution=self.init_res,
                                     resolution=self.resolution,
                                     w_space_dim=self.w_space_dim,
                                     image_channels=self.image_channels,
                                     final_tanh=self.final_tanh,
                                     fused_scale=self.fused_scale,
                                     randomize_noise=self.randomize_noise,
                                     fmaps_base=self.fmaps_base,
                                     fmaps_max=self.fmaps_max)

  def forward(self, z, l=None):
    w = self.mapping(z, l)
    w = self.truncation(w)
    x = self.synthesis(w)
    return x


class MappingModule(nn.Module):
  """Implements the latent space mapping module.

  Basically, this module executes several dense layers in sequence.
  """

  def __init__(self,
               input_space_dim=512,
               hidden_space_dim=512,
               final_space_dim=512,
               label_size=0,
               num_layers=8,
               normalize_input=True):
    super().__init__()

    self.input_space_dim = input_space_dim
    self.label_size = label_size
    self.num_layers = num_layers

    self.norm = PixelNormLayer() if normalize_input else nn.Identity()

    for i in range(num_layers):
      dim_mul = 2 if label_size else 1
      in_dim = input_space_dim * dim_mul if i == 0 else hidden_space_dim
      out_dim = final_space_dim if i == (num_layers - 1) else hidden_space_dim
      self.add_module(f'dense{i}', DenseBlock(in_dim, out_dim))
    if label_size:
      self.label_weight = nn.Parameter(torch.randn(label_size, input_space_dim))

  def forward(self, z, l=None):
    if z.ndim != 2 or z.shape[1] != self.input_space_dim:
      raise ValueError(f'Input latent code should be with shape [batch_size, '
                       f'input_dim], where `input_dim` equals to '
                       f'{self.input_space_dim}!\n'
                       f'But {z.shape} is received!')
    if self.label_size:
      if l is None:
        raise ValueError(f'Model requires an additional label (with size '
                         f'{self.label_size}) as inputs, but no label is '
                         f'received!')
      if l.ndim != 2 or l.shape != (z.shape[0], self.label_size):
        raise ValueError(f'Input label should be with shape [batch_size, '
                         f'label_size], where `batch_size` equals to that of '
                         f'latent codes ({z.shape[0]}) and `label_size` equals '
                         f'to {self.label_size}!\n'
                         f'But {l.shape} is received!')
      embedding = torch.matmul(l, self.label_weight)
      z = torch.cat((z, embedding), dim=1)

    w = self.norm(z)
    for i in range(self.num_layers):
      w = self.__getattr__(f'dense{i}')(w)
    return w


class TruncationModule(nn.Module):
  """Implements the truncation module."""

  def __init__(self,
               num_layers,
               w_space_dim=512,
               repeat_w=True,
               truncation_psi=0.7,
               truncation_layers=8):
    super().__init__()

    self.num_layers = num_layers
    self.w_space_dim = w_space_dim
    self.repeat_w = repeat_w
    if truncation_psi is not None and truncation_layers is not None:
      self.use_truncation = True
    else:
      self.use_truncation = False
      truncation_psi = 1.0
      truncation_layers = 0

    self.register_buffer('w_avg', torch.zeros(w_space_dim))

    layer_idx = np.arange(self.num_layers).reshape(1, self.num_layers, 1)
    coefs = np.ones_like(layer_idx, dtype=np.float32)
    coefs[layer_idx < truncation_layers] *= truncation_psi
    self.register_buffer('truncation', torch.from_numpy(coefs))

  def forward(self, w):
    if w.ndim == 2:
      if self.repeat_w:
        assert w.shape[1] == self.w_space_dim
        w = w.view(-1, 1, self.w_space_dim).repeat(1, self.num_layers, 1)
      else:
        assert w.shape[1] == self.w_space_dim * self.num_layers
        w = w.view(-1, self.num_layers, self.w_space_dim)
    assert w.ndim == 3 and w.shape[1:] == (self.num_layers, self.w_space_dim)
    if self.use_truncation:
      w_avg = self.w_avg.view(1, 1, self.w_space_dim)
      w = w_avg + (w - w_avg) * self.truncation
    return w


class SynthesisModule(nn.Module):
  """Implements the image synthesis module.

  Basically, this module executes several convolutional layers in sequence.
  """

  def __init__(self,
               init_resolution=4,
               resolution=1024,
               w_space_dim=512,
               image_channels=3,
               final_tanh=False,
               fused_scale='auto',
               randomize_noise=False,
               fmaps_base=16 << 10,
               fmaps_max=512):
    super().__init__()

    self.init_res = init_resolution
    self.init_res_log2 = int(np.log2(self.init_res))
    self.resolution = resolution
    self.final_res_log2 = int(np.log2(self.resolution))
    self.w_space_dim = w_space_dim
    self.fused_scale = fused_scale
    self.fmaps_base = fmaps_base
    self.fmaps_max = fmaps_max

    self.num_layers = (self.final_res_log2 - self.init_res_log2 + 1) * 2

    # Level of detail (used for progressive training).
    self.lod = nn.Parameter(torch.zeros(()))

    for res_log2 in range(self.init_res_log2, self.final_res_log2 + 1):
      res = 2 ** res_log2
      block_idx = res_log2 - self.init_res_log2

      # First convolution layer for each resolution.
      if res == self.init_res:
        self.add_module(
            f'layer{2 * block_idx}',
            FirstConvBlock(init_resolution=self.init_res,
                           channels=self.get_nf(res),
                           w_space_dim=self.w_space_dim,
                           randomize_noise=randomize_noise))
      else:
        if self.fused_scale == 'auto':
          fused_scale = (res >= _AUTO_FUSED_SCALE_MIN_RES)
        else:
          fused_scale = self.fused_scale
        self.add_module(
            f'layer{2 * block_idx}',
            UpConvBlock(resolution=res,
                        in_channels=self.get_nf(res // 2),
                        out_channels=self.get_nf(res),
                        w_space_dim=self.w_space_dim,
                        randomize_noise=randomize_noise,
                        fused_scale=fused_scale))

      # Second convolution layer for each resolution.
      self.add_module(
          f'layer{2 * block_idx + 1}',
          ConvBlock(resolution=res,
                    in_channels=self.get_nf(res),
                    out_channels=self.get_nf(res),
                    w_space_dim=self.w_space_dim,
                    randomize_noise=randomize_noise))

      # Output convolution layer for each resolution.
      self.add_module(
          f'output{block_idx}',
          LastConvBlock(in_channels=self.get_nf(res),
                        out_channels=image_channels))

    self.upsample = ResolutionScalingLayer()
    self.final_activate = nn.Tanh() if final_tanh else nn.Identity()

  def get_nf(self, res):
    """Gets number of feature maps according to current resolution."""
    return min(self.fmaps_base // res, self.fmaps_max)

  def forward(self, w):
    if w.ndim != 3 or w.shape[1:] != (self.num_layers, self.w_space_dim):
      raise ValueError(f'The input tensor should be with shape [batch_size, '
                       f'num_layers, w_space_dim], where '
                       f'`num_layers` equals to {self.num_layers}, and '
                       f'`w_space_dim` equals to {self.w_space_dim}!\n'
                       f'But {w.shape} is received!')

    lod = self.lod.cpu().tolist()
    for res_log2 in range(self.init_res_log2, self.final_res_log2 + 1):
      if res_log2 + lod <= self.final_res_log2:
        block_idx = res_log2 - self.init_res_log2
        if block_idx == 0:
          x = self.__getattr__(f'layer{2 * block_idx}')(w[:, 2 * block_idx])
        else:
          x = self.__getattr__(f'layer{2 * block_idx}')(x, w[:, 2 * block_idx])
        x = self.__getattr__(f'layer{2 * block_idx + 1}')(
            x, w[:, 2 * block_idx + 1])
        image = self.__getattr__(f'output{block_idx}')(x)
      else:
        image = self.upsample(image)
    image = self.final_activate(image)
    return image


class PixelNormLayer(nn.Module):
  """Implements pixel-wise feature vector normalization layer."""

  def __init__(self, epsilon=1e-8):
    super().__init__()
    self.eps = epsilon

  def forward(self, x):
    return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)


class InstanceNormLayer(nn.Module):
  """Implements instance normalization layer."""

  def __init__(self, epsilon=1e-8):
    super().__init__()
    self.eps = epsilon

  def forward(self, x):
    if x.ndim != 4:
      raise ValueError(f'The input tensor should be with shape [batch_size, '
                       f'channel, height, width], but {x.shape} is received!')
    x = x - torch.mean(x, dim=[2, 3], keepdim=True)
    x = x / torch.sqrt(torch.mean(x ** 2, dim=[2, 3], keepdim=True) + self.eps)
    return x


class ResolutionScalingLayer(nn.Module):
  """Implements the resolution scaling layer.

  Basically, this layer can be used to upsample feature maps from spatial domain
  with nearest neighbor interpolation.
  """

  def __init__(self, scale_factor=2):
    super().__init__()
    self.scale_factor = scale_factor

  def forward(self, x):
    return F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')


class BlurLayer(nn.Module):
  """Implements the blur layer."""

  def __init__(self,
               channels,
               kernel=(1, 2, 1),
               normalize=True,
               flip=False):
    super().__init__()
    kernel = np.array(kernel, dtype=np.float32).reshape(1, -1)
    kernel = kernel.T.dot(kernel)
    if normalize:
      kernel /= np.sum(kernel)
    if flip:
      kernel = kernel[::-1, ::-1]
    kernel = kernel[:, :, np.newaxis, np.newaxis]
    kernel = np.tile(kernel, [1, 1, channels, 1])
    kernel = np.transpose(kernel, [2, 3, 0, 1])
    self.register_buffer('kernel', torch.from_numpy(kernel))
    self.channels = channels

  def forward(self, x):
    return F.conv2d(x, self.kernel, stride=1, padding=1, groups=self.channels)


class NoiseApplyingLayer(nn.Module):
  """Implements the noise applying layer."""

  def __init__(self, resolution, channels, randomize_noise=False):
    super().__init__()
    self.randomize_noise = randomize_noise
    self.res = resolution
    self.register_buffer('noise', torch.randn(1, 1, self.res, self.res))
    self.weight = nn.Parameter(torch.zeros(channels))

  def forward(self, x):
    if x.ndim != 4:
      raise ValueError(f'The input tensor should be with shape [batch_size, '
                       f'channel, height, width], but {x.shape} is received!')
    if self.randomize_noise:
      noise = torch.randn(x.shape[0], 1, self.res, self.res).to(x)
    else:
      noise = self.noise
    return x + noise * self.weight.view(1, -1, 1, 1)


class StyleModulationLayer(nn.Module):
  """Implements the style modulation layer."""

  def __init__(self, channels, w_space_dim=512):
    super().__init__()
    self.channels = channels
    self.w_space_dim = w_space_dim
    self.dense = DenseBlock(in_channels=w_space_dim,
                            out_channels=channels * 2,
                            wscale_gain=1.0,
                            wscale_lr_multiplier=1.0,
                            activation_type='linear')

  def forward(self, x, w):
    if w.ndim != 2 or w.shape[1] != self.w_space_dim:
      raise ValueError(f'The input tensor should be with shape [batch_size, '
                       f'w_space_dim], where `w_space_dim` equals to '
                       f'{self.w_space_dim}!\n'
                       f'But {x.shape} is received!')
    style = self.dense(w)
    style = style.view(-1, 2, self.channels, 1, 1)
    return x * (style[:, 0] + 1) + style[:, 1]


class WScaleLayer(nn.Module):
  """Implements the layer to scale weight variable and add bias.

  NOTE: The weight variable is trained in `nn.Conv2d` layer (or `nn.Linear`
  layer), and only scaled with a constant number, which is not trainable in
  this layer. However, the bias variable is trainable in this layer.
  """

  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               gain=np.sqrt(2.0),
               lr_multiplier=1.0):
    super().__init__()
    fan_in = in_channels * kernel_size * kernel_size
    self.scale = gain / np.sqrt(fan_in) * lr_multiplier
    self.bias = nn.Parameter(torch.zeros(out_channels))
    self.lr_multiplier = lr_multiplier

  def forward(self, x):
    if x.ndim == 4:
      return x * self.scale + self.bias.view(1, -1, 1, 1) * self.lr_multiplier
    if x.ndim == 2:
      return x * self.scale + self.bias.view(1, -1) * self.lr_multiplier
    raise ValueError(f'The input tensor should be with shape [batch_size, '
                     f'channel, height, width], or [batch_size, channel]!\n'
                     f'But {x.shape} is received!')


class EpilogueBlock(nn.Module):
  """Implements the epilogue block of each conv block."""

  def __init__(self,
               resolution,
               channels,
               w_space_dim=512,
               randomize_noise=False,
               normalization_fn='instance'):
    super().__init__()
    self.apply_noise = NoiseApplyingLayer(resolution, channels, randomize_noise)
    self.bias = nn.Parameter(torch.zeros(channels))
    self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    if normalization_fn == 'pixel':
      self.norm = PixelNormLayer()
    elif normalization_fn == 'instance':
      self.norm = InstanceNormLayer()
    else:
      raise NotImplementedError(f'Not implemented normalization function: '
                                f'{normalization_fn}!')
    self.style_mod = StyleModulationLayer(channels, w_space_dim=w_space_dim)

  def forward(self, x, w):
    x = self.apply_noise(x)
    x = x + self.bias.view(1, -1, 1, 1)
    x = self.activate(x)
    x = self.norm(x)
    x = self.style_mod(x, w)
    return x


class FirstConvBlock(nn.Module):
  """Implements the first convolutional block.

  Basically, this block starts from a const input, which is
  `ones(channels, init_resolution, init_resolution)`.
  """

  def __init__(self,
               init_resolution,
               channels,
               w_space_dim=512,
               randomize_noise=False):
    super().__init__()
    self.const = nn.Parameter(
        torch.ones(1, channels, init_resolution, init_resolution))
    self.epilogue = EpilogueBlock(resolution=init_resolution,
                                  channels=channels,
                                  w_space_dim=w_space_dim,
                                  randomize_noise=randomize_noise)

  def forward(self, w):
    x = self.const.repeat(w.shape[0], 1, 1, 1)
    x = self.epilogue(x, w)
    return x


class UpConvBlock(nn.Module):
  """Implements the convolutional block with upsampling.

  Basically, this block is used as the first convolutional block for each
  resolution, which will execute upsampling.
  """

  def __init__(self,
               resolution,
               in_channels,
               out_channels,
               kernel_size=3,
               stride=1,
               padding=1,
               dilation=1,
               add_bias=False,
               fused_scale=False,
               wscale_gain=np.sqrt(2.0),
               wscale_lr_multiplier=1.0,
               w_space_dim=512,
               randomize_noise=False):
    """Initializes the class with block settings.

    Args:
      resolution: Spatial resolution of current layer.
      in_channels: Number of channels of the input tensor fed into this block.
      out_channels: Number of channels (kernels) of the output tensor.
      kernel_size: Size of the convolutional kernel.
      stride: Stride parameter for convolution operation.
      padding: Padding parameter for convolution operation.
      dilation: Dilation rate for convolution operation.
      add_bias: Whether to add bias onto the convolutional result.
      fused_scale: Whether to fuse `upsample` and `conv2d` together, resulting
        in `conv2d_transpose`.
      wscale_gain: The gain factor for `wscale` layer.
      wscale_lr_multiplier: The learning rate multiplier factor for `wscale`
        layer.
      w_space_dim: The dimension of disentangled latent space, w. This is used
        for style modulation.
      randomize_noise: Whether to add random noise.
    """
    super().__init__()

    self.fused_scale = fused_scale

    if self.fused_scale:
      self.weight = nn.Parameter(
          torch.randn(kernel_size, kernel_size, in_channels, out_channels))

    else:
      self.upsample = ResolutionScalingLayer()
      self.conv = nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            dilation=dilation,
                            groups=1,
                            bias=add_bias)

    fan_in = in_channels * kernel_size * kernel_size
    self.scale = wscale_gain / np.sqrt(fan_in) * wscale_lr_multiplier
    self.blur = BlurLayer(channels=out_channels)
    self.epilogue = EpilogueBlock(resolution=resolution,
                                  channels=out_channels,
                                  w_space_dim=w_space_dim,
                                  randomize_noise=randomize_noise)

  def forward(self, x, w):
    if self.fused_scale:
      kernel = self.weight * self.scale
      kernel = F.pad(kernel, (0, 0, 0, 0, 1, 1, 1, 1), 'constant', 0.0)
      kernel = (kernel[1:, 1:] + kernel[:-1, 1:] +
                kernel[1:, :-1] + kernel[:-1, :-1])
      kernel = kernel.permute(2, 3, 0, 1)
      x = F.conv_transpose2d(x, kernel, stride=2, padding=1)
    else:
      x = self.upsample(x)
      x = self.conv(x) * self.scale
    x = self.blur(x)
    x = self.epilogue(x, w)
    return x


class ConvBlock(nn.Module):
  """Implements the normal convolutional block.

  Basically, this block is used as the second convolutional block for each
  resolution.
  """

  def __init__(self,
               resolution,
               in_channels,
               out_channels,
               kernel_size=3,
               stride=1,
               padding=1,
               dilation=1,
               add_bias=False,
               wscale_gain=np.sqrt(2.0),
               wscale_lr_multiplier=1.0,
               w_space_dim=512,
               randomize_noise=False):
    """Initializes the class with block settings.

    Args:
      resolution: Spatial resolution of current layer.
      in_channels: Number of channels of the input tensor fed into this block.
      out_channels: Number of channels (kernels) of the output tensor.
      kernel_size: Size of the convolutional kernel.
      stride: Stride parameter for convolution operation.
      padding: Padding parameter for convolution operation.
      dilation: Dilation rate for convolution operation.
      add_bias: Whether to add bias onto the convolutional result.
      wscale_gain: The gain factor for `wscale` layer.
      wscale_lr_multiplier: The learning rate multiplier factor for `wscale`
        layer.
      w_space_dim: The dimension of disentangled latent space, w. This is used
        for style modulation.
      randomize_noise: Whether to add random noise.
    """
    super().__init__()

    self.conv = nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding,
                          dilation=dilation,
                          groups=1,
                          bias=add_bias)
    fan_in = in_channels * kernel_size * kernel_size
    self.scale = wscale_gain / np.sqrt(fan_in) * wscale_lr_multiplier
    self.epilogue = EpilogueBlock(resolution=resolution,
                                  channels=out_channels,
                                  w_space_dim=w_space_dim,
                                  randomize_noise=randomize_noise)

  def forward(self, x, w):
    x = self.conv(x) * self.scale
    x = self.epilogue(x, w)
    return x


class LastConvBlock(nn.Module):
  """Implements the last convolutional block.

  Basically, this block converts the final feature map to RGB image.
  """

  def __init__(self, in_channels, out_channels=3):
    super().__init__()
    self.conv = nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=1,
                          bias=False)
    self.scale = 1 / np.sqrt(in_channels)
    self.bias = nn.Parameter(torch.zeros(out_channels))

  def forward(self, x):
    x = self.conv(x) * self.scale
    x = x + self.bias.view(1, -1, 1, 1)
    return x


class DenseBlock(nn.Module):
  """Implements the dense block.

  Basically, this block executes fully-connected layer, weight-scale layer,
  and activation layer in sequence.
  """

  def __init__(self,
               in_channels,
               out_channels,
               add_bias=False,
               wscale_gain=np.sqrt(2.0),
               wscale_lr_multiplier=0.01,
               activation_type='lrelu'):
    """Initializes the class with block settings.

    Args:
      in_channels: Number of channels of the input tensor fed into this block.
      out_channels: Number of channels of the output tensor.
      add_bias: Whether to add bias onto the fully-connected result.
      wscale_gain: The gain factor for `wscale` layer.
      wscale_lr_multiplier: The learning rate multiplier factor for `wscale`
        layer.
      activation_type: Type of activation. Support `linear` and `lrelu`.

    Raises:
      NotImplementedError: If the input `activation_type` is not supported.
    """
    super().__init__()
    self.fc = nn.Linear(in_features=in_channels,
                        out_features=out_channels,
                        bias=add_bias)
    self.wscale = WScaleLayer(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=1,
                              gain=wscale_gain,
                              lr_multiplier=wscale_lr_multiplier)
    if activation_type == 'linear':
      self.activate = nn.Identity()
    elif activation_type == 'lrelu':
      self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    else:
      raise NotImplementedError(f'Not implemented activation function: '
                                f'{activation_type}!')

  def forward(self, x):
    if x.ndim != 2:
      x = x.view(x.shape[0], -1)
    x = self.fc(x)
    x = self.wscale(x)
    x = self.activate(x)
    return x
