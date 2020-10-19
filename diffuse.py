# python 3.6
"""diffuses target images to context images with In-domain GAN Inversion.

Basically, this script first copies the central region from the target image to
the context image, and then performs in-domain GAN inversion on the stitched
image. Different from `intert.py`, masked reconstruction loss is used in the
optimization stage.

NOTE: This script will diffuse every image from `target_image_list` to every
image from `context_image_list`.
"""

import os
import argparse
from tqdm import tqdm
import numpy as np

from utils.inverter import StyleGANInverter
from utils.logger import setup_logger
from utils.visualizer import HtmlPageVisualizer
from utils.visualizer import load_image, resize_image


def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('model_name', type=str, help='Name of the GAN model.')
  parser.add_argument('target_list', type=str,
                      help='List of target images to diffuse from.')
  parser.add_argument('context_list', type=str,
                      help='List of context images to diffuse to.')
  parser.add_argument('-o', '--output_dir', type=str, default='',
                      help='Directory to save the results. If not specified, '
                           '`./results/diffusion` will be used by default.')
  parser.add_argument('-s', '--crop_size', type=int, default=110,
                      help='Crop size. (default: 110)')
  parser.add_argument('-x', '--center_x', type=int, default=125,
                      help='X-coordinate (column) of the center of the cropped '
                           'patch. This field should be adjusted according to '
                           'dataset and image size. (default: 125)')
  parser.add_argument('-y', '--center_y', type=int, default=145,
                      help='Y-coordinate (row) of the center of the cropped '
                           'patch. This field should be adjusted according to '
                           'dataset and image size. (default: 145)')
  parser.add_argument('--learning_rate', type=float, default=0.01,
                      help='Learning rate for optimization. (default: 0.01)')
  parser.add_argument('--num_iterations', type=int, default=100,
                      help='Number of optimization iterations. (default: 100)')
  parser.add_argument('--num_results', type=int, default=5,
                      help='Number of intermediate optimization results to '
                           'save for each sample. (default: 5)')
  parser.add_argument('--loss_weight_feat', type=float, default=5e-5,
                      help='The perceptual loss scale for optimization. '
                           '(default: 5e-5)')
  parser.add_argument('--viz_size', type=int, default=256,
                      help='Image size for visualization. (default: 256)')
  parser.add_argument('--gpu_id', type=str, default='0',
                      help='Which GPU(s) to use. (default: `0`)')
  parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size. (default: 4)')
  return parser.parse_args()


def main():
  """Main function."""
  args = parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
  assert os.path.exists(args.target_list)
  target_list_name = os.path.splitext(os.path.basename(args.target_list))[0]
  assert os.path.exists(args.context_list)
  context_list_name = os.path.splitext(os.path.basename(args.context_list))[0]
  output_dir = args.output_dir or f'results/diffusion'
  job_name = f'{target_list_name}_TO_{context_list_name}'
  logger = setup_logger(output_dir, f'{job_name}.log', f'{job_name}_logger')

  logger.info(f'Loading model.')
  inverter = StyleGANInverter(
      args.model_name,
      learning_rate=args.learning_rate,
      iteration=args.num_iterations,
      reconstruction_loss_weight=1.0,
      perceptual_loss_weight=args.loss_weight_feat,
      regularization_loss_weight=0.0,
      logger=logger)
  image_size = inverter.G.resolution

  # Load image list.
  logger.info(f'Loading target images and context images.')
  target_list = []
  with open(args.target_list, 'r') as f:
    for line in f:
      target_list.append(line.strip())
  num_targets = len(target_list)
  context_list = []
  with open(args.context_list, 'r') as f:
    for line in f:
      context_list.append(line.strip())
  num_contexts = len(context_list)
  num_pairs = num_targets * num_contexts

  # Initialize visualizer.
  save_interval = args.num_iterations // args.num_results
  headers = ['Target Image', 'Context Image', 'Stitched Image',
             'Encoder Output']
  for step in range(1, args.num_iterations + 1):
    if step == args.num_iterations or step % save_interval == 0:
      headers.append(f'Step {step:06d}')
  viz_size = None if args.viz_size == 0 else args.viz_size
  visualizer = HtmlPageVisualizer(
      num_rows=num_pairs, num_cols=len(headers), viz_size=viz_size)
  visualizer.set_headers(headers)

  # Diffuse images.
  logger.info(f'Start diffusion.')
  latent_codes = []
  for target_idx in tqdm(range(num_targets), desc='Target ID', leave=False):
    # Load target.
    target_image = resize_image(load_image(target_list[target_idx]),
                                (image_size, image_size))
    visualizer.set_cell(target_idx * num_contexts, 0, image=target_image)
    for context_batch_idx in tqdm(range(0, num_contexts, args.batch_size),
                            desc='Context ID', leave=False):
      context_images = []
      for it in range(args.batch_size):
        context_idx = context_batch_idx + it
        if context_idx >= num_contexts:
          continue
        row_idx = target_idx * num_contexts + context_idx
        context_image = resize_image(load_image(context_list[context_idx]),
                                     (image_size, image_size))
        context_images.append(context_image)
        visualizer.set_cell(row_idx, 1, image=context_image)
      code, viz_results = inverter.easy_diffuse(target=target_image,
                                                context=np.asarray(context_images),
                                                center_x=args.center_x,
                                                center_y=args.center_y,
                                                crop_x=args.crop_size,
                                                crop_y=args.crop_size,
                                                num_viz=args.num_results)
      for key, values in viz_results.items():
        context_idx = context_batch_idx + key
        row_idx = target_idx * num_contexts + context_idx
        for viz_idx, viz_img in enumerate(values):
          visualizer.set_cell(row_idx, viz_idx + 2, image=viz_img)
      latent_codes.append(code)

  # Save results.
  os.system(f'cp {args.target_list} {output_dir}/target_list.txt')
  os.system(f'cp {args.context_list} {output_dir}/context_list.txt')
  np.save(f'{output_dir}/{job_name}_inverted_codes.npy',
          np.concatenate(latent_codes, axis=0))
  visualizer.save(f'{output_dir}/{job_name}.html')


if __name__ == '__main__':
  main()
