# python 3.6
"""Manipulates real images with In-domain GAN Inversion.

The real images should be first inverted to latent codes with `invert.py`. After
that, this script can be used for image manipulation with a given boundary.
"""

import os.path
import argparse
import numpy as np
from tqdm import tqdm

from models.helper import build_generator
from utils.logger import setup_logger
from utils.editor import manipulate
from utils.visualizer import load_image
from utils.visualizer import HtmlPageVisualizer


def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('model_name', type=str, help='Name of the GAN model.')
  parser.add_argument('image_dir', type=str,
                      help='Image directory, which includes original images, '
                           'inverted codes, and image list.')
  parser.add_argument('boundary_path', type=str,
                      help='Path to the boundary for semantic manipulation.')
  parser.add_argument('-o', '--output_dir', type=str, default='',
                      help='Directory to save the results. If not specified, '
                           '`./results/manipulation` will be used by default.')
  parser.add_argument('--step', type=int, default=7,
                      help='Number of manipulation steps. (default: 7)')
  parser.add_argument('--start_distance', type=float, default=-3.0,
                      help='Start distance for manipulation. (default: -3.0)')
  parser.add_argument('--end_distance', type=float, default=3.0,
                      help='End distance for manipulation. (default: 3.0)')
  parser.add_argument('--manipulate_layers', type=str, default='',
                      help='Indices of the layers to perform manipulation. '
                           'If not specified, all layers will be manipulated. '
                           'More than one layers should be separated by `,`. '
                           '(default: None)')
  parser.add_argument('--viz_size', type=int, default=256,
                      help='Image size for visualization. (default: 256)')
  parser.add_argument('--gpu_id', type=str, default='0',
                      help='Which GPU(s) to use. (default: `0`)')
  return parser.parse_args()


def main():
  """Main function."""
  args = parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
  image_dir = args.image_dir
  image_dir_name = os.path.basename(image_dir.rstrip('/'))
  assert os.path.exists(image_dir)
  assert os.path.exists(f'{image_dir}/image_list.txt')
  assert os.path.exists(f'{image_dir}/inverted_codes.npy')
  boundary_path = args.boundary_path
  assert os.path.exists(boundary_path)
  boundary_name = os.path.splitext(os.path.basename(boundary_path))[0]
  output_dir = args.output_dir or 'results/manipulation'
  job_name = f'{boundary_name.upper()}_{image_dir_name}'
  logger = setup_logger(output_dir, f'{job_name}.log', f'{job_name}_logger')

  # Load model.
  logger.info(f'Loading generator.')
  generator = build_generator(args.model_name)

  # Load image, codes, and boundary.
  logger.info(f'Loading images and corresponding inverted latent codes.')
  image_list = []
  with open(f'{image_dir}/image_list.txt', 'r') as f:
    for line in f:
      name = os.path.splitext(os.path.basename(line.strip()))[0]
      assert os.path.exists(f'{image_dir}/{name}_ori.png')
      assert os.path.exists(f'{image_dir}/{name}_inv.png')
      image_list.append(name)
  latent_codes = np.load(f'{image_dir}/inverted_codes.npy')
  assert latent_codes.shape[0] == len(image_list)
  num_images = latent_codes.shape[0]
  logger.info(f'Loading boundary.')
  boundary_file = np.load(boundary_path, allow_pickle=True)[()]
  if isinstance(boundary_file, dict):
    boundary = boundary_file['boundary']
    manipulate_layers = boundary_file['meta_data']['manipulate_layers']
  else:
    boundary = boundary_file
    manipulate_layers = args.manipulate_layers
  if manipulate_layers:
    logger.info(f'  Manipulating on layers `{manipulate_layers}`.')
  else:
    logger.info(f'  Manipulating on ALL layers.')

  # Manipulate images.
  logger.info(f'Start manipulation.')
  step = args.step
  viz_size = None if args.viz_size == 0 else args.viz_size
  visualizer = HtmlPageVisualizer(
      num_rows=num_images, num_cols=step + 3, viz_size=viz_size)
  visualizer.set_headers(
      ['Name', 'Origin', 'Inverted'] +
      [f'Step {i:02d}' for i in range(1, step + 1)]
  )
  for img_idx, img_name in enumerate(image_list):
    ori_image = load_image(f'{image_dir}/{img_name}_ori.png')
    inv_image = load_image(f'{image_dir}/{img_name}_inv.png')
    visualizer.set_cell(img_idx, 0, text=img_name)
    visualizer.set_cell(img_idx, 1, image=ori_image)
    visualizer.set_cell(img_idx, 2, image=inv_image)

  codes = manipulate(latent_codes=latent_codes,
                     boundary=boundary,
                     start_distance=args.start_distance,
                     end_distance=args.end_distance,
                     step=step,
                     layerwise_manipulation=True,
                     num_layers=generator.num_layers,
                     manipulate_layers=manipulate_layers,
                     is_code_layerwise=True,
                     is_boundary_layerwise=True)

  for img_idx in tqdm(range(num_images), leave=False):
    output_images = generator.easy_synthesize(
        codes[img_idx], latent_space_type='wp')['image']
    for s, output_image in enumerate(output_images):
      visualizer.set_cell(img_idx, s + 3, image=output_image)

  # Save results.
  visualizer.save(f'{output_dir}/{job_name}.html')


if __name__ == '__main__':
  main()
