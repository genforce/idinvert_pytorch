# python 3.6
"""Inverts given images to latent codes with In-Domain GAN Inversion.

Basically, for a particular image (real or synthesized), this script first
employs the domain-guided encoder to produce a initial point in the latent
space and then performs domain-regularized optimization to refine the latent
code.
"""

import os
import argparse
from tqdm import tqdm
import numpy as np

from utils.inverter import StyleGANInverter
from utils.logger import setup_logger
from utils.visualizer import HtmlPageVisualizer
from utils.visualizer import save_image, load_image, resize_image


def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_name', type=str, help='Name of the GAN model.')
  parser.add_argument('--image_list', type=str, default = '',
                      help='List of images to invert.')

  parser.add_argument('--test_dir', type=str, default = '',
                      help='directory of images to invert.')
  parser.add_argument('-o', '--output_dir', type=str, default='',
                      help='Directory to save the results. If not specified, '
                           '`./results/inversion/${IMAGE_LIST}` '
                           'will be used by default.')
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

  parser.add_argument('--loss_weight_ssim', type=float, default=1.0,
                      help='The perceptual loss scale for optimization. '
                           '(default: 1)')
  
  parser.add_argument('--loss_weight_enc', type=float, default=2.0,
                      help='The encoder loss scale for optimization.'
                           '(default: 2.0)')
  parser.add_argument('--viz_size', type=int, default=256,
                      help='Image size for visualization. (default: 256)')
  parser.add_argument('--gpu_id', type=str, default='0',
                      help='Which GPU(s) to use. (default: `0`)')
  return parser.parse_args()


def main():
  """Main function."""
  args = parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
  if args.image_list != '' and args.test_dir == '':
    assert os.path.exists(args.image_list)
    image_list_name = os.path.splitext(os.path.basename(args.image_list))[0]
  elif args.test_dir != '' and args.image_list == '' :
    assert os.path.exists(args.test_dir)
    image_list_name = os.path.splitext(os.path.basename(args.test_dir))[0]
  else:
    raise Exception("Use either --image_list or --test_dir. Using both arguments at the same time not supported.") 


  MODEL_DIR = os.path.join('models', 'pretrain')
  os.makedirs(MODEL_DIR, exist_ok=True)
  if(all(x not in os.listdir(MODEL_DIR) for x in  ["styleganinv_ffhq256_encoder.pth" , "styleganinv_ffhq256_generator.pth" , "vgg16.pth"])):
    raise Exception("styleganinv_ffhq256_encoder.pth , styleganinv_ffhq256_generator.pth and vgg16.pth missing")

  output_dir = args.output_dir or f'results/inversion/{image_list_name}'
  if not os.path.exists(output_dir):
        os.makedirs(output_dir)
  logger = setup_logger(output_dir, 'inversion.log', 'inversion_logger')

  logger.info(f'Loading model.')
  inverter = StyleGANInverter(
      args.model_name,
      learning_rate=args.learning_rate,
      iteration=args.num_iterations,
      reconstruction_loss_weight=1.0,
      perceptual_loss_weight=args.loss_weight_feat,
      regularization_loss_weight=args.loss_weight_enc,
      loss_weight_ssim = args.loss_weight_ssim,
      logger=logger)
  image_size = inverter.G.resolution

  # Load image list.
  logger.info(f'Loading image list.')
  image_list = []
  if args.image_list !='':

    with open(args.image_list, 'r') as f:
      for line in f:
        image_list.append(line.strip())

  if args.test_dir !='':
    for root, dirs, files in os.walk(args.test_dir):
      for file in files: 
        image_list.append(file)


  #print(len(image_list))
  logger.info(f'loaded {len(image_list)} images')

  # Initialize visualizer.
  save_interval = args.num_iterations // args.num_results
  headers = ['Name', 'Original Image', 'Encoder Output']
  for step in range(1, args.num_iterations + 1):
    if step == args.num_iterations or step % save_interval == 0:
      headers.append(f'Step {step:06d}')
  viz_size = None if args.viz_size == 0 else args.viz_size
  visualizer = HtmlPageVisualizer(
      num_rows=len(image_list), num_cols=len(headers), viz_size=viz_size)
  visualizer.set_headers(headers)

  # Invert images.
  logger.info(f'Start inversion.')
  latent_codes = []
  for img_idx in tqdm(range(len(image_list)), leave=False):
    if args.image_list !='':
      image_path = image_list[img_idx]
      image_name = os.path.splitext(os.path.basename(image_path))[0]
    elif args.test_dir !='':
      image_path = os.path.join( args.test_dir, image_list[img_idx])
      image_name = os.path.splitext(os.path.basename(image_list[img_idx]))[0]

    image = resize_image(load_image(image_path), (image_size, image_size))
    code, viz_results , ssim_loss = inverter.easy_invert(np.array(image), num_viz=args.num_results)
    latent_codes.append(code)
    save_image(f'{output_dir}/{image_name}_ori.png', image)
    save_image(f'{output_dir}/{image_name}_enc.png', viz_results[1])
    save_image(f'{output_dir}/{image_name}_inv.png', viz_results[-1])
    visualizer.set_cell(img_idx, 0, text=image_name)
    visualizer.set_cell(img_idx, 1, image=image)
    for viz_idx, viz_img in enumerate(viz_results[1:]):
      visualizer.set_cell(img_idx, viz_idx + 2, image=viz_img)


  # Save results.
  os.system(f'cp {args.image_list} {output_dir}/image_list.txt')
  np.save(f'{output_dir}/inverted_codes.npy',
          np.concatenate(latent_codes, axis=0))
  visualizer.save(f'{output_dir}/inversion.html')


if __name__ == '__main__':
  main()
