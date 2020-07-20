# In-Domain GAN Inversion for Real Image Editing

![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)
![PyTorch 1.2.0](https://img.shields.io/badge/pytorch-1.2.0-green.svg?style=plastic)

![image](./teaser.jpg)

**Figure:** *Real image editing using the proposed In-Domain GAN inversion with a fixed GAN generator.*

> **In-Domain GAN Inversion for Real Image Editing** <br>
> Jiapeng Zhu*, Yujun Shen*, Deli Zhao, Bolei Zhou <br>
> *European Conference on Computer Vision (ECCV) 2020*

[[Paper](https://arxiv.org/pdf/2004.00049.pdf)]
[[Project Page](https://genforce.github.io/idinvert/)]
[[Demo](https://www.youtube.com/watch?v=3v6NHrhuyFY)]

**NOTE:** This repository is a simple PyTorch version of [this repo](https://github.com/genforce/idinvert), and ONLY supports inference.

## Editing Tasks

### Pre-trained Models

Please download the pre-trained models from the following links and save them to `models/pretrain/`

| Description | Generator | Encoder |
| :---------- | :-------- | :------ |
| Model trained on [FFHQ](https://github.com/NVlabs/ffhq-dataset) dataset. | [face_256x256_generator](https://drive.google.com/file/d/1SjWD4slw612z2cXa3-n38JwKZXqDUerG/view?usp=sharing)    | [face_256x256_encoder](https://drive.google.com/file/d/1gij7xy05crnyA-tUTQ2F3yYlAlu6p9bO/view?usp=sharing)
| Model trained on [LSUN Tower](https://github.com/fyu/lsun) dataset.      | [tower_256x256_generator](https://drive.google.com/file/d/1lI_OA_aN4-O3mXEPQ1Nv-6tdg_3UWcyN/view?usp=sharing)   | [tower_256x256_encoder](https://drive.google.com/file/d/1Pzkgdi3xctdsCZa9lcb7dziA_UMIswyS/view?usp=sharing)
| Model trained on [LSUN Bedroom](https://github.com/fyu/lsun) dataset.    | [bedroom_256x256_generator](https://drive.google.com/file/d/1ka583QwvMOtcFZJcu29ee8ykZdyOCcMS/view?usp=sharing) | [bedroom_256x256_encoder](https://drive.google.com/file/d/1ebuiaQ7xI99a6ZrHbxzGApEFCu0h0X2s/view?usp=sharing)
| [Perceptual Model](https://drive.google.com/file/d/1qQ-r7MYZ8ZcjQQFe17eQfJbOAuE3eS0y/view?usp=sharing)

### Inversion

```bash
MODEL_NAME='styleganinv_ffhq256'
IMAGE_LIST='examples/test.list'
python invert.py $MODEL_NAME $IMAGE_LIST
```

**NOTE:** We find that 100 iterations are good enough for inverting an image, which takes about 8s (on P40). But users can always use more iterations (much slower) for a more precise reconstruction.

### Semantic Diffusion

```bash
MODEL_NAME='styleganinv_ffhq256'
TARGET_LIST='examples/target.list'
CONTEXT_LIST='examples/context.list'
python diffuse.py $MODEL_NAME $TARGET_LIST $CONTEXT_LIST
```

NOTE: The diffusion process is highly similar to image inversion. The main difference is that only the target patch is used to compute loss for **masked** optimization.

### Interpolation

```bash
SRC_DIR='results/inversion/test'
DST_DIR='results/inversion/test'
python interpolate.py $MODEL_NAME $SRC_DIR $DST_DIR
```

### Manipulation

```bash
IMAGE_DIR='results/inversion/test'
BOUNDARY='boundaries/expression.npy'
python manipulate.py $MODEL_NAME $IMAGE_DIR $BOUNDARY
```

**NOTE:** Boundaries are obtained using [InterFaceGAN](https://github.com/genforce/interfacegan).

### Style Mixing

```bash
STYLE_DIR='results/inversion/test'
CONTENT_DIR='results/inversion/test'
python mix_style.py $MODEL_NAME $STYLE_DIR $CONTENT_DIR
```

## BibTeX

```bibtex
@inproceedings{zhu2020indomain,
  title     = {In-domain GAN Inversion for Real Image Editing},
  author    = {Zhu, Jiapeng and Shen, Yujun and Zhao, Deli and Zhou, Bolei},
  booktitle = {Proceedings of European Conference on Computer Vision (ECCV)},
  year      = {2020}
}
```
