# Unmyelinated fiber segmentation

## Introduction

We released a training and a test script for the automatic segmentation of
unmyelinated fibers in Transmission Electron Microscope (TEM) images.

Most of the literature on axon segmentation is focused on myelinated fibers,
for example [DeepAxonSeg](https://www.nature.com/articles/s41598-018-22181-4),
but unmyelinated fibers (UMFs) are far more common in TEM images. The
segmentation of UMFs is also more challenging, because they may have a wide
range of sizes and shapes, they are often clumped together, and other cell
elements (such as vesicles) may mimic the appearance of UMFs.

Our approach is based on a standard U-Net architecture, which is a convolutional
network with skip connections. We use four downsampling and upsampling stages,
with batch-normalization in each layer and dropout in the middle layers
(bottleneck). To achieve good performance, we use a border class and class
weights based on inverse class frequencies, to mimic a border-aware loss
function similar to the one proposed in the
[original U-Net paper](https://arxiv.org/abs/1505.04597). We also designed a
sampling strategy for the tiles fed to the model at training time, to boost the
accuracy on elongated and large fibers.

## Usage

`umf_train.m` is the training script. It takes a directory with a set of images
and annotated segmentation maps, divide them into tiles, and train a model.
The dataset directory is specified by the `datapath` variable at the start of
the script and the working directory, where the tiles and the trained model are
stored, is specified by the `out_dir` variable.

The training script expects the annotations images to be the outlines, in white
and one-pixel thick, of the unmyelinated fibers, saved as a TIFF image.
Available annotation tools, such as
[Neurolucida](https://www.mbfbioscience.com/neurolucida), can export the
annotations in such a format.

`umf_test.m` is the test script. It runs a trained model on a list of images
dividing each image into tiles spaced by 1/8 of the tile size.
The output for each image is a segmentation map both as a binary mask and
overlaid on the original image. Moreover, for each image, a set of metrics is
reported: the Panoptic Quality (PQ), Segmentation Quality (SQ), and Recognition
Quality (RQ) measures for instance segmentation
(see [Panoptic Segmentation](https://arxiv.org/abs/1801.00868)); the Dice
coefficient and the Jaccard index for pixel-level segmentation.

`panoptic_quality.m` is a standalone function that computes the Panoptic
Quality (PQ), the Segmentation Quality (SQ), and the Recognition Quality (RQ)
measures given the manual segmentation and the segmentation produced by the
model.

### Running example

We released a trained model and a test image with annotated unmyelinated
regions in a separate image. To run the test script and get the segmentation
metric on the test image, run in Matlab:

```bash
umf_test
```

You should get the following results:

| **Image name**      | **PQ** | **SQ** | **RQ** | **Dice** | **Jaccard** |
| ----                | ----    | ----  | ----  | ----    | ----       |
| `sub-131_sam-8_Image_em` | 0.669   | 0.8   | 0.836 | 0.875  | 0.777    |

![Segmentation result](result.png)

## License

The code is released under the Apache-2 License (see `LICENSE.txt` for
details).

## Citation

If you find this repository useful in your research, please cite:

```bibtex
@article{Plebani2021crism,
  title={High-throughput segmentation of unmyelinated axons by deep learning},
  author={Emanuele Plebani and Natalia Biscola and Leif Havton and Bartek Rajwa
    and Abida SanjanaShemonti and Deborah Jaffey and Terry Powley and
    Janet R. Keast and Kun-Han Lu and and MuratDundar},
  booktitle={to be submitted to Nature Scientific Reports},
  year={2021}
}
```
