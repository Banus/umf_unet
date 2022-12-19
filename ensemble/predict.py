# encoding:utf-8
"""Run prediction on one or more images."""
import argparse
import os
import time

import torch
from PIL import Image
import utils as ut


def main():
    """Script entry point."""
    parser = argparse.ArgumentParser(
        description='Run prediction on one or more images.')
    parser.add_argument('image', nargs='+', help='Image file(s) to process.')
    # Supported models:
    # - ens: ensemble of 5 models at different resolutions, with histogram
    #        equalization for the first 2 (default)
    # - ens_he: ensemble of 5 models, with histogram equalization for all
    # - single: single model full resolution model, with histogram equalization
    parser.add_argument('--model', default='ens',
                        choices=('ens', 'ens_he', 'single'),
                        help='Model to use.')
    parser.add_argument('--scale', default=1.0, type=float,
                        help='Scale factor for the image.')
    parser.add_argument('--raw', action='store_true', help='return raw '
                        'predictions (before post-processing).')
    parser.add_argument('--out', default='predictions/',
                        help='Output directory.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    equalized = args.model == 'ens_he'
    models = ut.load_ensemble(equalized=equalized, device=device)

    for image_file in args.image:
        if not os.path.isfile(image_file):
            print(f'File {image_file} not found.')
            continue

        print(f'Processing {image_file}...', end='', flush=True)
        image = ut.load_image(image_file, scale=args.scale)

        start = time.time()
        pred = ut.predict_with_ensemble(models, image, equalized=equalized,
                                        use_ensemble=args.model != 'single')
        print("time: {:.3f} s".format(time.time()-start))

        if not args.raw:
            print(" cleaning...", end="", flush=True)
            start = time.time()
            pred = ut.clean_predictions(pred)
            print("time: {:.3f} s".format(time.time()-start))

        composite = ut.create_overlay(image, pred)

        # save results as PNGs
        name, _ = os.path.splitext(os.path.basename(image_file))
        os.makedirs(args.out, exist_ok=True)
        name = os.path.join(args.out, name)

        Image.fromarray(pred).save(f"{name}_pred.png")
        Image.fromarray(composite).save(f"{name}_overlay.png")


if __name__ == '__main__':
    main()
