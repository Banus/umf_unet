# encoding:utf-8
"""Model loading utilities."""
import numpy as np
import scipy.ndimage as ndi
from PIL import Image
import tifffile
import skimage.morphology as skm
from skimage.measure import regionprops

import torch
from torch import nn
import fastai.vision.all as fva
from models import CPnet

Image.MAX_IMAGE_PIXELS = None  # avoid DecompressionBombError


def _load_models_with_border_classes(device=None):
    """Load high resolution models with border classes."""
    codes = ["BG", "BU", "UF", "SC", "MF", "BS", "BM"]
    n_cls = len(codes)
    pixel_mean = 127.56555 / 255

    # dummy dataset to create the learner, with the correct input
    #  pre-processing
    umf_dataset = fva.DataBlock(
        blocks=(fva.ImageBlock(cls=fva.PILImageBW), fva.MaskBlock(codes)),
        get_items=lambda x: [""],
        get_x=lambda x: np.zeros((512, 512), dtype=np.uint8),
        get_y=lambda x: np.zeros((512, 512), dtype=np.uint8),
        batch_tfms=[fva.Normalize.from_stats(mean=[pixel_mean], std=[1.0])],)
    dls = umf_dataset.dataloaders(".", bs=1)

    def get_learner(name):
        """Load a model from a file."""
        model = CPnet([1, 32, 64, 128, 256], n_cls)
        learn = fva.Learner(dls, model, metrics=fva.foreground_acc,
                            opt_func=fva.ranger,
                            loss_func=nn.CrossEntropyLoss())
        learn.load(name, with_opt=False, device=device)
        learn.dls.to(device)
        return learn

    # model files are 'models/<model name>.pth' w.r.t. the current directory
    return [get_learner('ens_s1'), get_learner('ens_s1_4')]


def _load_soft_models(equalized=True, device=None):
    """Load low-resolution models with soft class predictions and no border."""
    # dummy dataset as above, but now with 3 classes (unmyelinated, Schwann,
    #  myelinated) as RGB values in a 3-channel image predicted by the model
    #  (instead of a 1-channel image with class indices); no border class
    normalize_tfms = [fva.Normalize.from_stats(mean=[0.5], std=[1.0])]
    dataset_lr = fva.DataBlock(
        blocks=(fva.ImageBlock(cls=fva.PILImageBW),
                fva.ImageBlock(cls=fva.PILImage)),
        get_items=lambda x: [""],
        get_x=lambda x: np.zeros((512, 512), dtype=np.uint8),
        get_y=lambda x: np.zeros((512, 512), dtype=np.uint8),
        batch_tfms=normalize_tfms if equalized else None)
    dls_lr = dataset_lr.dataloaders(".", bs=8)

    def mse_loss(y_pred, y_true):
        return torch.mean((y_pred - y_true[:, :4, :, :]) ** 2)

    def error(y_pred, y_true):
        return fva.rmse(y_pred[:, :4, :, :], y_true)

    def get_low_res(name):
        model = CPnet([1, 32, 64, 128, 256], 4)
        learn = fva.Learner(dls_lr, model, metrics=error, opt_func=fva.ranger,
                            loss_func=mse_loss)
        learn.load(name, with_opt=False, device=device)
        learn.dls.to(device)
        return learn

    # low-resolution models
    if equalized:
        return [get_low_res('ens_he_s2'), get_low_res('ens_he_s2_8'),
                get_low_res('ens_he_s4')]
    return [get_low_res('ens_s2'), get_low_res('ens_s2_8'),
            get_low_res('ens_s4')]


def load_ensemble(equalized=True, device=None):
    """Load ensemble of models."""
    models = _load_models_with_border_classes(device)
    models.extend(_load_soft_models(equalized, device))
    return models


#
# predict with ensemble
#

def _load_tiff(fname, gray=False):
    """Load the first image in a TIFF file."""
    with tifffile.TiffFile(fname) as tif:
        imgs = [page.asarray() for page in tif.pages]

    img = imgs[0]
    if gray and len(img.shape) > 2:
        img = img[..., :3] @ [0.299, 0.587, 0.114]   # to grayscale

    return img


def _load_image_file(file):
    """Load an image from a file."""
    # load tiff files with tifffile because PIL may be buggy
    if file.endswith('.tiff') or file.endswith('.tif'):
        return _load_tiff(file, gray=True)

    img = np.asarray(Image.open(file))
    # convert to grayscale if necessary
    return img[..., :3] @ [0.299, 0.587, 0.114] if len(img.shape) > 2 else img


def load_image(fname, scale=1.0):
    """Load an image from a file."""
    image = _load_image_file(fname).astype(np.uint8)

    # scale image (it may improve the results)
    if scale != 1.0:
        image = Image.fromarray(image).resize(
            (int(image.shape[1]*scale), int(image.shape[0]*scale)))
        image = np.array(image)

    return image


def _resize_multichannel(pred, shape):
    """Resize a multi-channel image (to resize predictions)."""
    n_chan, shape = pred.shape[-1], shape[::-1]
    return np.stack([np.asarray(Image.fromarray(pred[..., i]).resize(shape))
                     for i in range(n_chan)], axis=-1)


def _histeq_matlab(img, nbins=64):
    """Histogram equalization using Matlab's algorithm."""
    # get image histogram
    imhist, _ = np.histogram(img.flatten(), 256)
    cum = imhist.cumsum()  # cumulative histogram

    cumd = np.full(nbins, img.size/nbins).cumsum()  # cumulative target
    tol = np.min((np.concatenate((imhist[:-1], [0])),
                  np.concatenate(([0], imhist[1:]))), axis=0)/2.0
    err = cumd[:, None] - cum[None, :] + tol[None, :]

    # set errors beyond threshold to max
    large = np.nonzero(err < -img.size*np.sqrt(np.finfo(float).eps))
    if large[0].size > 0:
        err[large] = img.size

    # find optimal bin transform
    tr_ = np.round(255.0 * np.argmin(err, axis=0)/(nbins-1)).astype(np.uint8)

    return tr_[img.flatten()].reshape(img.shape)


def _tile(shape, size, overlap):
    """Find tile locations."""
    height, width = shape

    n_x = np.ceil((1.0+2*overlap)*width/size).astype(np.int)
    n_y = np.ceil((1.0+2*overlap)*height/size).astype(np.int)
    x_tiles = np.linspace(0, max(width-size, 0), n_x).astype(np.int)
    y_tiles = np.linspace(0, max(height-size, 0), n_y).astype(np.int)

    return x_tiles, y_tiles


def _mask(size, sigma=7.5, pad=20):
    """Apply mask on the predictions."""
    x_m = np.arange(size)
    x_m = np.abs(x_m - np.mean(x_m))
    mask = 1.0/(1.0+np.exp((x_m-(size/2.0-pad))/sigma))
    return np.outer(mask, mask)


def predict_tiled(learner, img, size=512, overlap=0.1, equalize=True):
    """Run predictions on tiles and assemble the result."""
    _eq = _histeq_matlab if equalize else lambda x: x
    n_classes = learner.model.nout
    height, width = img.shape[:2]
    small = min(height, width)

    x_tiles, y_tiles = _tile((height, width), size, overlap)
    mask = _mask(size)

    if small < size:
        padx = max(size-width, 0)//2+1
        pady = max(size-height, 0)//2+1
        img = np.pad(img, ((pady, pady), (padx, padx)), 'constant')
        height, width = img.shape[:2]
    pred = np.zeros((height, width, n_classes), dtype=np.float32)
    cnt = np.zeros((height, width), dtype=np.float32)

    for yy_ in y_tiles:
        for xx_ in x_tiles:
            tile = _eq(img[yy_:yy_+size, xx_:xx_+size])

            if hasattr(learner, 'no_bar'):  # suppress progress bar
                with learner.no_bar(), learner.no_logging():
                    _, _, probs = learner.predict(tile)
            else:
                _, _, probs = learner.predict(tile)
            probs = probs.cpu().numpy().transpose(1, 2, 0)

            pred[yy_:yy_+size, xx_:xx_+size] += probs*mask[..., None]
            cnt[yy_:yy_+size, xx_:xx_+size] += mask

    pred[cnt > 0] /= cnt[cnt > 0, None]
    if small < size:
        pred = pred[pady:-pady, padx:-padx]

    return pred


def postprocess_predictions(img, size_thr=50, steps=5):
    """Expand the predictions without merging elements."""
    ids, nn_ = ndi.label(img != 0, structure=np.ones((3, 3)))
    if nn_ <= 1:  # no elements, no need to expand
        return img
    ids = skm.remove_small_objects(ids, size_thr, connectivity=1)
    img = ids != 0

    for _ in range(steps):
        ids_old = ids.copy()
        ids_old[ids_old > 0] += nn_  # add offset to avoid ID conflicts
        ids, nn_ = ndi.label(skm.binary_dilation(ids > 0),
                             structure=np.ones((3, 3)))
        for reg in regionprops(ids):
            yy_, xx_ = reg.coords.T

            if np.unique(ids_old[yy_, xx_])[1:].size > 1:  # merged components
                ids[yy_, xx_] = ids_old[yy_, xx_]

    return ids > 0


def predict_with_ensemble(models, image, equalized=True, use_ensemble=True):
    """Predict with ensemble of models."""
    # map predictions to standard class codes
    code_map = np.array([0, 120, 255, 60, 180, 119, 121], dtype=np.uint8)

    if use_ensemble:
        factors = [1, np.sqrt(2), 2, np.sqrt(8), 4]
    else:  # single model predictions
        factors, models = [1], [models[0]]

    for factor, model in zip(factors, models):
        if factor == 1:
            img = image
        elif isinstance(factor, int):
            img = np.asarray(Image.fromarray(image).reduce(factor))
        else:
            shape = [int(np.round(s/factor)) for s in image.shape]
            img = np.asarray(Image.fromarray(image).resize(shape[::-1]))

        equalize = equalized or factor < 2
        pred_i = predict_tiled(model, img, equalize=equalize)

        if factor == 1:
            pred = pred_i
        elif factor < 2:
            pred += _resize_multichannel(pred_i, pred.shape[:2])
        else:
            pred[..., 2:5] += _resize_multichannel(
                pred_i[..., 1:], pred.shape[:2])

    pred /= len(factors)

    pred[pred < 0.5] = 0
    pred = np.argmax(pred, axis=2)
    out = code_map[pred]

    return out


def clean_predictions(img):
    """Dilate predictions to account for the border class."""
    pred = np.zeros(img.shape, dtype=np.uint8)
    if img.dtype == np.uint8:  # standard class codes
        for lbl, step in zip([180, 60, 255], [2, 5, 5]):
            pred[postprocess_predictions(img == lbl, steps=step)] = lbl
    else:
        pred = postprocess_predictions(img != 0, steps=5)
    return pred


def create_overlay(img, pred):
    """Create overlay of the predictions on the input image."""
    img = np.stack([img, img, img], axis=-1)
    pred = np.stack([pred == 255, pred == 60, pred == 180], axis=-1) * 255

    composite = img.copy()
    composite[np.any(pred > 0, axis=-1)] = pred[np.any(pred > 0, axis=-1)]

    return composite


if __name__ == "__main__":
    pass
