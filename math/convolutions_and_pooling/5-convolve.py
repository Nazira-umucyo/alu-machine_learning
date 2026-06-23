#!/usr/bin/env python3
"""Module for convolution on images using multiple kernels"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Perform a convolution on images using multiple kernels"""
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
    sh, sw = stride
    if padding == 'same':
        ph = int(np.ceil((kh - 1) / 2))
        pw = int(np.ceil((kw - 1) / 2))
    elif padding == 'valid':
        ph, pw = 0, 0
    elif isinstance(padding, tuple):
        ph, pw = padding
    else:
        ph, pw = 0, 0
    images = images.astype('float64')
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    mode='constant')
    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1
    output = np.zeros((m, oh, ow, nc))
    for k in range(nc):
        for i in range(oh):
            for j in range(ow):
                output[:, i, j, k] = np.sum(
                    padded[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :] *
                    kernels[:, :, :, k], axis=(1, 2, 3))
    return output
