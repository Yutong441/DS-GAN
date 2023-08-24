'''
Pytorch implementation of discrete wavelet transform and dual tree complex
wavelet transform.
'''
import torch
from torch import nn
import torch.nn.functional as F
import pywt
from . import gan_nn as gn

# ---------------------------------
# Load wavelet of a selected family
# ---------------------------------


def load_wavelet(family, dual_tree=False, inverse=False):
    '''Load the wavelet of a particular family into memory.
    Args:
        `family`: choices for DWT: 'haar', 'db', 'sym', 'coif', 'bior', 'rbio',
        'dmey', 'gaus', 'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor'
        choices for dual tree CWT: 'qshift_a', 'qshift_b', 'qshift_c'.
        'qshift_d', 'qshift_b_bp', 'q_shift_06', 'q_shift_32'
        `inverse`: whether the kernel is for inverse wavelet transform
        `dual_tree`: False for DWT; 'a' or 'b' for the real or imaginary branch
    Outputs:
        `wave_dict`: a dictionary with keys 'high_pass' and 'low_pass' for the
        respective filters
    '''
    if not dual_tree:
        wave = pywt.Wavelet(family)
        if not inverse:
            wave_dict = {'high_pass': wave.dec_hi[::-1],
                         'low_pass': wave.dec_lo[::-1]}
        else:
            wave_dict = {'high_pass': wave.rec_hi, 'low_pass': wave.rec_lo}

    return wave_dict


def get_wavelet(family, dual_tree=False, inverse=False, device="cpu"):
    '''Obtain the wavelet kernel for 2D discrete wavelet transform
    Output: [O, H, W]
    '''
    w = load_wavelet(family, dual_tree, inverse)
    high_pass = torch.tensor(w['high_pass'], device=device).reshape(-1, 1)
    low_pass = torch.tensor(w['low_pass'], device=device).reshape(-1, 1)

    return (torch.stack([low_pass.T*low_pass,
                         low_pass.T*high_pass,
                         high_pass.T*low_pass,
                         high_pass.T*high_pass], dim=0)).float()

# ----------------------------
# discrete wavelet transform
# ----------------------------


def DWT_2D(x, family=None, filters=None, inverse=False):
    '''Argument details: refer to `get_wavelet`.
    Args:
        `filters`: [4, H, W]

    Example usage:
    >>> img = torch.Tensor (img)
    >>> dwt_img = DWT_2D (img, family='bior2.2')
    '''
    if filters is None:
        filters = get_wavelet(family, inverse=inverse, device=x.device)
    b, c, h, w = x.shape
    if not inverse:
        x = x.reshape(-1, 1, h, w)
    pad_amount = gn.same_padding(x, kernel=filters.shape[2], stride=2)
    padded = F.pad(x, pad_amount)
    if not inverse:
        conv_x = F.conv2d(padded, filters[:, None], stride=2)
    else:
        conv_pad = filters.shape[2]
        conv_x = F.conv_transpose2d(padded, filters[:, None], stride=2,
                                    padding=conv_pad)
    return conv_x.reshape(-1, 4*c, h//2, w//2) if not inverse else conv_x

# ------------------------------------------
# Neural network layer incorporated with DWT
# ------------------------------------------


def conv_dwt(in_chan, out_chan, ks=4, stride=2, batchnorm=True, relu=True):
    padding = (ks - stride)//2
    bias = False if batchnorm else True
    model = []
    if relu:
        model.append(nn.LeakyReLU(0.2, True))
    model.append(nn.Conv2d(in_chan, out_chan, ks, stride=stride,
                           padding=padding, bias=bias))
    if batchnorm:
        model.append(nn.BatchNorm2d(out_chan))
    return nn.Sequential(*model)


def deconv_dwt(in_chan, out_chan, ks=4, stride=2, batchnorm=True,
               relu=True):
    padding = (ks - stride)//2
    bias = False if batchnorm else True
    block = []
    if relu:
        block.append(nn.ReLU(True))
    block.append(nn.ConvTranspose2d(
        in_chan, out_chan, kernel_size=ks, stride=stride, bias=bias,
        padding=padding))
    if batchnorm:
        block.append(nn.BatchNorm2d(out_chan))
    return nn.Sequential(*block)
