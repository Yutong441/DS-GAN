import numpy as np
import torch
import torch.nn.functional as F


def standardize_img(img, mask):
    brain_voxels = torch.masked_select(img, mask.type(torch.bool))
    img_mean = float(brain_voxels.mean())
    img_std = float(brain_voxels.std())
    masked_img = (img - img_mean)/img_std
    masked_img = masked_img.clip(-2, 2)
    masked_img = (masked_img + 2)/4
    out_img = masked_img.masked_fill_((1 - mask).type(torch.bool), 0)
    return out_img


def aliase(img, ratio=0.07, preserve=0.07, random=True):
    '''
    Simulate wrap-around aliasing artifacts
    Args:
        `img`: [H, W, D]
        `ratio`: corrupt the phase of how many percent of the phase encodes.
        Original paper corrupts 30 encoding lines, corresponding to 0.1875 in a
        160 pixel image
        `preserve`: preserve the central phase encodes

    Reference: https://openreview.net/pdf?id=H1hWfZnjM
    '''
    H, W, D = img.shape
    fft_img = torch.fft.fftn(img.type(torch.complex64))  # 2D FFT
    fft_img = torch.fft.fftshift(fft_img)

    angle = torch.zeros(img.shape, device=img.device)
    if not random:
        torch.manual_seed(100)

    rand_num = torch.pi*(2 * torch.rand(int(H*ratio)) - 1)
    if not random:
        torch.manual_seed(100)
    for index, i in enumerate(torch.randint(0, H, [int(H*ratio)])):
        angle[i] += rand_num[index]

    # preserve the central phase encodes
    x_lim = int(H * (1 - preserve))
    x_lim = [x_lim // 2, H - (x_lim // 2 + x_lim % 2)]
    angle[x_lim[0]:x_lim[1]] = 0

    # corrupted phase image has a magnitude of 1
    corrupted = torch.polar(torch.ones(img.shape, device=img.device), angle)
    # multiply by the phase image for translation
    fft_img = torch.fft.ifftshift(corrupted*fft_img)
    ifft = torch.fft.ifftn(fft_img)
    return torch.abs(ifft).to(torch.float32)


def aliase_dir(img, random=True, ratio=None, axis=None):
    '''
    Downsample along an arbitrary axis
    Args:
        `img`: [1, D, H, W]
        `random`: if False, keep the same sampling pattern and magnitude of
        corruption
        `axis`: 0 (axial plane), 1 (coronal plane)
    '''
    if ratio is None:
        ratio = float(torch.rand([1])/10)
    if axis is None:
        axis = int(torch.randint(0, 3, [1]))

    img = torch.moveaxis(img, axis, 0)
    new_img = aliase(img, ratio, random=random)
    new_img = torch.moveaxis(new_img, 0, axis)
    return new_img


def downsample_z(img, z_res=None, offset=None):
    '''
    Downsample along the z axis (depth) of `img`
    Args:
        `img`: pytorch tensor, [H, W, D], in resolution of 1mm x 1mm x 1mm
        `z_res`: new resolution in mm along the z axis
        `offset`: start downsampling at which z location
    '''
    if z_res is None:
        z_res = int(torch.randint(1, 7, [1]))
    if offset is None:
        offset = int(torch.randint(-4, 5, [1]))

    H, W, D = img.shape
    offset1 = max(offset, 0)
    down_z = img[..., np.arange(offset1, D, step=z_res)]
    offset2 = min(offset, 0)
    if offset2 < 0:
        new_shape = list(img.shape)
        new_shape[2] = -offset2
        zero_pad = torch.zeros(new_shape, device=img.device)
        down_z = torch.cat([zero_pad, down_z], axis=-1)

    up_z = F.interpolate(down_z.permute(2, 0, 1).unsqueeze(0).unsqueeze(0),
                         (D, H, W), antialias=False, mode="trilinear")
    up_z = up_z[0, 0].permute(1, 2, 0)
    return up_z
