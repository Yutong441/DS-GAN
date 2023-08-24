import numpy as np
import torch
import torchist
import torchmetrics.functional as MF


def nmse(good, bad):
    '''good, bad: D, H, W, C'''
    numer = torch.sum((good - bad)**2, dim=(0, 1, 2))
    denom = torch.sum(good**2, dim=(0, 1, 2))
    return torch.sqrt(numer/denom).tolist()


def rmse(good, bad):
    numer = torch.mean((good - bad)**2, dim=(0, 1, 2))
    return torch.sqrt(numer).tolist()


def ssim(good, bad):
    """ `good`, `bad`: shape = [D, H, W, C] """
    C = good.shape[-1]
    all_si = []
    for i in range(C):
        good_inp = good[..., i].unsqueeze(1)
        bad_inp = bad[..., i].unsqueeze(1)
        si = MF.structural_similarity_index_measure(good_inp, bad_inp)
        all_si.append(si.tolist())

    return all_si


def psnr(good, bad):
    """
    Formula from https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    term1 = 20.*torch.log(torch.amax(good, dim=(0, 1, 2)) + 1e-5)
    term2 = 10.*torch.log(torch.mean((good - bad)**2, dim=(0, 1, 2)) + 1e-5)
    psnr_val = (term1-term2)/np.log(10.)
    return psnr_val.tolist()


def mutual_information_pt(inp1, inp2):
    '''
    Mutual information between 2 images
    I(x, y) = p(x, y) \log(\frac{p(x, y)}{px py})
    Args:
        `inp1`, `inp2`: [..., D, H, W, C]
    '''
    C = inp1.shape[-1]
    out = []
    for i in range(C):
        hist_inp = torch.stack([inp1[..., i], inp2[..., i]], dim=-1)
        hist_inp = hist_inp.reshape([-1, 2])
        hist_2d = torchist.histogramdd(hist_inp, bins=20)
        # hist_2d = hist_2d.hist
        pxy = hist_2d / hist_2d.sum()
        px = pxy.sum(1)
        py = pxy.sum(0)
        px_py = px[:, None] * py[None, :]
        nzs = pxy > 0
        mi = torch.sum(pxy[nzs] * torch.log(pxy[nzs] / px_py[nzs]))
        out.append(float(mi))
    return out


def diff_median(good, bad, mask, channel_names):
    out_dict = {}
    for i, key in enumerate(channel_names):
        mask_good = torch.masked_select(good[..., i], mask)
        # TODO add the following line
        mask_good = torch.masked_select(mask_good, mask_good > 0)
        mask_bad = torch.masked_select(bad[..., i], mask)

        out_dict["median_"+key] = torch.median(mask_bad).tolist()
        true_med = torch.median(mask_good).tolist()
        val = true_med - out_dict["median_"+key]
        out_dict["diff_median_"+key] = val
    return out_dict


def sum_metrics(good_pt, bad_pt, input_pt, channel_names):
    if len(good_pt.shape) == 5:
        B, C, D, H, W = good_pt.shape
        # B, D, H, W, C
        good = good_pt.permute(0, 2, 3, 4, 1).reshape([-1, H, W, C])
        bad = bad_pt.permute(0, 2, 3, 4, 1).reshape([-1, H, W, C])
        inp = input_pt.permute(0, 2, 3, 4, 1)
    elif len(good_pt.shape) == 4:
        # D, H, W, C
        good = good_pt.permute(0, 2, 3, 1)
        bad = bad_pt.permute(0, 2, 3, 1)
        inp = input_pt.permute(0, 2, 3, 1)

    bad = bad.clip(0, 1)
    metric_dict = {}
    nmse_val = nmse(good, bad)
    ssim_val = ssim(good, bad)
    psnr_val = psnr(good, bad)
    rmse_val = rmse(good, bad)
    mi_val = mutual_information_pt(good, bad)

    for i, key in enumerate(channel_names):
        metric_dict["RNMSE_"+key] = nmse_val[i]
        metric_dict["SSIM_"+key] = ssim_val[i]
        metric_dict["PSNR_"+key] = psnr_val[i]
        metric_dict["RMSE_"+key] = rmse_val[i]
        metric_dict["MI_"+key] = mi_val[i]

    out_dict = diff_median(good, bad, inp[..., 0] > 0, channel_names)
    metric_dict = {**metric_dict, **out_dict}
    return metric_dict
