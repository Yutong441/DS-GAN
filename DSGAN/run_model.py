import os
import re
import numpy as np
import nibabel as nib
import torch
from .utils import dataloader
from .models import create_model


def preprocess(img_paths, select_depths=128, out_shape=[160, 192]):
    '''
    Method:
    1. downsample isotropic FLAIR (for evaluation mode)
    2. select the central 128 slices
    3. pad to 160 x 192
    '''
    if type(img_paths[0]) == str:
        t1 = nib.load(img_paths[0])
        affine = t1.affine
        ori_shape = t1.shape
        imgs = [nib.load(i).get_fdata() for i in img_paths]
    else:
        imgs = img_paths

    img = torch.tensor(np.stack(imgs, axis=-1), dtype=torch.float32)
    if select_depths:
        img = dataloader.select_central(img, select_depths)
    # put the channel axis as axis 0, depth axis as axis 1
    img = img.permute(3, 2, 0, 1)
    img = dataloader.padding_to_shape(img, out_shape)
    return img.unsqueeze(0), affine, ori_shape


def remove_padding(x, to_shape):
    if to_shape != x.shape[2:]:
        curr_shape = np.array(x.shape)[::-1][:2][::-1]
        shape_diff = curr_shape - np.array(to_shape)
        pad_dim = [shape_diff[0]//2,
                   curr_shape[0] - (shape_diff[0]//2+shape_diff[0] % 2),
                   shape_diff[1]//2,
                   curr_shape[1] - (shape_diff[1]//2+shape_diff[1] % 2)]
        return x[..., pad_dim[0]:pad_dim[1], pad_dim[2]:pad_dim[3]]
    else:
        return x


def restore_depth(img, D):
    start = (D - img.shape[1])//2
    end = D - img.shape[1] - start
    top_zeros = torch.zeros((img.shape[0], start, *img.shape[2:4]),
                            device=img.device)
    bot_zeros = torch.zeros((img.shape[0], end, *img.shape[2:4]),
                            device=img.device)
    out = torch.cat([top_zeros, img, bot_zeros], dim=1)
    return out


def postprocess(img, ori_shape):
    img = remove_padding(img, ori_shape[0:2])
    img = restore_depth(img, ori_shape[2])
    return img


def modify_opt(opt, pred_map="MD", in_chan=["T1", "FLAIR"]):
    # override user config
    opt.out_chan = [pred_map]
    opt.output_nc = 1
    opt.use_attention = False
    opt.model = "pix2pix"
    opt.spectral = True
    opt.spectral_gen = False

    if len(in_chan) == 2:
        suffix = ""
    else:
        suffix = "_" + in_chan[0]
        if in_chan[0] == "FLAIR":
            opt.dimension = 2

    opt.save_dir = "results/" + pred_map + suffix
    opt.in_chan = ["DL_yc/"+i for i in in_chan]
    opt.input_nc = len(opt.in_chan)
    return opt


def predict(opt, pred_map, in_chan):
    print("synhesizing "+pred_map)
    opt = modify_opt(opt, pred_map, in_chan)
    model = create_model(opt)
    model.setup(opt)

    if len(opt.in_chan) == 2:
        prefix = "s"
    else:
        prefix = "/s" + os.path.basename(opt.in_chan[0]).lower()
    ID = prefix+opt.out_chan[0]

    if os.path.isdir(opt.dir_txt):
        dir_list = sorted(os.listdir(opt.dir_txt))
        dir_list = [opt.dir_txt+"/"+i for i in dir_list]
    else:
        with open(opt.dir_txt, 'r') as f:
            dir_list = f.readlines()
            dir_list = [re.sub("\\n", "", i) for i in dir_list]

    with torch.no_grad():
        model.eval()
        for i in dir_list:
            inp_files = [i+"/"+j+"_MNI.nii.gz" for j in opt.in_chan]
            file_exist = [os.path.exists(j) for j in inp_files]
            if sum(file_exist) == len(file_exist):
                data, affine, ori_shape = preprocess(inp_files)

                if opt.dimension == 2:
                    data = data.squeeze(0).permute(1, 0, 2, 3)

                data.to(model.device)
                fake = model.netG(data)

                if opt.dimension == 3:
                    fake = postprocess(fake[0], ori_shape)
                else:
                    fake = postprocess(fake.permute(1, 0, 2, 3), ori_shape)

                fake = fake.detach().cpu().permute(2, 3, 1, 0).numpy()
                # the CNN returns MD times 100; need to divide by 100
                if opt.out_chan[0] == "MD":
                    fake /= 100
                flair_out = nib.Nifti1Image(fake[..., 0], affine=affine)
                flair_out.to_filename(
                    i+"/synthetic/{}_MNI.nii.gz".format(ID))
