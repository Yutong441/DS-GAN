import re
import os
import numpy as np
import pandas as pd
import nibabel as nib

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader


def padding_to_shape(x, to_shape):
    '''
    Args:
        `x`: pytorch tensor. The last two dimensions to be padded
        `to_shape`: the shape of the final padded tensor
    '''
    if to_shape != x.shape[2:]:
        shape_diff = np.array(to_shape) - np.array(x.shape)[::-1][:2][::-1]
        pad_dim = [shape_diff[0]//2, shape_diff[0]//2+shape_diff[0] % 2,
                   shape_diff[1]//2, shape_diff[1]//2+shape_diff[1] % 2]
        return F.pad(x, pad_dim)
    else:
        return x


def standardize_img(img, mask):
    masked_img = np.ma.masked_array(img, mask=mask)
    masked_img = (masked_img - masked_img.mean())/masked_img.std()
    masked_img = masked_img.clip(-2, 2)
    masked_img = (masked_img + 2)/4
    return np.ma.filled(masked_img, fill_value=0)


def select_central(img, N):
    start = (img.shape[2] - N)//2
    return img[:, :, start:(start + N)]


def downsample(img, to_size, dimension):
    '''
    Args:
        `img`: pytorch tensor; shape C, D, H, W
        `to_size`: list/tuple of 2 elements to the desired height and width
        `dimension`: perform 2D or 3D downsampling
    '''
    if dimension == 2:
        img = F.interpolate(img, to_size, antialias=True, mode="bilinear")
    elif dimension == 3:
        C, D, H, W = img.shape
        new_depth = np.round(D*to_size[0]/H)
        downsize = (int(new_depth), *to_size)
        img = F.interpolate(img.unsqueeze(0), downsize, mode="trilinear")
        img = img.squeeze(0)
    return img


def collate_img2d(batch):
    if len(batch) > 1:
        out_dict = {}
        for key in batch[0].keys():
            out_dict[key] = torch.concatenate([i[key] for i in batch], axis=0)
        return out_dict
    else:
        return batch[0]


def read_file_list(file_list):
    all_f = []
    for i in file_list:
        one_df = pd.read_csv(i, index_col=[0])
        one_df["ID"] = [os.path.basename(re.sub("/DL_yc.*$", "", 
            os.path.dirname(j))) for j in one_df.iloc[:, 0]]
        # ID is the directory containing the files
        one_df["dataset"] = os.path.basename(os.path.dirname(i))
        all_f.append(one_df)
    all_f = pd.concat(all_f, axis=0)
    return all_f


def full_dpath(ID_list_str, app_str):
    ID_list = ID_list_str.split(",")
    return ["data/"+i+"/"+app_str for i in ID_list]


class dataloader(torch.utils.data.Dataset):
    def __init__(self, mode, input_file, target_file=None,
                 transformation='default',
                 common_shape=None, downsize=None,
                 dimension=2, select_depths=None,
                 slice_num=None, in_chan=None, out_chan=None):
        '''
        Args:
            `root_dir`: top-level directory for a particular dataset, which
            should contains `train`, `test` and `labels` folder
            `mode`: 'train', 'test' or 'validate'
            `common_shape`: the height and width for all images. If an image
            does not have this shape, zero padding is used.
            `select_channels`: which channels to be in the input
            `select_depths`: an integer, number of central slices to select
            `dimension`: whether to treat each 3D volume as 2D slices, i.e.,
            put the depth axis with the batch axis
        '''
        self.input_files = read_file_list(input_file)
        if target_file is not None:
            self.target_files = read_file_list(target_file)
        else:
            self.target_files = None

        if common_shape is not None:
            self.common_shape = [int(i) for i in common_shape]
        else:
            self.common_shape = None
        if downsize is not None:
            self.downsize = [int(i) for i in downsize]
        else:
            self.downsize = None

        if transformation == 'default' and mode == "train":
            transformation1 = torch.nn.Sequential(
                    # transforms.RandomHorizontalFlip(0.5),
                    # transforms.RandomVerticalFlip(0.5),
                    transforms.RandomAffine(0, translate=(0.05, 0.05),
                                            scale=(0.95, 1.0))
                    )
            transformation2 = torch.nn.Sequential(
                    transforms.ColorJitter(brightness=0.1),
                    transforms.GaussianBlur(3)
                    )
            self.transform1 = torch.jit.script(transformation1)
            self.transform2 = torch.jit.script(transformation2)
        else:
            self.transform1 = None
            self.transform2 = None
        self.dimension = dimension
        self.select_depths = select_depths
        self.mode = mode
        self.slice_num = slice_num
        self.in_chan = in_chan
        self.out_chan = out_chan

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        '''
        Output a dictionary:
        A: input image
        B: target image
        '''
        input_img = self._prep_img(
            self.input_files[self.in_chan].iloc[index].values)
        ID = self.input_files["ID"].iloc[index]
        dataset = self.input_files["dataset"].iloc[index]

        if self.target_files is not None:
            target_img = self._prep_img(
                self.target_files[self.out_chan].iloc[index].values)
            # multiply MD images by 100 to make the range approach [0, 1]
            for index, key in enumerate(self.out_chan):
                if key == "MD":
                    target_img[index] = 100 * target_img[index]
                else:
                    target_img[index] = target_img[index].clip(0, 1)
            out_dict = self._augment(input_img, target_img)
        else:
            out_dict = {"A": input_img}

        out_dict["ID"] = ID
        out_dict["dataset"] = dataset
        return out_dict

    def _prep_img(self, img_list, mask=None):
        '''
        Assemble and transform a list of images
        '''
        img = [nib.load(i).get_fdata() for i in img_list]

        if mask is not None:
            img = [standardize_img(i, mask) for i in img]

        img = torch.tensor(np.stack(img, axis=-1), dtype=torch.float32)
        if self.select_depths:
            img = select_central(img, self.select_depths)
        # put the channel axis as axis 0, depth axis as axis 1
        img = img.permute(3, 2, 0, 1)

        if self.common_shape is not None:
            img = padding_to_shape(img, self.common_shape)
        if self.downsize is not None:
            img = downsample(img, self.downsize, self.dimension)
        return img

    def _augment(self, img1, img2):
        if self.transform1 is not None:
            old_shape1 = img1.shape
            old_shape2 = img2.shape
            new_shape = [-1, 1, *img1.shape[2:]]
            # make sure the input and target share the same transform
            merge_img = torch.cat([img1.reshape(new_shape),
                                   img2.reshape(new_shape)], axis=0)
            merge_img = self.transform1(merge_img)

            N = old_shape1[0]*old_shape1[1]
            img1_out = merge_img[:N].reshape(old_shape1)
            img2_out = merge_img[N:].reshape(old_shape2)
        else:
            img1_out = img1
            img2_out = img2

        if self.transform2 is not None:
            old_shape1 = img1_out.shape
            new_shape = [-1, 1, *img1_out.shape[2:]]
            img1_out = self.transform2(img1_out.reshape(new_shape))
            img1_out = img1_out.reshape(old_shape1)

        if self.dimension == 2:
            # D, C, H, W
            img1_out = img1_out.permute(1, 0, 2, 3)
            img2_out = img2_out.permute(1, 0, 2, 3)
            if self.mode == "train" and self.slice_num is not None:
                samples = torch.randint(0, img1_out.shape[0],
                                        (self.slice_num, ))
                img1_out = img1_out[samples]
                img2_out = img2_out[samples]
        return {"A": img1_out, "B": img2_out}


def create_dataset(opt, mode="train"):
    '''
    If mode is train, return the train (in training mode), train (in eval mode)
    and validation datasets
    If mode is test, return the train, validation and test datasets all in eval
    modes.
    '''
    drop_last = False
    if opt.gpu_ids[0] != "cpu" and opt.batch_size > 1 and opt.isTrain:
        drop_last = True

    load_args = dict(select_depths=128,
                     transformation=opt.transformation,
                     common_shape=opt.common_shape,
                     downsize=opt.downsize, dimension=opt.dimension,
                     slice_num=opt.slice_num,
                     in_chan=opt.in_chan,
                     out_chan=opt.out_chan)
    data_args = dict(num_workers=opt.num_workers,
                     pin_memory=False if opt.gpu_ids[0] == "cpu" else True,
                     # pin_memory_device=opt.gpu_ids[0],
                     batch_size=1,
                     drop_last=drop_last,
                     collate_fn=collate_img2d if opt.dimension == 2 else None)

    evalda = dataloader("test",
                        full_dpath(opt.data_root, "input_train.csv"),
                        full_dpath(opt.data_root, "target_train.csv"),
                        **load_args)
    eval_loader = DataLoader(dataset=evalda, shuffle=False, **data_args)
    valda = dataloader("test",
                       full_dpath(opt.data_root, "input_val.csv"),
                       full_dpath(opt.data_root, "target_val.csv"),
                       **load_args)
    val_loader = DataLoader(dataset=valda, shuffle=False, **data_args)

    if mode == "train":
        trainda = dataloader("train",
                             full_dpath(opt.data_root, "input_train.csv"),
                             full_dpath(opt.data_root, "target_train.csv"),
                             **load_args)
        train_loader = DataLoader(dataset=trainda, shuffle=True, **data_args)
        return train_loader, eval_loader, val_loader

    elif mode == "test":
        testda = dataloader("test",
                            full_dpath(opt.test_root, "input.csv"),
                            full_dpath(opt.test_root, "target.csv"),
                            **load_args)
        test_loader = DataLoader(dataset=testda, shuffle=False, **data_args)
        return eval_loader, val_loader, test_loader
