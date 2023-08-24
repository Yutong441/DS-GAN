import os
import numpy as np
import matplotlib.pyplot as plt


def preproc(img, chan, level, loc=None):
    if loc is not None:
        out_img = img[:, loc[4]:loc[5], loc[0]:loc[1], loc[2]:loc[3]]
        new_level = int(level*(loc[3] - loc[2])/img.shape[-1])
    else:
        out_img, new_level = img, level
    out_img = np.rot90(np.swapaxes(out_img[chan, :, :, new_level], 1, 0),
                       axes=(0, 1))
    return out_img


class Visual:
    def __init__(self, num_lev):
        self.num_lev = num_lev
        self.real_img = []
        self.fake_img = []
        self.inp_img = []

    def add_img(self, img_dict):
        '''img shape: C, D, H, W'''
        if len(img_dict["input"].shape) == 4:
            self.inp_img.append(img_dict["input"].permute(1, 0).cpu().numpy())
            self.real_img.append(img_dict["real"].permute(1, 0).cpu().numpy())
            self.fake_img.append(img_dict["fake"].permute(1, 0).cpu().numpy())
        elif len(img_dict["input"].shape) == 5:
            self.inp_img.append(img_dict["input"][0].cpu().numpy())
            self.real_img.append(img_dict["real"][0].cpu().numpy())
            self.fake_img.append(img_dict["fake"][0].cpu().numpy())

    def _show_one(self, level, savepath, loc=None):
        input_nc = self.inp_img[0].shape[0]
        ncol = input_nc + 2
        nrow = len(self.inp_img)

        fig, ax = plt.subplots(nrow, ncol, figsize=(ncol*4.8, nrow*4.8))
        kwargs = dict(cmap="gray", vmax=1)
        for j in range(nrow):
            for i in range(ncol):
                if i < input_nc:
                    ax[j, i].imshow(preproc(self.inp_img[j], i, level, loc),
                                    **kwargs)
                elif i == input_nc:
                    ax[j, i].imshow(preproc(self.real_img[j], 0, level, loc),
                                    **kwargs)
                elif i == input_nc + 1:
                    ax[j, i].imshow(preproc(self.fake_img[j], 0, level, loc),
                                    **kwargs)

        [axi.set_axis_off() for axi in ax.ravel()]
        plt.savefig(savepath)

    def show(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        W = self.inp_img[0].shape[3]
        depths = np.linspace(0, W, num=self.num_lev + 2)
        depths = [int(i) for i in depths[1:-1]]
        for i in depths:
            self._show_one(i, save_dir+"/level{}.jpg".format(i))
            self._show_one(i, save_dir+"/level{}_temporal.jpg".format(i),
                           loc=[16+1, 65+1, 69+1, 102+1, 14-8, 55-8])
            self._show_one(i, save_dir+"/level{}_occipital.jpg".format(i),
                           loc=[18+1, 76+1, 0+1, 44+1, 41-8, 90-8])

        # empty cache
        self.real_img = []
        self.fake_img = []
        self.inp_img = []
