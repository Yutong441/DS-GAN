'''Anysize Unet with more extensive inception module incorporating DWT'''
import torch
from torch import nn
from . import dwt_nn as dn


class unet(nn.Module):
    def __init__(self, input_nc, output_nc, wavelet="bior2.2"):
        super(unet, self).__init__()
        self.register_buffer("filter_r", dn.get_wavelet(wavelet))

        N = input_nc
        self.conv1 = dn.conv_dwt(input_nc, 64, relu=False,
                                 batchnorm=False)  # 4
        self.conv2 = dn.conv_dwt(64+4*N, 128)  # 4
        self.conv3 = dn.conv_dwt(128+16*N, 256)  # 16
        self.conv4 = dn.conv_dwt(256+64*N, 512)  # 64
        self.conv5 = dn.conv_dwt(512+256*N, 512)  # 256
        self.conv6 = dn.conv_dwt(512, 512, ks=3, stride=1)  # 256

        self.up5 = dn.deconv_dwt(512, 512, ks=3, stride=1)
        self.up4 = dn.deconv_dwt(512+512, 512)  # x/16
        self.up3 = dn.deconv_dwt(512+512+256*N, 256)  # x/8
        self.up2 = dn.deconv_dwt(256+256+64*N, 128)  # x/4
        self.up1 = dn.deconv_dwt(128+128+16*N, 64)  # x/2
        self.up0 = dn.deconv_dwt(64+64+4*N, 64)  # x/1
        self.final = nn.Sequential(nn.ReLU(True), nn.Conv2d(
            64+input_nc, output_nc, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        # setup wavelet transform
        wave1 = dn.DWT_2D(2*x-1, filters=self.filter_r)
        wave2 = dn.DWT_2D(wave1, filters=self.filter_r)
        wave3 = dn.DWT_2D(wave2, filters=self.filter_r)
        wave4 = dn.DWT_2D(wave3, filters=self.filter_r)

        # encoding path
        conv1 = self.conv1(x)
        conv2 = self.conv2(torch.cat((conv1, wave1), dim=1))
        conv3 = self.conv3(torch.cat((conv2, wave2), dim=1))
        conv4 = self.conv4(torch.cat((conv3, wave3), dim=1))
        conv5 = self.conv5(torch.cat((conv4, wave4), dim=1))
        conv6 = self.conv6(conv5)

        # decoding path
        up5 = self.up5(conv6)
        up4 = self.up4(torch.cat((up5, conv5), dim=1))
        up3 = self.up3(torch.cat((up4, conv4, wave4), dim=1))
        up2 = self.up2(torch.cat((up3, conv3, wave3), dim=1))
        up1 = self.up1(torch.cat((up2, conv2, wave2), dim=1))
        up0 = self.up0(torch.cat((up1, conv1, wave1), dim=1))
        out = self.final(torch.cat((up0, x), dim=1))
        return (1 + out)/2
