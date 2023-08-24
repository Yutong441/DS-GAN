# This is an adaption of Anysize GAN (Connah 2020) to DAGAN
import torch
from torch import nn
from . import gan_nn as gn


class unet(nn.Module):
    def __init__(self, input_nc, output_nc, *args, **kwargs):
        super(unet, self).__init__()
        self.conv1 = nn.Sequential(*gn.conv_layer(input_nc, 64,
                                                  leaky_relu=True))  # x/2
        self.conv2 = nn.Sequential(*gn.conv_layer(64, 128))  # x/4
        self.conv3 = nn.Sequential(*gn.conv_layer(128, 256))  # x/8
        self.conv4 = nn.Sequential(*gn.conv_layer(256, 512))  # x/16
        self.conv5 = nn.Sequential(*gn.conv_layer(512, 512))  # x/32
        self.conv6 = nn.Sequential(*gn.conv_layer(512, 512, stride=1))

        self.up5 = nn.Sequential(*gn.conv_layer(512, 512, deconv=True,
                                                stride=1))
        self.up4 = nn.Sequential(*gn.conv_layer(1024, 1024, deconv=True)
                                 )  # x/16
        self.up3 = nn.Sequential(*gn.conv_layer(1536, 256, deconv=True))  # x/8
        self.up2 = nn.Sequential(*gn.conv_layer(512, 128, deconv=True))  # x/4
        self.up1 = nn.Sequential(*gn.conv_layer(256, 64, deconv=True))  # x/2
        self.up0 = nn.Sequential(*gn.conv_layer(128, 64, deconv=True))  # x/1
        self.final = nn.Sequential(*gn.conv_layer(64, output_nc, kernel=1,
                                                  stride=1, norm=False),
                                   torch.nn.Tanh())

    def forward(self, x):
        # encoding path
        conv1 = self.conv1(2*x - 1)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)

        # decoding path
        up5 = self.up5(conv6)
        up4 = self.up4(torch.cat((up5, conv5), dim=1))
        up3 = self.up3(torch.cat((up4, conv4), dim=1))
        up2 = self.up2(torch.cat((up3, conv3), dim=1))
        up1 = self.up1(torch.cat((up2, conv2), dim=1))
        up0 = self.up0(torch.cat((up1, conv1), dim=1))
        out = self.final(up0)
        return (1 + out)/2
