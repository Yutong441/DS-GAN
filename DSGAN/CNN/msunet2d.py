# from https://github.com/CN-zdy/MSU_Net/blob/main/models.py
import torch
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, ks=3, act_func="relu"):
        super(conv_block, self).__init__()
        if act_func == "relu":
            act = nn.ReLU(inplace=True)
        elif act_func == "leaky":
            act = nn.LeakyReLU(0.2, inplace=True)

        padding = (ks - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=ks, stride=1,
                      padding=padding, bias=True),
            nn.BatchNorm2d(ch_out), act,
            nn.Conv2d(ch_out, ch_out, kernel_size=ks, stride=1,
                      padding=padding, bias=True),
            nn.BatchNorm2d(ch_out), act
        )

    def forward(self, x):
        return self.conv(x)


class conv_3_1(nn.Module):
    def __init__(self, ch_in, ch_out, act_func="relu"):
        super(conv_3_1, self).__init__()
        self.conv_3 = conv_block(ch_in, ch_out, ks=3, act_func=act_func)
        self.conv_7 = conv_block(ch_in, ch_out, ks=7, act_func=act_func)
        self.conv = nn.Conv2d(ch_out * 2, ch_out, kernel_size=1, stride=1,
                              padding=0, bias=True)

    def forward(self, x):
        x3 = self.conv_3(x)
        x7 = self.conv_7(x)
        x = torch.cat((x3, x7), dim=1)
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, bilinear=False):
        super(up_conv, self).__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1,
                          bias=True),
                nn.BatchNorm2d(ch_in),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.ConvTranspose2d(
                ch_in, ch_out, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.up(x)
        return x


class MSU_Net(nn.Module):
    def __init__(self, input_nc=1, output_nc=1):
        super(MSU_Net, self).__init__()
        filters_number = [32, 64, 128, 256, 512]
        # filters_number = [16, 32, 64, 128, 256]
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_3_1(ch_in=input_nc, ch_out=filters_number[0])
        self.Conv2 = conv_3_1(ch_in=filters_number[0],
                              ch_out=filters_number[1], act_func="leaky")
        self.Conv3 = conv_3_1(ch_in=filters_number[1],
                              ch_out=filters_number[2], act_func="leaky")
        self.Conv4 = conv_3_1(ch_in=filters_number[2],
                              ch_out=filters_number[3], act_func="leaky")
        self.Conv5 = conv_3_1(ch_in=filters_number[3],
                              ch_out=filters_number[4], act_func="leaky")

        self.Up5 = up_conv(ch_in=filters_number[4], ch_out=filters_number[3])
        self.Up_conv5 = conv_3_1(ch_in=filters_number[4],
                                 ch_out=filters_number[3])

        self.Up4 = up_conv(ch_in=filters_number[3], ch_out=filters_number[2])
        self.Up_conv4 = conv_3_1(ch_in=filters_number[3],
                                 ch_out=filters_number[2])

        self.Up3 = up_conv(ch_in=filters_number[2], ch_out=filters_number[1])
        self.Up_conv3 = conv_3_1(ch_in=filters_number[2],
                                 ch_out=filters_number[1])

        self.Up2 = up_conv(ch_in=filters_number[1], ch_out=filters_number[0])
        self.Up_conv2 = conv_3_1(ch_in=filters_number[1],
                                 ch_out=filters_number[0])

        self.Conv_1x1 = nn.Conv2d(
            filters_number[0], output_nc, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        return d1
