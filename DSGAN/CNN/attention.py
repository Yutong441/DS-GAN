# https://github.com/leaderj1001/Stand-Alone-Self-Attention/blob/master/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, groups=1, bias=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, \
            "out_channels should be divided by groups."

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1,
                                  kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1,
                                  kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                  bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                    bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                    bias=bias)

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding,
                             self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(
                3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(
                3, self.kernel_size, self.stride)

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(
                batch, self.groups, self.out_channels // self.groups,
                height, width, -1)
        v_out = v_out.contiguous().view(
                batch, self.groups, self.out_channels // self.groups,
                height, width, -1)

        q_out = q_out.view(batch, self.groups,
                           self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(
                batch, -1, height, width)
        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out',
                             nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out',
                             nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out',
                             nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)


class AttentionConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, groups=1, bias=False):
        super(AttentionConv3D, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, \
            "out_channels should be divided by groups."

        rest_chan = out_channels - 2 * (out_channels // 3)
        self.rel_d = nn.Parameter(torch.randn(out_channels // 3, 1, 1, 1,
                                  kernel_size, 1, 1), requires_grad=True)
        self.rel_h = nn.Parameter(torch.randn(out_channels // 3, 1, 1, 1,
                                  1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(rest_chan, 1, 1, 1,
                                  1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1,
                                  bias=bias)
        self.query_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1,
                                    bias=bias)
        self.value_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1,
                                    bias=bias)

        self.reset_parameters()

    def forward(self, x):
        batch, channels, depth, height, width = x.size()

        padded_x = F.pad(x, [self.padding]*6)
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(
                3, self.kernel_size, self.stride).unfold(
                4, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(
                3, self.kernel_size, self.stride).unfold(
                4, self.kernel_size, self.stride)

        if self.out_channels % 3 == 0:
            k_out_d, k_out_h, k_out_w = k_out.split(
                    self.out_channels // 3, dim=1)
        else:
            k_out_d, k_out_h, k_out_w, k_out_n = k_out.split(
                    self.out_channels // 3, dim=1)
            k_out_w = torch.cat([k_out_w, k_out_n], dim=1)

        k_out = torch.cat((
            k_out_d + self.rel_d,
            k_out_h + self.rel_h,
            k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(
                batch, self.groups, self.out_channels // self.groups,
                depth, height, width, -1)
        v_out = v_out.contiguous().view(
                batch, self.groups, self.out_channels // self.groups,
                depth, height, width, -1)

        q_out = q_out.view(
                batch, self.groups, self.out_channels // self.groups,
                depth, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bncdhwk,bncdhwk -> bncdhw', out, v_out).view(
                batch, -1, depth, height, width)
        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out',
                             nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out',
                             nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out',
                             nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)
        init.normal_(self.rel_d, 0, 1)
