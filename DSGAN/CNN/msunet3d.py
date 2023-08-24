import functools
import torch
from torch import nn
from .attention import AttentionConv3D
from .ECA import ECA


class MSU_Net(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=32,
                 norm_layer=nn.BatchNorm3d, use_dropout=False,
                 use_attention=False, use_ECA=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For
            example, # if |num_downs| == 7, image of size 128x128 will become
            of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(MSU_Net, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None,
                                             submodule=None,
                                             norm_layer=norm_layer,
                                             innermost=True,
                                             use_ECA=use_ECA)
        # add the innermost layer
        for i in range(num_downs - 5):
            # add intermediate layers with ngf * 8 filters
            args = dict(input_nc=None, submodule=unet_block,
                        norm_layer=norm_layer, use_dropout=use_dropout)
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, **args)

        if use_attention:
            att1, att2 = "before", "after"
        else:
            att1, att2 = "none", "none"
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None,
                                             submodule=unet_block,
                                             norm_layer=norm_layer,
                                             use_attention=att1)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None,
                                             submodule=unet_block,
                                             norm_layer=norm_layer,
                                             use_attention=att2)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None,
                                             submodule=unet_block,
                                             norm_layer=norm_layer,
                                             use_attention=att1)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc,
                                             submodule=unet_block,
                                             outermost=True,
                                             norm_layer=norm_layer)
        # add the outermost layer

    def forward(self, inp):
        """Standard forward"""
        out = self.model(2*inp - 1)
        return (1 + out)/2


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False,
                 use_attention="none", norm_layer=nn.BatchNorm3d,
                 use_dropout=False, use_ECA=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined
            submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc

        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        downconv = MultiScaleConv(input_nc, inner_nc, act_func=downrelu)

        if outermost:
            upconv = MultiScaleTransConv(inner_nc * 2, outer_nc, uprelu,
                                         norm_layer=norm_layer)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = MultiScaleTransConv(inner_nc, outer_nc, uprelu,
                                         norm_layer=norm_layer)
            down = [downrelu, downconv]
            if use_ECA:
                down = down + [ECA(dimension=3)]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = MultiScaleTransConv(inner_nc * 2, outer_nc, uprelu,
                                         norm_layer=norm_layer)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        if use_attention == "before":
            model = [AttentionConv3D(input_nc, input_nc, 1)] + model
        elif use_attention == "after":
            model = model + [AttentionConv3D(input_nc, input_nc, 1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class MultiScaleConv(nn.Module):
    def __init__(self, input_nc, output_nc, act_func, kern_sizes=[4, 6],
                 norm_layer=nn.BatchNorm3d, stride=2):
        super(MultiScaleConv, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        modules = []
        for ks in kern_sizes:
            padding = (ks - stride)//2
            one_mod = [
                nn.Conv3d(input_nc, output_nc, kernel_size=ks, stride=stride,
                          padding=padding, bias=use_bias),
                norm_layer(output_nc), act_func]
            modules.append(nn.Sequential(*one_mod))

        self.all_modules = nn.ModuleList(modules)
        self.conv = nn.Conv3d(output_nc*2, output_nc, kernel_size=1, stride=1,
                              padding=0, bias=use_bias)

    def forward(self, x):
        out = [mod(x) for mod in self.all_modules]
        out = torch.cat(out, dim=1)
        return self.conv(out)


class MultiScaleTransConv(nn.Module):
    def __init__(self, input_nc, output_nc, act_func, kern_sizes=[4, 6],
                 norm_layer=nn.BatchNorm3d, stride=2):
        super(MultiScaleTransConv, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        modules = nn.ModuleList()
        for ks in kern_sizes:
            padding = (ks - stride)//2
            one_mod = [
                nn.ConvTranspose3d(input_nc, output_nc, kernel_size=ks,
                                   stride=stride, padding=padding,
                                   bias=use_bias),
                norm_layer(output_nc), act_func]
            modules.append(nn.Sequential(*one_mod))

        self.all_modules = nn.ModuleList(modules)
        self.conv = nn.Conv3d(output_nc*2, output_nc, kernel_size=1, stride=1,
                              padding=0, bias=use_bias)

    def forward(self, x):
        out = [mod(x) for mod in self.all_modules]
        out = torch.cat(out, dim=1)
        return self.conv(out)
