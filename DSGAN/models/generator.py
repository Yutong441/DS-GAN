import re
from . import networks


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False,
             init_type='normal', init_gain=0.02, gpu_ids=[], dimension=2,
             use_attention=False, spectral=False, use_ECA=False):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks
        | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network:
        batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2


    Returns a generator
    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for
        256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and
        [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few
        downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project
        (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for
    non-linearity.
    """
    net = None
    norm_layer = networks.get_norm_layer(norm_type=norm,
                                         dimension=dimension)

    if dimension == 2:
        if netG == 'resnet_9blocks':
            from ..CNN.resnet2d import ResnetGenerator
            net = ResnetGenerator(input_nc, output_nc, ngf,
                                  norm_layer=norm_layer,
                                  use_dropout=use_dropout, n_blocks=9)
        elif netG == 'resnet_6blocks':
            from ..CNN.resnet2d import ResnetGenerator
            net = ResnetGenerator(input_nc, output_nc, ngf,
                                  norm_layer=norm_layer,
                                  use_dropout=use_dropout, n_blocks=6)
        elif netG == 'unet_wpd':
            from ..CNN.unet_any_wpd import unet
            net = unet(input_nc, output_nc)
        elif "unet_" in netG:
            from ..CNN.unet2d import UnetGenerator
            nlayers = int(re.sub("^unet_", "", netG))
            net = UnetGenerator(input_nc, output_nc, nlayers, ngf,
                                norm_layer=norm_layer,
                                use_dropout=use_dropout,
                                use_attention=use_attention,
                                use_ECA=use_ECA)

        if spectral:
            net.apply(networks.add_sn)

    elif dimension == 3:
        if "msunet_" in netG:
            from ..CNN.msunet3d import MSU_Net
            nlayers = int(re.sub("^msunet_", "", netG))
            net = MSU_Net(input_nc, output_nc, nlayers, ngf,
                          norm_layer=norm_layer,
                          use_dropout=use_dropout,
                          use_attention=use_attention,
                          use_ECA=use_ECA)

        elif "unet_" in netG:
            from ..CNN.unet3d import UnetGenerator
            nlayers = int(re.sub("^unet_", "", netG))
            net = UnetGenerator(input_nc, output_nc, nlayers, ngf,
                                norm_layer=norm_layer,
                                use_dropout=use_dropout,
                                use_attention=use_attention,
                                use_ECA=use_ECA)

        if spectral:
            net.apply(networks.add_sn)
    return networks.init_net(net, init_type, init_gain, gpu_ids)
