from . import networks


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch',
             init_type='normal', init_gain=0.02, gpu_ids=[],
             spectral=False, dimension=2):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator;
        effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the
        network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized
        images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers
        in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic]
        (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is
        real or not.
        It encourages greater color diversity but has no effect on spatial
        statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU
    for non-linearity.
    """
    net = None
    norm_layer = networks.get_norm_layer(norm_type=norm, dimension=dimension)

    if dimension == 2:
        from ..CNN import patch_gan as PG
        if netD == 'basic':  # default PatchGAN classifier
            net = PG.NLayerDiscriminator(input_nc, ndf, n_layers=3,
                                         norm_layer=norm_layer)
        elif netD == 'n_layers':  # more options
            net = PG.NLayerDiscriminator(input_nc, ndf, n_layers_D,
                                         norm_layer=norm_layer)
        elif netD == 'pixel':     # classify if each pixel is real or fake
            net = PG.PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)

        if spectral:
            net.apply(networks.add_sn)
    elif dimension == 3:
        from ..CNN import patch_gan3d as PG3
        if netD == "basic":
            net = PG3.NLayerDiscriminator(input_nc, ndf, n_layers=3,
                                          norm_layer=norm_layer,
                                          spectral=False)
        if spectral:
            net.apply(networks.add_sn)
    return networks.init_net(net, init_type, init_gain, gpu_ids)
