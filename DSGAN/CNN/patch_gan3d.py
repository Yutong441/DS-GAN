import functools
import torch.nn as nn


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3,
                 norm_layer=nn.BatchNorm3d, spectral=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            # no need to use bias as BatchNorm3d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 4
        padw = 1
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2,
                              padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw,
                          stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        self.seq1 = nn.Sequential(*sequence)

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence = [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw,
                      stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        self.seq2 = nn.Sequential(*sequence)

        sequence = [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1,
                              padding=padw)]
        if spectral:
            sequence[-1] = nn.utils.spectral_norm(sequence[-1])
        # output 1 channel prediction map
        self.seq3 = nn.Sequential(*sequence)

    def forward(self, inp, matching=False):
        """Standard forward."""
        out1 = self.seq1(2*inp - 1)
        out2 = self.seq2(out1)
        out3 = self.seq3(out2)
        if matching:
            return out3, [out1, out2]
        else:
            return out3
