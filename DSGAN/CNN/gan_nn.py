'''
This document contains pytorch functions used to construct GAN. The functions
all implement the `padding='SAME'` property as in tensorflow.
'''
import torch
from torch import nn
import torch.nn.functional as F

def same_padding (input_tensor, kernel, stride, deconv=False):
    '''This function implements torch.nn.Conv2d with `padding='SAME'` as in
    tensorflow
    '''
    def find_length (inp_len, k = kernel, s =stride, deconv=deconv):
        if inp_len%s ==0 or inp_len ==1 or deconv: out_len = max (k-s, 0)
        else: out_len = max (k- inp_len%s, 0)
        return out_len

    height, width = input_tensor.size()[2:]
    pad_height = find_length (height)
    pad_width  = find_length (width)
    top = pad_height//2+ pad_height%2
    bottom = pad_height- top
    left = pad_width//2+ pad_width%2
    right = pad_width - left 
    return [right, left, bottom, top]

def trim (arr, pad):
    right, left, bottom, top = pad
    arr2 = arr.shape[2]
    arr3 = arr.shape[2]
    return arr [:,:, top:(arr2-bottom), left:(arr3-right)]

def pad_one_axis (arr_A, arr_B, pad_amount, axis, tol):
    assert abs (pad_amount) < tol, \
            'The difference in array size should be smaller than {}'.format(tol)
    if axis == 2:
        top = abs (pad_amount)//2 + abs (pad_amount)%2
        bottom = abs (pad_amount) - top
        left, right = 0, 0
    elif axis == 3:
        left = abs (pad_amount)//2 + abs (pad_amount)%2
        right = abs (pad_amount) - left
        top, bottom= 0, 0
    else: raise Exception ('axis can only be 2 or 3')

    if pad_amount <0:
        return F.pad (arr_A, pad = [right, left, bottom, top]), arr_B
    elif pad_amount >0:
        return trim (arr_A, pad = [right, left, bottom, top]), arr_B
    else: return arr_A, arr_B

def same_concat (A, B, concat_axis, tolerance=2):
    '''Concatenate tensors that have slightly different shapes, by padding the
    A with zeros or trimming A.
    Args:
        `A`: a pytorch tensor of [number, channel, height, width]
        `B`: same format as `A`
        `concat_axis`: along which axis concatenation takes place
        `tolerance`: difference in shapes that can be accepted
    '''
    pad_ax2 = A.shape[2] - B.shape[2]
    pad_ax3 = A.shape[3] - B.shape[3]
    arr_A, arr_B = pad_one_axis (A, B, pad_ax2, axis=2, tol=tolerance)
    arr_A, arr_B = pad_one_axis (arr_A, arr_B, pad_ax3, axis=3, tol=tolerance)
    return torch.cat ([arr_A, arr_B], axis=concat_axis)


class conv2d(nn.Conv2d):
    '''A new class of nn.Conv2d that implements padding='SAME' '''

    def __init__(self, in_chan, out_chan, kernel, stride, bias=True,
                 wavelet=None):
        self.k = kernel
        self.s = stride
        super(conv2d, self).__init__(in_chan, out_chan, kernel, stride,
                                     padding=0, dilation=1, groups=1,
                                     bias=bias, padding_mode='zeros')

    def forward(self, x):
        '''Args:
            `x`: input pytorch tensor, shape=[B, C, H, W] '''
        padding = same_padding(x, self.k, self.s)
        return self._conv_forward(F.pad(x, pad=padding), self.weight,
                                  self.bias)


class deconv2d (nn.ConvTranspose2d):
    '''The padding for nn.ConvTranspose2d is calculated as dilation*(kernel)-pad. 
    Therefore, in order to produce no padding, it is essential to use padding =
    dilation*(kernel-1), as stored in the `torch_padding` attribute in this
    class. Then the 'SAME' padding will be implemented va `F.pad`.
    '''
    def __init__ (self, in_chan, out_chan, kernel, stride, dilation=1, bias= True):
        self.k = kernel
        self.s = stride
        self.torch_padding = dilation*(self.k-1)
        super (deconv2d, self).__init__(in_chan, out_chan, kernel, stride,
                padding=0, dilation=dilation, groups=1, bias=bias,
                padding_mode='zeros')

    def forward(self, x):
        s_padding = same_padding (x, self.k, self.s, deconv=True)
        return F.conv_transpose2d(F.pad(x, pad=s_padding), self.weight,
                self.bias, self.stride, padding=self.torch_padding,
                groups=self.groups, output_padding=0)

class pooling (nn.Module):
    def __init__ (self, kernel, stride):
        super (pooling, self).__init__()
        self.k = kernel
        self.s = stride

    def forward (self, x):
        padding = same_padding (x, self.k, self.s)
        return F.max_pool2d (F.pad(x, pad=padding), self.k, self.s)

def conv_layer (in_chan, out_chan, kernel=4, stride=2, bias=True, deconv=False,
        leaky_relu = False, norm = True, norm_act = True, dropout= None,
        spect=False, standout=False, wavelet=None, batch_before=False):
    block = []

    #convolution +/- spectral normalisation
    if batch_before: block.append (nn.BatchNorm2d (in_chan))
    if not deconv: block.append(conv2d(in_chan, out_chan, kernel, stride, bias=bias))
    else: block.append(deconv2d(in_chan, out_chan, kernel, stride, bias=bias))
    if spect: block[0] = nn.utils.spectral_norm(block[0])

    # no batch normalisation +/- dropout
    if leaky_relu: 
        if dropout is not None: block.append (nn.Dropout2d(dropout)) 
        block.append(nn.LeakyReLU(0.2))

    else: # with batch normalisation +/- dropout
        if norm: 
            block.append(nn.BatchNorm2d(out_chan))
            if dropout is not None: block.append (nn.Dropout2d(dropout)) 
            if norm_act: block.append(nn.LeakyReLU(0.2))

    # standout dropout with leak relu
    if type (block[-1]) == nn.LeakyReLU and standout == True:
        block [-1] = Standout_with_lrelu ()

    return block

class Standout_with_lrelu (nn.Module):
    ''' Apply Standout dropout with leaky relu. From
    https://papers.nips.cc/paper/5032-adaptive-dropout-for-training-deep-neural-networks.pdf
    https://github.com/gngdb/adaptive-standout/blob/master/standout/layers.py
    '''
    def __init__ (self, alpha=1., beta=0.):
        super (Standout_with_lrelu, self).__init__ ()
        self.alpha = alpha
        self.beta = beta

    def forward (self, x):
        activation = torch.sigmoid (self.alpha*x + self.beta)
        if self.training: #training mode
            mask = activation > activation.new(activation.shape).uniform_ (0., 1.)
            return mask*F.leaky_relu (x, 0.2)
        else: return activation*F.leaky_relu(x)

def drop_connect (model, drop_prob=None):
    '''from https://discuss.pytorch.org/t/dropconnect-implementation/70921'''
    orig_params = []
    for n, p in model.named_parameters():
        orig_params.append(p.clone())
        p.data = torch.nn.functional.dropout(p.data, p=drop_prob) * (1 - drop_prob)
    return orig_params

def restore_connect (model, orig_params):
    for orig_p, (n, p) in zip(orig_params, model.named_parameters()):  
        p.data = orig_p.data
