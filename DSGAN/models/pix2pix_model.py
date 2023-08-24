import time
import torch
from .base_model import BaseModel
from .generator import define_G
from .discriminator import define_D
from . import loss
from ..utils import metrics


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from
    input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in
    the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for
        existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can
            use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned
        datasets.
        """
        # changing the default values to match the pix2pix paper
        # (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256',
                            dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0,
                                help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a
            subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test
        # scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'G_NMSE', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test
        # scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test
        # scripts will call <BaseModel.save_networks> and
        # <BaseModel.load_networks>
        if opt.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = define_G(opt.input_nc, opt.output_nc,
                             opt.ngf, opt.netG, opt.norm,
                             not opt.no_dropout, opt.init_type,
                             opt.init_gain, self.gpu_ids,
                             dimension=opt.dimension,
                             use_attention=opt.use_attention,
                             spectral=opt.spectral_gen)

        # define a discriminator; conditional GANs need to take both input and
        # output images; Therefore, #channels for D is input_nc + output_nc
        if opt.isTrain:
            self.netD = define_D(opt.input_nc + opt.output_nc,
                                 opt.ndf, opt.netD,
                                 opt.n_layers_D, opt.norm,
                                 opt.init_type, opt.init_gain,
                                 self.gpu_ids,
                                 spectral=opt.spectral,
                                 dimension=opt.dimension)

        if opt.isTrain:
            # define loss functions
            self.criterionGAN = loss.GANLoss(opt.gan_mode,
                                             opt.real_lab).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created
            # by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr,
                                                betas=(opt.beta1, 0.999),
                                                weight_decay=opt.L2_reg)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr,
                                                betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, inputs):
        """Unpack input data from the dataloader and perform necessary
        pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.
            include A, B, A_paths, B_paths

        The option 'direction' can be used to swap images in domain A and
        domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        # TODO: is to(device) necessary after pin memory?
        self.real_A = inputs['A' if AtoB else 'B'].to(self.device)
        self.real_B = inputs['B' if AtoB else 'A'].to(self.device)
        self.ID = inputs["ID"]
        self.dataset = inputs["dataset"]
        self.start_time = time.time()

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and
        <test>."""
        # G(A)
        self.fake_B = self.netG(self.real_A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        if self.opt.gan_mode == "wgan":
            for p in self.netD.parameters():
                p.data.clamp_(-0.01, 0.01)
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.add_noise(self.fake_B)), 1)
        # we use conditional GANs; we need to feed both input and output to the
        # discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.add_noise(self.real_B)), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        lam = self.opt.lambda_d
        self.loss_D = (self.loss_D_fake + self.loss_D_real)*0.5*lam
        self.loss_D.backward()

    def backward_G_nomatch(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.add_noise(self.fake_B)), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        if self.opt.gan_mode == "wgangp":
            real_AB = torch.cat((self.real_A, self.real_B), 1)
            self.loss_G_GAN += loss.cal_gradient_penalty(
                    self.netD, real_AB, fake_AB.detach())

    def backward_G_match(self):
        fake_AB = torch.cat((self.real_A, self.add_noise(self.fake_B)), 1)
        pred_fake, fakeM = self.netD(fake_AB, matching=True)

        real_AB = torch.cat((self.real_A, self.add_noise(self.real_B)), 1)
        pred_real, realM = self.netD(real_AB, matching=True)

        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # add feature matching loss
        lam = self.opt.lambda_FM
        for i in range(len(realM)):
            self.loss_G_GAN += torch.mean((realM[i] - fakeM[i])**2)*lam

        if self.opt.gan_mode == "wgangp":
            self.loss_G_GAN += loss.cal_gradient_penalty(
                    self.netD, real_AB, fake_AB.detach())

    def backward_G(self):
        if self.opt.lambda_FM == 0:
            self.backward_G_nomatch()
        else:
            self.backward_G_match()

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B
                                          ) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G_NMSE = loss.nmse(self.real_B, self.fake_B
                                     ) * self.opt.lambda_NMSE
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_NMSE
        self.loss_G.backward()

    def optimize_parameters(self, index=0):
        self.forward()                   # compute fake images: G(A)
        # update D
        if index % self.opt.gen_train == 0:
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            self.backward_D()                # calculate gradients for D
            self.optimizer_D.step()          # update D's weights

        # update G
        if index % self.opt.disc_train == 0:
            self.set_requires_grad(self.netD, False)
            # D requires no gradients when optimizing G
            self.optimizer_G.zero_grad()        # set G's gradients to zero
            self.backward_G()                   # calculate graidents for G
            self.optimizer_G.step()             # update G's weights

    def get_metrics(self):
        one_met = metrics.sum_metrics(self.real_B, self.fake_B.detach(),
                                      self.real_A, self.opt.out_chan)
        # self.start_time is reset after self.get_input
        one_met["time"] = time.time() - self.start_time
        if type(self.ID) == list:
            one_met["ID"] = self.ID[0]
            one_met["dataset"] = self.dataset[0]
        else:
            one_met["ID"] = self.ID
            one_met["dataset"] = self.dataset
        self.all_met.append(one_met)
