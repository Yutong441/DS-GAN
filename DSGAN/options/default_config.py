# default config and help page
cf = {
    # --------------------essential configs--------------------
    "data_root": dict(
        type=str, default="SCANS",
        help="dataset ID, must be present in the data/ folder"
        ),
    "test_root": dict(
        type=str, default="SCANS",
        help="dataset ID, must be present in the data/ folder"
        ),
    "save_dir": dict(
        type=str, default="results/pix2pix",
        help="models are saved here"
        ),
    "gpu_ids": dict(
        type=str, default="-1",
        help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU"
        ),

    # --------------------model parameters--------------------
    "model": dict(
        type=str, default="pix2pix",
        help="chooses which model to use. [cycle_gan | pix2pix"
        ),
    "in_chan": dict(
        type=str, default="T1,FLAIR",
        help="names of the input channels"
        ),
    "out_chan": dict(
        type=str, default="FA",
        help="names of the output channels"
        ),
    "ngf": dict(
        type=int, default=64,
        help="# of gen filters in the last conv layer"
        ),
    "ndf": dict(
        type=int, default=64,
        help="# of discrim filters in the first conv layer"
        ),
    "netD": dict(
        type=str, default="basic",
        help="specify discriminator architecture [basic|n_layers|pixel]." +
             "The basic model is a 70x70 PatchGAN." +
             "n_layers allows you to specify the layers in the discriminator"
        ),
    "netG": dict(
        type=str, default="unet_5",
        help="specify generator architecture" +
             "[resnet_9blocks | resnet_6blocks | unet_256 | unet_128]"
            ),
    "n_layers_D": dict(
        type=int, default=3,
        help="only used if netD==n_layers"
        ),
    "norm": dict(
        type=str, default="instance",
        help="instance normalization or batch normalization" +
             "[instance | batch | none]"
        ),
    "init_type": dict(
        type=str, default="normal",
        help="network initialization[normal | xavier | kaiming | orthogonal]"
        ),
    "init_gain": dict(
        type=float, default=0.02,
        help="scaling factor for normal, xavier and orthogonal."
        ),
    "use_attention": dict(
        action="store_true",
        help="use attention in generator"
        ),
    "no_dropout": dict(
        action="store_true",
        help="no dropout for the generator"
        ),
    "disc_noise": dict(
        type=float, default=0.05,
        help="add Gaussian noise to discriminator"
        ),
    "real_lab": dict(
        type=float, default=1.0,
        help="discriminator label for real images"
        ),
    "spectral": dict(
        action="store_true",
        help="whether to perform spectral normalization in discriminator"
        ),
    "spectral_gen": dict(
        action="store_true",
        help="whether to perform spectral normalization in generator"
        ),
    "disc_train": dict(
        type=int, default=1,
        help="train the discriminator how many times more than the generator"
        ),
    "gen_train": dict(
        type=int, default=1,
        help="train the generator how many times more than the discriminator"
        ),

    # --------------------dataset--------------------
    "direction": dict(
        type=str, default="AtoB",
        help="AtoB or BtoA"
        ),
    "num_workers": dict(
        default=4, type=int,
        help="# threads for loading data"
        ),
    "batch_size": dict(
        type=int, default=1,
        help="input batch size"
        ),
    "slice_num": dict(
        type=int, default=10,
        help="Number of slices per volume during training" + \
        "(only if dimension is 2)"
        ),
    "transformation": dict(
        type=str, default="default",
        help="anything other than default will turn off transformation"
        ),
    "common_shape": dict(
        type=str, default="160,192",
        help="Height and width to pad to"
        ),
    "downsize": dict(
        type=str, default="160,192",
        help="downsample to what height and width after padding"
        ),
    "dimension": dict(
        type=int, default=3,
        help="whether to process 3D volumes as 2D slices"
        ),
    "synthetic": dict(
        action="store_true",
        help="whether or not to use synthetic inputs"
        ),


    # --------------------save results--------------------
    "target_metric": dict(
        type=str, default="RNMSE",
        help="save the epoch with the highest value in which metric"
        ),
    "better_metric": dict(
        type=str, default="smaller",
        help="whether it is better to have a smaller value in the target " +
             "metric or a larger value"
        ),
    "verbose": dict(
        action="store_true",
        help="if specified, print more debugging information"
    ),
    "suffix": dict(
        default="", type=str,
        help="customized suffix: opt.name = opt.name + suffix:" +
             "e.g., {model}_{netG}_size{load_size}"
        ),

    # --------------------training--------------------
    "epoch": dict(
        type=str, default="latest",
        help="which epoch to load? set to latest to use latest cached model"
        ),
    "load_iter": dict(
        type=int, default="0",
        help="which iteration to load?" +
             "if load_iter > 0, the code will load models by" +
             "iter_[load_iter]; otherwise," +
             "the code will load models by [epoch]"
        ),
    "lambda_L1": dict(
        type=float, default=100.0,
        help="weight for L1 loss"
        ),
    "lambda_NMSE": dict(
        type=float, default=1.0,
        help="weight for MSE loss"
        ),
    "lambda_FM": dict(
        type=float, default=0.0,
        help="feature matching loss"
        ),
    "lambda_d": dict(
        type=float, default=1.0,
        help="discriminator weight"
        ),
    "L2_reg": dict(
        type=float, default=0.,
        help="weight for L2 regularization of the generator"
        ),
    'lambda_A': dict(
        type=float, default=10.0,
        help='weight for cycle loss (A -> B -> A)'
        ),
    'lambda_B': dict(
        type=float, default=10.0,
        help='weight for cycle loss (B -> A -> B)'
        ),
    'lambda_identity': dict(
        type=float, default=0.5,
        help='use identity mapping. Setting lambda_identity other than 0' +
        'has an effect of scaling the weight of the identity mapping loss.' +
        'For example, if the weight of the identity loss should be 10' +
        'times smaller than the weight of the reconstruction loss, ' +
        'please set lambda_identity = 0.1'
        ),

    # --------------------save net--------------------
    "continue_train": dict(
        action="store_true",
        help="continue training: load the latest model"
        ),

    # --------------------training method--------------------
    "eval_freq": dict(
        type=int, default=1,
        help="how often to evaluate the accuracy of the model"
        ),
    "n_epochs": dict(
        type=int, default=40,
        help="number of epochs with the initial learning rate"
        ),
    "n_epochs_decay": dict(
        type=int, default=20,
        help="number of epochs to linearly decay learning rate to zero"
        ),
    "beta1": dict(
        type=float, default=0.5,
        help="momentum term of adam"
        ),
    "lr": dict(
        type=float, default=0.0002,
        help="initial learning rate for adam"
        ),
    "gan_mode": dict(
        type=str, default="vanilla",
        help="the type of GAN objective. [vanilla| lsgan | wgangp]." +
             "vanilla GAN loss is the cross-entropy objective used" +
             "in the original GAN paper."
         ),
    "pool_size": dict(
        type=int, default=50,
        help="the size of image buffer that stores previously generated images"
        ),
    "lr_policy": dict(
        type=str, default="linear",
        help="learning rate policy. [linear | step | plateau | cosine]"
        ),
    "lr_decay_iters": dict(
        type=int, default=50,
        help="multiply by a gamma every lr_decay_iters iterations"
        ),

    # --------------------test--------------------
    "num_test": dict(
        type=int, default=5,
        help="how many test images to run"
        ),
    "num_lev": dict(
        type=int, default=3,
        help="how many levels to show"
        ),
    "dir_txt": dict(
        type=str, default="",
        help="either name of a directory, or path to a txt file" +
             "containing a list of directories"
        ),
}
