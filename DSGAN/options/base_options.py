import argparse
import os
import shutil
import torch
import pandas as pd
from .default_config import cf
from . import logger


def str2list(x, conv_int=True):
    if x != "None":
        outlist = [i for i in x.split(",")]
        if conv_int:
            outlist = [int(i) for i in outlist]
        return outlist
    else:
        return None


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and
    saving the options.
    It also gathers additional options defined in <modify_commandline_options>
    functions in both dataset class and model class.
    """

    def __init__(self, isTrain):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        self.isTrain = isTrain

    def initialize(self, parser):
        """Define the common options that are used in both training and
        test."""
        # basic parameters
        self.initialized = True
        for key, val in cf.items():
            parser.add_argument("--"+key, **val)
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            format_class = argparse.ArgumentDefaultsHelpFormatter
            parser = argparse.ArgumentParser(formatter_class=format_class)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def initialize_dir(self, opt):
        # if not continue to train the model in the training mode
        if not opt.continue_train and self.isTrain:
            if os.path.exists(opt.save_dir):
                shutil.rmtree(opt.save_dir)

        if not os.path.exists(opt.save_dir):
            os.mkdir(opt.save_dir)
            os.mkdir(opt.save_dir+"/schedulers")
            os.mkdir(opt.save_dir+"/samples")
            os.mkdir(opt.save_dir+"/summary")

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        logger.write_log(opt, message)

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up
        gpu device."""
        opt = self.gather_options()
        self.initialize_dir(opt)
        if self.isTrain:
            self.print_options(opt)

        opt.isTrain = self.isTrain
        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])
            # enable performance acceleration during training
            if opt.isTrain:
                torch.backends.cudnn.benchmark = True
        else:
            opt.gpu_ids = ["cpu"]

        # set dataset options
        opt.downsize = str2list(opt.downsize)
        opt.common_shape = str2list(opt.common_shape)

        # find out output channel names
        target_path = "data/"+opt.data_root.split(",")[0]+"/target_train.csv"
        target_df = pd.read_csv(target_path, index_col=[0])
        opt.n_samples = len(target_df)
        out_chan = []
        for i in str2list(opt.out_chan, conv_int=False):
            if i in target_df.columns:
                out_chan.append(i)
        opt.out_chan = out_chan
        opt.output_nc = len(out_chan)

        # ----------------------------------------
        input_path = "data/"+opt.data_root.split(",")[0]+"/input_train.csv"
        input_df = pd.read_csv(input_path, index_col=[0])
        in_chan = []
        for i in str2list(opt.in_chan, conv_int=False):
            if i in input_df.columns:
                in_chan.append(i)
        opt.in_chan = in_chan
        opt.input_nc = len(in_chan)

        # ----------------------------------------
        # training parameters
        opt.target_metric = opt.target_metric+"_"+opt.out_chan[0]
        opt.disc_train = float(opt.disc_train)/float(opt.gen_train)
        if opt.disc_train < 1:
            opt.gen_train = int(1/opt.disc_train)
            opt.disc_train = 1
        else:
            opt.gen_train = 1
            opt.disc_train = int(opt.disc_train)
        self.opt = opt
        return self.opt
