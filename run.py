import os
from DSGAN.options.base_options import BaseOptions
from DSGAN.run_model import predict

if __name__ == "__main__":
    opt = BaseOptions(isTrain=False).parse()
    assert os.path.exists(opt.dir_txt)
    for i in ["MD", "FA"]:
        predict(opt, i, in_chan=opt.in_chan.split(","))
