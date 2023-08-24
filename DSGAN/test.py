import torch
from .utils.dataloader import create_dataset
from .utils import train_stats as TS
from .utils.visualize import Visual
from .models import create_model


def test(opt):
    evalda, valda, testda = create_dataset(opt, mode="test")
    data_dict = {"train": evalda, "val": valda, "test": testda}
    model = create_model(opt)
    model.setup(opt)
    vis = Visual(opt.num_lev)

    with torch.no_grad():
        model.eval()

        # compute statistics
        for key, loader in data_dict.items():
            for i, data in enumerate(loader):
                model.set_input(data)
                model.forward()
                model.get_metrics()

            model.save_metrics("final_{}.csv".format(key), save_full=True)
        TS.sum_stats(opt.save_dir)
        TS.sum_graph(opt.save_dir)

        # save images
        for i, data in enumerate(valda):
            model.set_input(data)
            model.forward()
            vis.add_img({"input": model.real_A, "real": model.real_B,
                         "fake": model.fake_B})
            if i > opt.num_test:
                break

        vis.show(opt.save_dir+"/samples")
