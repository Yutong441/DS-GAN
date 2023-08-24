import torch
from .utils.dataloader import create_dataset
from .options.logger import write_log
from .models import create_model


def train(opt):
    trainda, evalda, valda = create_dataset(opt, mode="train")
    model = create_model(opt)
    # load networks, create scheduler
    model.setup(opt)

    for epoch in range(model.starting_epoch,
                       opt.n_epochs + opt.n_epochs_decay):
        write_log(opt, "epoch {}".format(epoch))
        # update learning rates in the beginning of every epoch.
        model.update_learning_rate()

        index = 0
        for data in trainda:
            for i in range(0, data["A"].shape[0], opt.batch_size):
                model.set_input({key: val[i:(i+opt.batch_size)
                                          ] for key, val in data.items()})
                model.optimize_parameters(index)
                model.get_current_losses()
                index += 1

        # evaluate
        if epoch % opt.eval_freq == 0:
            with torch.no_grad():
                model.eval()
                for key, loader in {"train": evalda, "val": valda}.items():
                    for data in loader:
                        model.set_input(data)
                        model.forward()
                        model.get_metrics()

                    model.save_metrics("metrics_{}.csv".format(key),
                                       save_full=False)

                model.save_networks('latest')
                model.train()
