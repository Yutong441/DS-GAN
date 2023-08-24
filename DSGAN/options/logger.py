import os
import pandas as pd


def write_csv_mode(filename):
    if not os.path.exists(filename):
        write_mode, header = "w", True
    else:
        write_mode, header = "a", False
    return write_mode, header


def better_epoch(opt):
    if os.path.exists(opt.save_dir+"/metrics_val.csv"):
        metric_df = pd.read_csv(opt.save_dir+"/metrics_val.csv", index_col=[0])
        all_vals = metric_df[opt.target_metric].values
        if opt.better_metric == "larger":
            proceed = all_vals[-1] == all_vals.max()
        elif opt.better_metric == "smaller":
            proceed = all_vals[-1] == all_vals.min()
        return proceed
    else:
        return True


def write_log(opt, text):
    with open(opt.save_dir+"/log.txt", "a") as f:
        f.write(text+"\n")
