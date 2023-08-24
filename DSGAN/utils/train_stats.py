import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sum_dataset(df, ndec):
    df = df.drop(columns=["ID", "dataset"])
    df_mean = df.abs().mean(axis=0).round(ndec).values.squeeze()
    df_std = df.abs().std(axis=0).round(ndec).values.squeeze()
    one_dict = {}
    for j, col in enumerate(df.columns):
        one_dict[col] = "{} (\u00b1 {})".format(df_mean[j], df_std[j])
    return one_dict


def sum_stats(save_dir, ndec=3):
    df_paths = glob.glob(save_dir+"/final*.csv")
    all_df_stats = {}
    for i in df_paths:
        one_df = pd.read_csv(i, index_col=[0])
        datasets = np.unique(one_df["dataset"].values)

        for j in datasets:
            one_dict = sum_dataset(one_df[one_df["dataset"] == j], ndec)
            key = re.sub("final_", "", os.path.basename(i))
            key = re.sub(".csv", "", key)
            all_df_stats[key+"_"+j] = one_dict

    all_df_stats = pd.DataFrame.from_records(all_df_stats)
    all_df_stats.to_csv(save_dir+"/summary/metric_table.csv")


def sum_graph(save_dir):
    loss = pd.read_csv(save_dir+"/loss.csv", index_col=[0])
    train_loss = pd.read_csv(save_dir+"/metrics_train.csv", index_col=[0])
    val_loss = pd.read_csv(save_dir+"/metrics_val.csv", index_col=[0])

    # plot training losses
    drop_col = [i for i in loss.columns if re.search("^median", i)]
    loss = loss.drop(drop_col)
    loss.index = np.arange(len(loss))
    loss.plot(subplots=True)
    plt.savefig(save_dir+"/summary/loss.jpg")

    # plot metric changes
    for key, value in {"train": train_loss, 'val': val_loss}.items():
        value.index = np.arange(len(value))
        value.plot(subplots=True)
        plt.savefig(save_dir+"/summary/metrics_{}.jpg".format(key))
