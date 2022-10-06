from __future__ import print_function
import argparse
import contextlib
import datetime
import os
from statistics import mean
import six
import sys
import time
import unicodedata
import matplotlib.pyplot as plt

# import packages
import numpy as np
import pandas as pd
from scipy import stats
import scipy as sc
import math

traffic_threshold = 5  # 5kbps is the activity threshold


def save_to_file(dataset, filepath, suffix=""):
    name_without_ext = os.path.splitext(filepath)
    # skip extraction if extracted before
    if suffix == "":

        name = f"{name_without_ext[0]}.xlsx"
        print("Saving file ", name, "...")

    else:
        name = f"{name_without_ext[0]}-{suffix}.xlsx"
        print("Saving file ", name, "...")

    dataset.to_excel(name, index=True)


def savefigure_to_file(plot, filepath, suffix=""):
    name_without_ext = os.path.splitext(filepath)
    # skip extraction if extracted before
    if suffix == "":

        name = f"{name_without_ext[0]}.png"
        print("Saving file ", name, "...")

    else:
        name = f"{name_without_ext[0]}-{suffix}.png"
        print("Saving file ", name, "...")

    plot.savefig(name)


def iqr(df, colname, bounds=[0.25, 0.75]):
    s = df[colname]
    q = s.quantile(bounds)
    return df[~s.clip(*q).isin(q)]


def out_low(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    out_low = q1 - 1.5 * iqr
    out_high = q3 + 1.5 * iqr
    # print("calculated low=", out_low)
    return out_low


def out_high(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    out_high = q3 + 1.5 * iqr
    # print("calculated high=", out_high)

    return out_high


def smooth_outliers(df, colname, window=10, maxmin="True"):
    rl_obj = df.rolling(window)[colname]
    q1 = rl_obj.quantile(0.25)
    q3 = rl_obj.quantile(0.75)
    iqr = q3 - q1
    low_cut = q1 - 1.5 * iqr
    low_cut = low_cut.round(0)
    high_cut = q3 + 1.5 * iqr
    high_cut = high_cut.round(0)
    # do not erase outliers but instead replace them with high and low values for smoothing
    # print("low_cut=", low_cut.head(200))
    # print("high_cut=", high_cut.head(200))
    # print("shape of cuts=", low_cut.shape)
    df.loc[(df[colname] < low_cut), colname] = low_cut
    df.loc[(df[colname] > high_cut), colname] = high_cut
    return df


def copy_audio(destdir, filepath):

    if not filepath:
        print("--file is mandatory")
        sys.exit(2)

    print(destdir, filepath)
    if not os.path.isdir(filepath):
        print(filepath, "is not a valid dir on your system")
        sys.exit(1)

    # STEP 1: loading from csv file
    dataset = pd.read_csv(
        filepath + "/consumer/results.csv", parse_dates=["time"], index_col="time"
    )
    # dataset = pd.read_csv(filepath)
    # STEP 2  initial stripping unrelated fields for the purpose
    dataset = dataset.drop("protocol", axis=1)
    dataset = dataset.drop("src_port", axis=1)
    dataset = dataset.drop("dst_port", axis=1)
    dataset = dataset.drop("application_type", axis=1)
    dataset = dataset.drop("local_ping", axis=1)
    # STEP 3: getting rid of undesired duplicant data(caused by database errors)

    dataset = dataset.sort_values(by="time", ascending=True)
    # print("Loaded raw data shape is ", dataset.shape)
    # dataset = dataset.drop_duplicates()
    # print("Data shape after duplicate removal is ", dataset.shape)

    # STEP 4: Seggrate flows in teams data
    flows = dataset.flwid.unique()
    dataset_arr = []
    for flow in flows:
        dataset_arr.append(dataset.query("flwid == @flow"))

    window_size = int(10)
    # STEP 5
    # In each flow dataset derive additional metrics
    for dataset in dataset_arr:
        # sort data in proper order locate duplicate timestamps and remove
        dataset = dataset.sort_values(
            ["time", "downstream_bytes", "upstream_bytes"],
            ascending=[True, True, True],
            inplace=False,
        )
        # Below one removes duplicate timestamps for the same flow id
        # which are normally not possible to happen at all!!!
        dataset = dataset[~dataset.index.duplicated(keep="first")]

        # is this the specified flow we are interested in
        flw = dataset.flwid.unique()
        print("flow=", int(flw[0]))

        # first save an original set of data before modifying
        # save_to_file(dataset, filepath, suffix=f"{int(flw[0])}")

        # first resample in  a second resolution because
        # we know that there are some missing samples
        # in the dataset which kills most of the post processing
        # algorithms who totally count on per second resolution

        # dataset["time"] = pd.to_datetime(dataset["time"].astype("str"))
        # dataset = dataset.set_index("time")
        print("Initial data shape is ", dataset.shape)
        dataset = dataset.resample("1S").ffill()
        print("Resampled data shape is ", dataset.shape)
        # derive new columns
        dataset["down_pkts"] = dataset["downstream_packets"].diff()
        dataset["up_pkts"] = dataset["upstream_packets"].diff()
        dataset["down_kbps"] = (8 * dataset["downstream_bytes"].diff()) / 1024
        dataset["up_kbps"] = (8 * dataset["upstream_bytes"].diff()) / 1024

        # if flw != 136017:
        #    continue

        # interpolate missing samples for upstream server ping
        dataset["server_ping"] = dataset["server_ping"].interpolate(
            method="linear", limit_direction="forward"
        )
        dataset["server_pngdiff"] = abs(dataset["server_ping"].diff())

        print("Shape of dataset before smoothing ", dataset.shape)
        # Smooth outliers in the downstream per second throughput
        dataset = smooth_outliers(
            dataset, colname="down_kbps", window=window_size, maxmin="True"
        )
        # Smooth outliers in the downstream per second packet rate
        dataset = smooth_outliers(
            dataset, colname="up_pkts", window=window_size, maxmin="True"
        )
        print("Shape of dataset after smoothing ", dataset.shape)

        # now calculate the average of down&up packets per second
        # be careful means calculated per window
        downmean = dataset.rolling(window_size, min_periods=5)["down_pkts"].mean()
        downmean.loc[(downmean == 0)] = 1
        # print("Down stream mean= ", downmean.head(100))
        # similarly derive up and down latencies
        dataset["down_lat"] = (
            (abs(dataset["down_pkts"] - downmean)) * (1 / (downmean)) * 1000
        )
        # dataset["down_lat50"] = (
        #    abs(dataset["down_pkts"] - 50)) * (1/50) * 1000

        # and compute differences from latency values
        dataset["down_lat_diff"] = abs(dataset["down_lat"].diff())
        # dataset["down_lat50_diff"] = abs(dataset["down_lat50"].diff())

        # And one more time calculate jitter across a window of depth window samples
        # summary = dataset.rolling(window_size, min_periods=int(window_size/2), )["down_lat_diff"].agg(
        #    ["mean", "max", "min", "count"]).add_prefix("downjitter_")
        summary = (
            dataset.ewm(com=0.5, adjust=True)["down_lat_diff"]
            .mean()
            .add_prefix("downjitter_")
        )

        # And one more time calculate jitter across a window of depth 10 samples
        ping_summary = (
            dataset.rolling(window=window_size, min_periods=int(window_size / 2))[
                "server_pngdiff"
            ]
            .agg(["mean", "max", "min", "count"])
            .add_prefix("server_ping_jitter_")
        )

        # Downstream throughput with moving average
        # And one more time calculate jitter across a window of depth 10 samples
        # tput = dataset.rolling(window_size, min_periods=int(window_size/2))["down_kbps"].agg(
        #     ["mean", "max", "min", "count"]).add_prefix("ingress_tghput_")
        tput = (
            dataset.rolling(
                window_size, min_periods=int(window_size / 2), win_type="exponential"
            )["down_kbps"]
            .agg(["mean"])
            .add_prefix("ingress_tghput_")
        )

        mean_thpt = tput["ingress_tghput_mean"].mean()

        # exponantial moving averages
        # tput = dataset.ewm(com=0.5)[
        #    "down_kbps"].mean().add_prefix("ingress_tghput_")
        # mean_thpt = tput.mean()

        print(f"Calculated average throughput in the flow {flw[0]} is {mean_thpt}")
        if mean_thpt < traffic_threshold or math.isnan(mean_thpt):
            continue
        # Copy audio file to destination with flow name
        cmdstr = f"cp {filepath}/audio.wav {destdir}/audio-{int(flw[0])}.wav"
        print("copy command = ", cmdstr)
        os.system(cmdstr)
