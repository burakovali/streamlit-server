from __future__ import print_function

import argparse
import contextlib
import datetime
import os
from re import A
from statistics import mean
from turtle import down
import six
import sys
import time
import unicodedata
import matplotlib.pyplot as plt
import markdown as markdown


# # This makes plots prettier
# import seaborn

# seaborn.set()

# import packages
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import argrelextrema
import scipy as sc
import math

import streamlit as st

# import plotly.express as px
# import plotly.graph_objs as go
# from plotly.subplots import make_subplots


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


@st.cache(ttl=300, max_entries=10)
def filter_out_idle_flows(dataset, traffic_threshold):
    # STEP 4: Seggrate flows in teams data
    flows = dataset.flwid.unique()
    dataset_arr = []  # dataframe per flowid
    active_set = []  # training dataset generated per active flows
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

        # derive new columns
        # Below one removes duplicate timestamps for the same flow id
        # which are normally not possible to happen at all!!!
        dataset = dataset[~dataset.index.duplicated(keep="first")]
        dataset["down_pkts"] = dataset["downstream_packets"].diff()
        dataset["up_pkts"] = dataset["upstream_packets"].diff()
        dataset["down_kbps"] = (8 * dataset["downstream_bytes"].diff()) / 1024
        dataset["up_kbps"] = (8 * dataset["upstream_bytes"].diff()) / 1024

        # first save an original set of data before modifying
        flw = dataset.flwid.unique()
        # print("flow=", flw)

        # interpolate missing samples for upstream server ping

        # Smooth outliers in the downstream per second throughput
        dataset = smooth_outliers(
            dataset, colname="down_kbps", window=window_size, maxmin="True"
        )
        # Smooth outliers in the downstream per second packet rate
        dataset = smooth_outliers(
            dataset, colname="up_pkts", window=window_size, maxmin="True"
        )

        # now calculate the average of down&up packets per second
        # be careful means calculated per window
        downmean = dataset.rolling(window_size, min_periods=5)["down_pkts"].mean()
        downmean.loc[(downmean == 0)] = 1
        # print("Down stream mean= ", downmean.head(100))

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

        # print(f"Calculated average throughput in the flow {flw[0]} is {mean_thpt}")
        if mean_thpt < traffic_threshold or math.isnan(mean_thpt):
            # print("Discarding flow...")
            continue  # discard flow with no further computations

        active_set.append(dataset)

    active_dataframe = pd.concat(active_set)

    return active_dataframe


@st.cache(ttl=360, max_entries=1)
def filter_for_flow(df, flow):
    return df.query("flwid == @flow")


@st.cache
def get_active_flows(df, threshold):
    df = filter_out_idle_flows(df, threshold)
    return list(df["flwid"].unique())


@st.cache
def file_stats(df, threshold):
    # if df == None:
    #    return 0, 0, 0
    # compute number of flows
    tflows = df.flwid.unique()
    # active flows
    df = filter_out_idle_flows(df, threshold)
    aflows = df.flwid.unique()
    # st.session_state["df"] = df
    # st.session_state["idle_threshold"] = IDLE_THRESHOLD
    # st.session_state["filename"] = filename_selected
    iflows = len(tflows) - len(aflows)
    aflows = len(aflows)
    tflows = len(tflows)

    return tflows, iflows, aflows


# @st.cache(ttl=60,max_entries=1)
def extract_feature_set(dataset):
    window_size = int(10)
    # sort data in proper order locate duplicate timestamps and remove
    dataset = dataset.sort_values(
        ["time", "downstream_bytes", "upstream_bytes"],
        ascending=[True, True, True],
        inplace=False,
    )
    print("Initial data shape is ", dataset.shape)
    dataset = dataset.resample("1S").ffill()
    print("Resampled data shape is ", dataset.shape)

    # Below one removes duplicate timestamps for the same flow id
    # which are normally not possible to happen at all!!!
    dataset = dataset[~dataset.index.duplicated(keep="first")]
    dataset["down_pkts"] = dataset["downstream_packets"].diff()
    dataset["up_pkts"] = dataset["upstream_packets"].diff()
    dataset["down_kbps"] = (8 * dataset["downstream_bytes"].diff()) / 1024
    dataset["up_kbps"] = (8 * dataset["upstream_bytes"].diff()) / 1024
    # Smooth outliers in the downstream per second throughput
    dataset = smooth_outliers(
        dataset, colname="down_kbps", window=window_size, maxmin="True"
    )
    # Smooth outliers in the downstream per second packet rate
    dataset = smooth_outliers(
        dataset, colname="up_pkts", window=window_size, maxmin="True"
    )

    # now calculate the average of down&up packets per second
    # be careful means calculated per window
    downmean = dataset.rolling(window_size, min_periods=5)["down_pkts"].mean()
    downmean.loc[(downmean == 0)] = 1

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

    # print(f"Calculated average throughput in the flow {flw[0]} is {mean_thpt}")
    # if mean_thpt < traffic_threshold or math.isnan(mean_thpt):
    #     print("Discarding flow...")
    #     continue  # discard flow with no further computations

    # now drop unnecessary cols for training set
    dataset = dataset.drop("down_pkts", axis=1)
    dataset = dataset.drop("up_pkts", axis=1)
    dataset = dataset.drop("downstream_bytes", axis=1)
    dataset = dataset.drop("upstream_bytes", axis=1)
    dataset = dataset.drop("downstream_packets", axis=1)
    dataset = dataset.drop("upstream_packets", axis=1)
    dataset = dataset.drop("application_type", axis=1)
    dataset = dataset.drop("dst_port", axis=1)
    dataset = dataset.drop("protocol", axis=1)
    dataset = dataset.drop("flwid", axis=1)
    dataset = dataset.drop("src_port", axis=1)

    # dump dataset into react store in json format
    return dataset


@st.cache(ttl=90, max_entries=2)
def search_anamollies(gdataset):

    # initial stripping unrelated fields for the purpose
    gdataset = gdataset.drop("protocol", axis=1)
    gdataset = gdataset.drop("src_port", axis=1)
    gdataset = gdataset.drop("dst_port", axis=1)
    gdataset = gdataset.drop("application_type", axis=1)
    gdataset = gdataset.drop("local_ping", axis=1)
    gdataset = gdataset.drop("server_ping", axis=1)

    # read global dataset which was filtered before for active flows
    # gdataset
    df_anomally = pd.DataFrame(columns=["flwid", "time", "event"])

    summary = []
    ORDER = 5
    # Seggrate flows in teams data
    flws = gdataset.flwid.unique()
    # print("Flows ", flws)
    dataset_arr = []
    for flow in flws:

        dataset_arr.append(gdataset.query("flwid == @flow"))

    # In each flow dataset derive additional metrics
    for df in dataset_arr:

        # # now calculate the average of down&up packets in a 5 seconds windows for detecting MUTE
        # # be careful means calculated per window
        # downmean = df.rolling(ORDER, min_periods=ORDER)["down_kbps"].mean()
        # downmute = downmean.loc[(downmean < 3) & (downmean != 0)]
        # if len(downmute > 0):
        #     print("Downmutes; \n", downmute)

        tx_zero_df = df.loc[df["up_kbps"] == 0]
        # print("tx zero", tx_zero_df)
        if len(tx_zero_df) == 0:
            continue
        MIN_REPEAT = 3
        # check for consecutrive zero throughput samples in upstream flow
        old_index = tx_zero_df.index[0]
        repeat = 0
        rx_samples = []
        anomally = []
        disconnected = -1
        redial_detected = False
        # downmean = df.rolling(ORDER, min_periods=ORDER)["down_kbps"].mean()
        # downmute = downmean.loc[(downmean < 3) & (downmean != 0)]
        # if len(downmute > 0):
        #     print("Downmutes; \n", downmute)
        #     df_anomally.loc[len(df_anomally)] = [
        #         (downmute.flwid.unique())[0],
        #         downmute["time"],
        #         "rx muted",
        #     ]

        for index in tx_zero_df.index:
            # print("old index ", old_index)
            # print("Index in frame = ", index)
            diff = pd.Timedelta(index - old_index).seconds
            # print("Time delta is", diff)
            if 1 == diff:
                repeat += 1
                rx_samples.append(tx_zero_df["down_kbps"][index])
                # print("Consecutive repeats is", repeat)
                if repeat >= MIN_REPEAT:
                    if sum(rx_samples) > 0:
                        if redial_detected is False:
                            print(f"A possible ring tone is located at: {index}\n")
                            # df_anomally = df_anomally.append(
                            #     pd.Series([(df.flwid.unique())[0], index, "dial tone"]),
                            #     ignore_index=True,
                            # )

                            df_anomally.loc[len(df_anomally)] = [
                                (df.flwid.unique())[0],
                                index,
                                "dial tone",
                            ]
                            ring_start = index
                            redial_detected = True
                    else:
                        print("Flow is disconnected at: ", index)
                        # anomally.append(f"Flow is disconnected at: {index}")
                        disconnected = index
                        # df_anomally = df_anomally.append(
                        #     pd.Series([(df.flwid.unique())[0], index, "dropped"]),
                        #     ignore_index=True,
                        # )
                        df_anomally.loc[len(df_anomally)] = [
                            (df.flwid.unique())[0],
                            index,
                            "dropped",
                        ]

                        break

                    rx_samples.clear()

            else:
                repeat = 0

            old_index = index

        # calculate duration

        if -1 != disconnected:
            print("cropping data frame!")
            df = df.loc[:disconnected]

        # df_sum = df[{"down_kbps", "up_kbps", "mos"}].describe()
        # summary.append(df_sum)
        # summary_df = pd.concat(summary)

        # data_to_store = {
        #     "stats": summary_df.to_dict(),
        #     "anomally": df_anomally.to_dict(),
        # }
        # # summary.to_json(date_format="iso", orient="split")
        # summary_df = pd.concat(summary)
        # summary_df.to_json(date_format="iso", orient="split")
    # return data_to_store
    return df_anomally


def callback_function(state, key):
    # access the widget`s setting via st.session_state[key]
    # set the session state you intended to set in the widget
    st.session_state[state] = st.session_state[key]


# curdir = os.path.dirname(os.path.realpath(__file__)) + r'\\'
# css_file = os.path.join(curdir, 'style.css')
def local_css(css_file):
    with open(css_file) as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)


# this function is called to sytle web page with  larger fonts
