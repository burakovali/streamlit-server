from distutils.command.upload import upload
from ossaudiodev import SNDCTL_DSP_SAMPLESIZE
from types import NoneType
import pandas as pd
import numpy as np

from streamlit_common import *
import streamlit as st
import contextlib
import datetime
import os
import plotly.express as px

from streamlit_common import search_anamollies

# emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(
    page_title="Voip Flows Dashboard", page_icon=":bar_chart:", layout="wide"
)


local_css("style.css")


# Constants
IDLE_THRESHOLD = 5  # 5kbps
FILEPATH = "results/"
FILENAMES = ["dataset-1.csv", "dataset-2.csv", "dataset-3.csv"]
# Initialize the session state object for the initial values of the states
if "dropped" not in st.session_state:
    st.session_state["dropped"] = ""
# "st.session_state object:", st.session_state


# file upload
@st.cache(ttl=360, max_entries=1)
def file_upload(filename):
    if not os.path.isfile(f"{FILEPATH}{filename}"):
        print(FILEPATH, "is not a valid file on your system")
        return None

    df = pd.read_csv(f"{FILEPATH}{filename}", parse_dates=["time"], index_col="time")
    return df


# SIDEBAR
st.sidebar.title("Navigation")
st.sidebar.header("Please Filter Here")


if "filename" not in st.session_state:

    default_ix = 0
else:

    default_ix = FILENAMES.index(st.session_state.filename)

filename = st.sidebar.selectbox(
    "Select flow dataset",
    options=FILENAMES,
    index=default_ix,
    key="filename_key",
    on_change=callback_function,
    args=("filename", "filename_key"),
)


df = file_upload(filename)
tflows, iflows, aflows = file_stats(df, IDLE_THRESHOLD)
# Analye all active flows for anamollies
df_actives = filter_out_idle_flows(df, IDLE_THRESHOLD)
anamollies = search_anamollies(df_actives)

dropped = st.sidebar.checkbox(
    "show only dropped calls",
    key="dropped_key",
    on_change=callback_function,
    value=st.session_state.dropped,
    args=("dropped", "dropped_key"),
)
if dropped:
    flows = list(anamollies.flwid.unique())
else:
    flows = get_active_flows(df, IDLE_THRESHOLD)

if len(df) > 0:
    if "flow" not in st.session_state:
        dfault_ix = 0
    else:
        try:
            dfault_ix = flows.index(st.session_state.flow)
        except Exception as err:
            dfault_ix = 0

    flow = st.sidebar.selectbox(
        "Select the flow",
        options=flows,
        key="flow_key",
        index=dfault_ix,
        on_change=callback_function,
        args=("flow", "flow_key"),
    )

# MAIN PAGE
st.title(":bar_chart: Microsoft Teams voip call data explorer")
st.markdown("##")

first_column, left_column, middle_column, right_column = st.columns(4)
with first_column:
    st.subheader("Selected file:")
    st.subheader(filename)
with left_column:
    st.subheader("Total flows:")
    st.subheader(tflows)
with middle_column:
    st.subheader("Idle flows:")
    st.subheader(iflows)
with right_column:
    st.subheader("Active flows:")
    st.subheader(aflows)


# Store a subset of filtered flow dataframe in the session_state
st.session_state["df"] = filter_for_flow(df, flow)

# display anamollies
st.subheader("Dropped Calls:")

st.dataframe(anamollies, 600)
# st.session_state
# st.write(st.session_state)

# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
