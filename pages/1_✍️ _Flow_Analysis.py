from __future__ import annotations
import os
from re import template
import streamlit as st
from streamlit_common import *
from streamlit_common import extract_feature_set
from synchronize import match_align
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# pull session variables
if "df" not in st.session_state:
    df = None
    flow = None
else:
    df = st.session_state.df
    flow = df.flwid.unique()

# Constants
IDLE_THRESHOLD = 5

# Local functions
# @st.cache(ttl=1, max_entries=1)
def update_tgraphs(dataset, audio):

    if audio is None:
        ROWS = 3
    else:
        ROWS = 4

    # plot thpt
    # dataset.down_kbps.plot(figsize=(20, 8))
    # locate local min and max for 5 seconds order
    ORDER = 5
    ilocs_min_rx = argrelextrema(dataset.down_kbps.values, np.less_equal, order=ORDER)[
        0
    ]
    ilocs_min_tx = argrelextrema(dataset.up_kbps.values, np.less_equal, order=ORDER)[0]

    line_fig = make_subplots(
        rows=ROWS,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("rx traffic", "tx traffic"),
    )
    line_fig.add_trace(
        go.Scatter(
            name="downstream traffic",
            x=dataset.index,
            y=dataset["down_kbps"],
            mode="lines",
            marker=dict(color="red"),
            showlegend=True,
            line=dict(shape="linear", color="rgb(10, 12, 240)", dash="solid", width=2),
        ),
        row=1,
        col=1,
    )

    line_fig.add_trace(
        go.Scatter(
            name="upstream traffic",
            x=dataset.index,
            y=dataset["up_kbps"],
            mode="lines",
            marker=dict(color="orange"),
            showlegend=True,
        ),
        row=2,
        col=1,
    )

    line_fig.add_trace(
        go.Scatter(
            name="rx_min",
            x=dataset.iloc[ilocs_min_rx].down_kbps.index,
            y=dataset.iloc[ilocs_min_rx].down_kbps,
            mode="markers",
            marker=dict(color="red", symbol="circle"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    line_fig.add_trace(
        go.Scatter(
            name="tx_,min",
            x=dataset.iloc[ilocs_min_tx].up_kbps.index,
            y=dataset.iloc[ilocs_min_tx].up_kbps,
            mode="markers",
            marker=dict(color="red", symbol="circle"),
            line=dict(width=1),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    line_fig.add_trace(
        go.Scatter(
            name="mos",
            x=dataset.index,
            y=dataset["mos"],
            mode="lines",
            marker=dict(color="black"),
            showlegend=True,
        ),
        row=3,
        col=1,
    )
    if audio is not None:
        line_fig.add_trace(
            go.Scatter(
                name="audio",
                y=audio,
                mode="lines",
                marker=dict(color="blue"),
                showlegend=True,
            ),
            row=4,
            col=1,
        )

    flow = dataset.flwid[0]
    line_fig.update_layout(
        yaxis_title="Throughput (kbps)",
        title=f"Stats for the flow={flow}",
        height=1200,
        width=1200,
        hovermode="x",
        template="seaborn",
    )

    return line_fig


def update_agraphs(dataset, audio):
    line_fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("mos", "audio"),
    )
    line_fig.add_trace(
        go.Scatter(
            name="mos",
            x=dataset.index,
            y=dataset["mos"],
            mode="lines",
            marker=dict(color="blue"),
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    line_fig.add_trace(
        go.Scatter(
            name="audio",
            y=audio,
            mode="lines",
            marker=dict(color="blue"),
            showlegend=True,
        ),
        row=2,
        col=1,
    )

    flow = dataset.flwid[0]
    line_fig.update_layout(
        yaxis_title="Mean opinion score",
        title=f"Stats for the flow={flow}",
        height=1200,
        width=1200,
        hovermode="x",
    )

    return line_fig


if "audio_enabled" not in st.session_state:
    st.session_state["audio_enabled"] = False

audio_enabled = st.sidebar.checkbox(
    "load audio file",
    key="audio_enabled_key",
    on_change=callback_function,
    value=st.session_state.audio_enabled,
    args=("audio_enabled", "audio_enabled_key"),
)


# Computations
if df is not None:
    df = filter_out_idle_flows(df, IDLE_THRESHOLD)
    drops = search_anamollies(df)
    drops = drops.drop("flwid", axis=1)
    st.sidebar.subheader("Events detected")
    st.sidebar.write(drops)
    st.title(f"Flow : {flow[0]}")
    st.markdown("##")
    # update graphs
    if audio_enabled:
        wav_signal = match_align(8, 0.4, f"results/audio/audio-{flow[0]}.wav")
        audio_file = open(f"results/audio/audio-{flow[0]}.wav", "rb")
        audio_bytes = audio_file.read()
        st.plotly_chart(update_tgraphs(df, wav_signal))
        st.audio(
            audio_bytes,
            format="audio/wav",
        )
    else:
        st.plotly_chart(update_tgraphs(df, None))


else:
    st.subheader("Go back to home page and pickup a dataset!")

# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
