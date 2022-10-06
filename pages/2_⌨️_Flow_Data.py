from http.client import ImproperConnectionState
import os
import streamlit as st
from streamlit_common import *
from streamlit_common import extract_feature_set

# pull session variables
if "df" not in st.session_state:
    df = None
    flow = None
else:
    df = st.session_state.df
    flow = df.flwid.unique()

# Constants
IDLE_THRESHOLD = 5


# Computations
# tflows, iflows, aflows = file_stats(df, IDLE_THRESHOLD)
if df is not None:
    df = filter_out_idle_flows(df, IDLE_THRESHOLD)
    st.title(f" Dataset of the flow : {flow[0]}")
    st.markdown("##")
else:
    st.subheader("Go back to home page and pickup a dataset!")
# left_column, middle_column, right_column = st.columns(3)

# with left_column:
#     st.subheader("Total flows:")

# with middle_column:
#     st.subheader("Idle flows:")
#     # st.subheader(iflows)
# with right_column:
#     st.subheader("Active flows:")
#     # st.subheader(aflows)

# st.subheader("Raw data")
if df is not None:
    df_p = extract_feature_set(df)
    st.write(f"## *Processed data:{df_p.shape}*")
    st.dataframe(
        df_p.style.background_gradient(subset=["up_kbps", "down_kbps", "mos"]), 1000
    )
    st.write("## *stats*")
    st.write(df_p[["down_kbps", "up_kbps", "local_ping", "server_ping"]].describe())
    st.write(f"## *raw data:{df.shape}*")
    st.dataframe(df)


# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
