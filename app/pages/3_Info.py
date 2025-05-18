import db_utils
import pandas as pd
import streamlit as st
import utils


st.set_page_config(
    page_title="App Information",
    page_icon="🛈",
)

st.title("App Information")
st.write(
    """
    This app provides an interface to explore and analyze Telegram channels. 
    It allows users to view channel information, clustering process, and explore clusters.

    The application was developed as part of a master thesis called Telegram data clustering. The author is Ivan Žižka.
    """
)
