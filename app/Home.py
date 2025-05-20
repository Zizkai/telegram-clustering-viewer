import logging

import db_utils
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Home",
    page_icon="👋",
)


# Initialize session state if not set
if "channel" not in st.session_state:
    st.session_state.channel = None


st.title("Welcome to the Telegram Data Clustering App")
st.write("""
In the section below, you can select one of the channels. After choosing a channel and confirming your selection,
information about the channel and its clustering will be displayed. The selection also includes benchmark datasets
that were used for evaluating the clustering methods. Once a channel is selected, you can navigate to the left side
of the page and choose the "Explore Clusters" option. On this page, you can filter individual clusters and browse
through them. Each cluster is described using keywords. After selecting a cluster, a description of the cluster will 
appear, and you can browse the messages that belong to it. Messages are primarily displayed in English (if available), 
and you can also view the original text.
         """)


st.write("# Telegram Data Clustering")

with st.spinner("loading channels..."):
    channels = db_utils.get_channel_names()

with st.form("channel_selector"):
    channel = st.selectbox("Channel:", channels, help="Select a channel to view the data")
    selection_button = st.form_submit_button("Select")


if selection_button:
    # save the selected channel in the session state
    st.session_state["channel"] = channel

    # Display the selected channel
    st.write(f"**Selected channel**: {channel}")
    channel_info = db_utils.get_channel_info(channel)
    st.write(f"**Number of messages**: {channel_info['messages']}")
    st.write(f"**Channel created**: {channel_info['channel_created']}")
    # st.write(f"Clustering in DB: {db_utils.check_if_clustering_exists(channel)}")

    # create histogram of messages and use loading spinner
    st.write("## Channel activity over time")
    with st.spinner("Loading channel activity..."):
        histogram = db_utils.get_channel_message_histogram(channel)
    st.line_chart(histogram.set_index("month"), x_label="Month", y_label="Number of messages")

    st.write("## Clustering information")
    st.write("**Number of clusters**: ", len(db_utils.get_cluster_ids(channel)))

    with st.spinner("Loading clustering information..."):
        try:
            lustering_info = db_utils.get_clustering_info(channel)
        except Exception as e:
            logging.error(f"Error getting clustering info: {e}")
            clustering_info = None
    if clustering_info is None:
        st.write("Clustering info no in DB.")
        st.stop()

    inter = clustering_info["inter"]
    n_clusters = clustering_info["num_clusters"]
    silhouette = clustering_info["silhouette"]
    dbi = clustering_info["dbi"]
    # create elbow plot

    if inter is not None:
        st.write("### Elbow plot")
        d = pd.DataFrame({"inter": inter, "n_clusters": n_clusters})
        st.line_chart(d.set_index("n_clusters"), x_label="Number of clusters", y_label="Intertia")

    st.write("### Silhouette plot")
    d = pd.DataFrame({"silhouette": silhouette, "n_clusters": n_clusters})
    st.line_chart(d.set_index("n_clusters"), x_label="Number of clusters", y_label="Silhouette")

    st.write("### DBI plot")
    d = pd.DataFrame({"dbi": dbi, "n_clusters": n_clusters})
    st.line_chart(
        d.set_index("n_clusters"),
        x_label="Number of clusters",
        y_label="DBI",
        use_container_width=True,
    )
