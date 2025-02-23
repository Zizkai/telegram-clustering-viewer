import sqlite3

import pandas as pd
import streamlit as st
import utils

st.set_page_config(
    page_title="Cluster Exlporer",
    page_icon="🔍",
)


def get_unique_cluster_ids():
    connection = sqlite3.connect("data.db")
    query = "SELECT DISTINCT cluster FROM messages"
    df = pd.read_sql_query(query, connection)
    connection.close()
    return sorted(df["cluster"].tolist())


def get_cached_descriptions(cluster):
    connection = sqlite3.connect("data.db")
    query = "SELECT * FROM cluster_description WHERE cluster = ?"
    df = pd.read_sql_query(query, connection, params=(cluster,))
    connection.close()
    return df


# Streamlit app
st.title("Cluster Data Viewer")

# Dropdown menu for selecting cluster ID
st.sidebar.header("Select Cluster ID")
cluster_ids = get_unique_cluster_ids()  # Example cluster IDs, you can modify this list
selected_cluster_id = st.sidebar.selectbox("Choose a cluster ID:", cluster_ids)
# add checkbox for summarization
summary_checkbox = st.sidebar.checkbox("Generate cluster description", value=True)
location_checkbox = st.sidebar.checkbox("Generate locations", value=False)
# used cached data
cached_checkbox = st.sidebar.checkbox("Use cached data", value=False)


def get_data_from_db(cluster_id):
    connection = sqlite3.connect("data.db")
    query = "SELECT * FROM messages WHERE cluster = ?"
    df = pd.read_sql_query(query, connection, params=(cluster_id,))
    connection.close()
    df.rename(columns={"text_original": "text"}, inplace=True)
    return df


# Display data corresponding to the selected cluster ID
if st.sidebar.button("Show Data"):
    cluster_data = get_data_from_db(selected_cluster_id)
    if not cluster_data.empty:
        st.write(f"Displaying data for Cluster ID: {selected_cluster_id}")
        if summary_checkbox:
            st.header("Cluster Description")
            if cached_checkbox:
                cached_data = get_cached_descriptions(selected_cluster_id)
                description = cached_data.iloc[0]["description"]
                topic = cached_data.iloc[0]["topic"]
            else:
                with st.spinner("Generating cluster description..."):
                    description, topic = utils.describe_cluster(
                        cluster_data, selected_cluster_id
                    )
            st.write(description)
            st.write(f"Topic: {topic}")
        if location_checkbox:
            st.header("Locations")
            with st.spinner("Loading main locations..."):
                st.write(
                    utils.extract_locations_from_clusters(
                        cluster_data, selected_cluster_id
                    )
                )
        st.header("Messages:")
        for _, row in cluster_data.iterrows():
            st.write(f"Message ID: {row['id']}")
            st.write(f"Date: {row['date']}")
            st.write(f"Text: {row['text_en']}")
            st.write("---")
    else:
        st.write(f"No data available for Cluster ID: {selected_cluster_id}")
