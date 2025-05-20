import db_utils
import streamlit as st


def cluster_selection_logic():
    # check if channel has keywords
    keywords = db_utils.get_clustering_keywords(st.session_state.channel)
    if keywords.empty:
        keywords = db_utils.get_cluster_ids(st.session_state.channel)
    else:
        str_keywords = []
        for _, row in keywords.iterrows():
            str_keywords.append(", ".join(row["keywords"]) + (f" (ID: {row['cluster_id']})"))
        keywords = str_keywords
    return keywords


def load_app():
    # Streamlit app
    st.title("Cluster Data Viewer")
    st.write(f"**Selected channel**: {st.session_state.channel}")

    # Dropdown menu for selecting cluster ID
    st.sidebar.header("Select Cluster")
    clusters = cluster_selection_logic()
    selected_cluster_id = st.sidebar.selectbox("**Choose a cluster**:", clusters)
    try:
        selected_cluster_id = int(selected_cluster_id.split("(ID: ")[1].split(")")[0])
    except (AttributeError, ValueError):
        selected_cluster_id = int(selected_cluster_id)

    # add checkbox for summarization
    # summary_checkbox = st.sidebar.checkbox("Generate cluster description (Using LLM)", value=True)
    summary_checkbox = False

    # Display data corresponding to the selected cluster ID
    if st.sidebar.button("Show Data"):
        cluster_data = db_utils.get_messages_by_cluster(st.session_state.channel, selected_cluster_id)
        if not cluster_data.empty:
            st.write(f"Displaying data for Cluster ID: {selected_cluster_id}")
            if summary_checkbox:
                pass
                # st.header("Cluster Description")
                # with st.spinner("Generating cluster description..."):
                #     description, topic = utils.describe_cluster(cluster_data["text"].tolist())
                # st.write(description)
                # st.write(f"Topic: {topic}")
            else:
                df_description = db_utils.get_cluster_description(st.session_state.channel, selected_cluster_id)
                if not df_description.empty:
                    st.write(f"**Cluster Description**: {df_description['summary'].iloc[0]}")
                    st.write(f"**Keywords**: {df_description['keywords'].iloc[0]}")
                else:
                    st.write("No description available for this cluster.")
            st.write(f"**Number of messages in cluster:** {cluster_data.shape[0]}")

            st.header("Messages:")
            for _, row in cluster_data.iterrows():
                tab1, tab2 = st.tabs(["Eng", "Original Text"])
                with tab1:
                    st.write(f"Message ID: {row['id']}")
                    st.write(f"Date: {row['date']}")
                    st.write(f"Text: {row['text_en']}")
                with tab2:
                    st.write(f"Message ID: {row['id']}")
                    st.write(f"Date: {row['date']}")
                    st.write(f"Text: {row['text']}")
                st.write("---")
        else:
            st.write(f"No data available for Cluster ID: {selected_cluster_id}")


st.set_page_config(
    page_title="Cluster Exlporer",
    page_icon="🔍",
)


if "channel" not in st.session_state or st.session_state.channel is None:
    st.warning("**Please select a channel in the Home page.**", icon="⚠️")
else:
    # Load the app
    with st.spinner("Loading data from DB..."):
        load_app()
