import db_utils
import numpy as np
import pandas as pd
import plotly.express as px
import sklearn as sk
import streamlit as st


@st.cache_data
def compute_centroids(channel):
    messages_embeddings = db_utils.get_cluster_embeddings(channel)
    centroids_df = []
    for cluster_id in messages_embeddings["cluster_id"].unique():
        cluster_embeddings = messages_embeddings[messages_embeddings["cluster_id"] == cluster_id]
        cluster_embeddings = cluster_embeddings["embedding"].to_list()
        # compute centroid
        centroid = np.mean(cluster_embeddings, axis=0)
        centroids_df.append({"cluster_id": cluster_id, "centroid": centroid})

    centroids_df = pd.DataFrame(centroids_df)

    # reduce dimensionality
    pca = sk.decomposition.PCA(n_components=2)
    reduced = pca.fit_transform(centroids_df["centroid"].to_list())
    centroids_df["x"] = reduced[:, 0]
    centroids_df["y"] = reduced[:, 1]
    return centroids_df


def create_dimension_reduction(channel, method="PCA"):
    messages_embeddings = db_utils.get_cluster_embeddings(channel)

    embedding = messages_embeddings["embedding"].tolist()
    embedding = [list(e) for e in embedding]

    if method == "PCA":
        pca = sk.decomposition.PCA(n_components=2)
        reduced = pca.fit_transform(embedding)
    else:
        raise ValueError(f"Unknown method: {method}")

    messages_embeddings["x"] = reduced[:, 0]
    messages_embeddings["y"] = reduced[:, 1]
    return messages_embeddings


def create_cluster_plot(channel, method="PCA"):
    centroids = compute_centroids(channel=channel)
    # ploty crete scatter plot
    fig = px.scatter(centroids, x="x", y="y", color="cluster_id")
    return fig


st.set_page_config(
    page_title="Clustering process",
    page_icon="📊",
)

st.title("Clustering Process")
fig = create_cluster_plot(st.session_state.channel)
st.plotly_chart(fig)
