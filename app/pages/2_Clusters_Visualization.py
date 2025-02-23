import re
import sqlite3

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.manifold import MDS


def get_centroids():
    df = pd.read_csv("rybar_openai_clustered_cleared.csv")
    df["embedding"] = df["embedding"].apply(
        lambda x: [float(num) for num in re.split(r"\s+", x.strip("[]").strip())]
    )
    df["embedding"] = df["embedding"].apply(lambda x: np.array(x))
    centroids = df.groupby("cluster", as_index=False).agg(
        {"embedding": lambda x: np.mean(x, axis=0)}
    )
    return centroids


def get_cluster_topic(cluster_id):
    connection = sqlite3.connect("data.db")
    query = "SELECT topic FROM cluster_description WHERE cluster = ?"
    df = pd.read_sql_query(query, connection, params=(cluster_id,))
    connection.close()
    return df.iloc[0]["topic"]


def cluster_distances(df):
    clusters = df["cluster"].unique().tolist()
    distance_matrix = np.zeros((len(clusters), len(clusters)))
    for i in range(len(clusters)):
        for j in range(len(clusters)):
            if i == j:
                continue
            cluster1 = df[df["cluster"] == clusters[i]]
            cluster2 = df[df["cluster"] == clusters[j]]
            center1 = np.mean(cluster1["embedding"].values, axis=0)
            center2 = np.mean(cluster2["embedding"].values, axis=0)
            distance = np.linalg.norm(center1 - center2)
            distance_matrix[i, j] = distance
    return distance_matrix


with st.spinner("Computing distance matrix..."):
    df = get_centroids()
    distance_matrix = cluster_distances(df)

with st.spinner("Computing 2D points..."):
    # Create MDS model to reduce the distance matrix to 2D
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    points_2d = mds.fit_transform(distance_matrix)

# Create an interactive plot using Plotly
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=points_2d[:, 0],
        y=points_2d[:, 1],
        mode="markers+text",
        text=[f"{get_cluster_topic(i)} ({i})" for i in range(len(points_2d))],
        textposition="top right",
        marker=dict(size=12, color="blue"),
    )
)

fig.update_layout(
    title="2D Interactive Visualization of Cluster Distances",
    xaxis_title="Dimension 1",
    yaxis_title="Dimension 2",
    template="plotly_white",
)

st.plotly_chart(fig)
