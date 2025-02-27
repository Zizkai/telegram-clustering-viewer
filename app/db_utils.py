import pandas as pd
import psycopg2
import streamlit as st


def get_connection(name="database"):
    db_credentials = st.secrets[name]
    conn = psycopg2.connect(
        dbname=db_credentials["dbname"],
        user=db_credentials["user"],
        password=db_credentials["password"],
        host=db_credentials["host"],
        port=db_credentials["port"],
    )
    return conn


def get_channel_names():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT channel FROM messages")
    channels = cur.fetchall()
    cur.close()
    conn.close()
    ret = [channel[0] for channel in channels]
    ret = sorted(ret)
    return ret


def get_cluster_ids(channel, clustering_method="kmeans", embedding_method="text-embeddings-3-small"):
    if channel is None:
        return []
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
                SELECT DISTINCT cluster_id 
                FROM clustering 
                WHERE channel = %s AND clustering_method = %s
                    AND embedding_method = %s
                """,
        (channel, clustering_method, embedding_method),
    )
    cluster_ids = cur.fetchall()
    cur.close()
    conn.close()
    ret = [cluster[0] for cluster in cluster_ids]
    ret = sorted(ret)
    return ret


def get_messages_by_cluster(
    channel, cluster_id, clustering_method="kmeans", embedding_method="text-embeddings-3-small"
):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
            SELECT m.id as id, m.date as date, 
                m.text_en as text_en, m.text as text
            FROM messages m
            INNER JOIN clustering c
            ON (m.id = c.id AND m.channel = c.channel) 
            WHERE c.cluster_id = %s
                AND c.channel = %s
                AND c.clustering_method = %s
                AND c.embedding_method = %s""",
        (cluster_id, channel, clustering_method, embedding_method),
    )
    messages = cur.fetchall()
    cur.close()
    conn.close()
    messages = pd.DataFrame(messages, columns=["id", "date", "text_en", "text"])
    return messages


def get_channel_messages(channel, columns=None):
    if columns is None:
        columns = ["id", "channel", "text"]
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
            SELECT 
                *
            FROM messages 
                WHERE channel like %s""",
        (channel,),
    )
    messages = cur.fetchall()
    cur.close()
    conn.close()

    messages = pd.DataFrame(
        messages,
        columns=["id", "text", "text_en", "channel", "lang", "views", "text_original", "date", "entities", "hashtags"],
    )
    messages = messages[columns]
    return messages


def get_number_of_msg(channel):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM messages WHERE channel = %s", (channel,))
    count = cur.fetchone()[0]
    cur.close()
    conn.close()
    return count


def get_channel_first_msg(channel):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT MIN(date) FROM messages WHERE channel = %s", (channel,))
    first_msg = cur.fetchone()[0]
    cur.close()
    conn.close()
    return first_msg


def get_channel_info(channel):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT description, messages, channel_created FROM channels WHERE channel = %s", (channel,))
    info = cur.fetchone()
    cur.close()
    conn.close()
    if info is None:
        return {
            "description": "No description available",
            "messages": "No messages available",
            "channel_created": "No data available",
        }
    info = {"description": info[0], "messages": info[1], "channel_created": info[2]}
    return info


def get_channel_message_histogram(channel):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT 
            DATE_TRUNC('month', date) AS month, 
            COUNT(*) AS message_count
        FROM messages
        WHERE channel = %s
        GROUP BY month
        ORDER BY month;
        """,
        (channel,),
    )
    histogram = cur.fetchall()
    cur.close()
    conn.close()
    histogram = pd.DataFrame(histogram, columns=["month", "message_count"])
    return histogram


def get_cluster_embeddings(channel, clustering_method="kmeans", embedding_method="text-embeddings-3-small"):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT 
            c.id, e.embedding, c.cluster_id
        FROM clustering c
        INNER JOIN embeddings e
        ON (c.id = e.id AND c.channel = e.channel)
        WHERE c.channel = %s
            AND c.clustering_method = %s
            AND c.embedding_method = %s
        """,
        (channel, clustering_method, embedding_method),
    )
    embeddings = cur.fetchall()

    cur.close()
    conn.close()

    embeddings = pd.DataFrame(embeddings, columns=["id", "embedding", "cluster_id"])
    return embeddings