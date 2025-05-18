import pandas as pd
import psycopg2
import streamlit as st


def get_connection(name="database") -> psycopg2.extensions.connection:
    """
    Returns a connection to the database using the credentials stored in Streamlit secrets (secrets.toml).
    The credentials should be stored in the following format:
    [database]
    dbname = <database_name>
    user = <user_name>
    password = <password>
    host = <host>
    port = <port>

    :param name: The name of the database credentials in Streamlit secrets.
    :return: A connection to the database.
    """
    db_credentials = st.secrets[name]
    conn = psycopg2.connect(
        dbname=db_credentials["dbname"],
        user=db_credentials["user"],
        password=db_credentials["password"],
        host=db_credentials["host"],
        port=db_credentials["port"],
    )
    return conn


def get_clustering_info(channel: str) -> dict[str, list[int]]:
    """
    Returns a DataFrame with the clustering information about given channel
    :param channel: The name of the channel.
    :return: dict.
    """

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT 
            channel, num_clusters, silhouette, dbi, inter
        FROM clustering_info
        WHERE channel = %s
        """,
        (channel,),
    )
    clustering_info = cur.fetchall()
    cur.close()
    conn.close()
    clustering_info = clustering_info[0]
    clustering_info = {
        "num_clusters": clustering_info[1],
        "silhouette": list(map(float, clustering_info[2])),
        "dbi": list(map(float, clustering_info[3])),
        "inter": list(map(float, clustering_info[4])) if clustering_info[4] is not None else None,
    }

    return clustering_info


def get_channel_names() -> list[str]:
    """
    Returns a list of distinct channel names from the channels table in the database.
    :return: A list of distinct channel names.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT channel FROM channels")
    channels = cur.fetchall()
    cur.close()
    conn.close()
    ret = [channel[0] for channel in channels]
    ret = sorted(ret)
    return ret


def llm_judge_channels() -> list[str]:
    """
    Returns a list of distinct channel names from the llm_as_a_judge_texts table in the database.
    :return: A list of distinct channel name.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT channel FROM llm_as_a_judge_texts")
    clusters = cur.fetchall()
    cur.close()
    conn.close()
    ret = [cluster[0] for cluster in clusters]
    ret = sorted(ret)
    return ret


def get_llm_judge_text_data(channel: str) -> pd.DataFrame:
    """
    Return a Dataframe with the content of the llm_as_a_judge_texts table for a given channel.
    :param channel: The name of the channel.
    :return: A DataFrame with the content of the llm_as_a_judge_texts table for a given channel.
    """

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT 
            id, channel, anchor, text1, text2, anchor_translation, text1_translation, text2_translation
        FROM llm_as_a_judge_texts
        WHERE channel = %s
        """,
        (channel,),
    )
    texts = cur.fetchall()
    cur.close()
    conn.close()

    texts = pd.DataFrame(
        texts,
        columns=[
            "id",
            "channel",
            "anchor",
            "text1",
            "text2",
            "anchor_translation",
            "positive_translation",
            "negative_translation",
        ],
    )

    return texts


def get_cluster_description(channel: str, cluster_id: int) -> pd.DataFrame:
    """
    Returns a DataFrame with the description of the cluster for a given channel and cluster ID.
    :param channel: The name of the channel.
    :param cluster_id: The ID of the cluster.
    :return: A DataFrame with the description of the cluster.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT summary, keywords FROM cluster_summaries
        WHERE channel = %s AND cluster_id = %s
        """,
        (channel, cluster_id),
    )
    description = cur.fetchall()
    cur.close()
    conn.close()

    description = pd.DataFrame(description, columns=["summary", "keywords"])

    return description if not description.empty else None


def get_llm_judge_decision_data(channel, llm_model) -> pd.DataFrame:
    """
    Return a Dataframe with the content of the llm_as_a_judge_decision table for a given channel.
    :param channel: The name of the channel.
    :param llm_model: The name of the LLM model.
    :return: A DataFrame with the content of the llm_as_a_judge_decision table for a given channel.
    """

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT 
            id, channel, llm_model, reasoning, decision, correct_decision
        FROM llm_as_a_judge_decisions
        WHERE channel = %s AND llm_model = %s
        """,
        (channel, llm_model),
    )
    decisions = cur.fetchall()
    cur.close()
    conn.close()

    decisions = pd.DataFrame(
        decisions,
        columns=["id", "channel", "llm_model", "reasoning", "decision", "correct_decision"],
    )

    return decisions


def get_cluster_ids(channel: str) -> list[int]:
    """
    Returns a sorted list of distinct cluster IDs for a given channel from the clustering table in the database.
    :param channel: The name of the channel.
    :return: A sorted list of distinct cluster IDs.
    """

    def get_benchmark_data():
        query = f"""
        SELECT DISTINCT cluster_id FROM benchmark_clustering 
        WHERE channel = '{channel}'
        """
        return query

    if channel is None:
        return []
    elif "Benchmark" in channel:
        query_select = get_benchmark_data()
    else:
        query_select = f"""
        SELECT DISTINCT cluster_id 
        FROM clustering 
        WHERE channel = '{channel}'
        """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(query_select)
    cluster_ids = cur.fetchall()
    cur.close()
    conn.close()
    ret = [cluster[0] for cluster in cluster_ids]
    ret = sorted(ret)
    return ret


def get_messages_by_cluster(channel: str, cluster_id: id) -> pd.DataFrame:
    """
    Returns a DataFrame of messages for a given cluster ID from the messages table in the database.
    :param channel: The name of the channel.
    :param cluster_id: The ID of the cluster.

    :return: A DataFrame of messages for the given cluster ID. Columns are: ["id", "date", "text_en", "text"]
    """

    conn = get_connection()
    cur = conn.cursor()
    if "Benchmark" in channel:
        query = """
            SELECT m.id as id, m.date as date, 
                m.text_en as text_en, m.text_original as text
            FROM messages m
            INNER JOIN benchmark_clustering c
            ON (m.id = c.msg_id AND m.channel = c.channel_msg) 
            WHERE c.cluster_id = %s
                AND c.channel = %s"""
        cur.execute(query, (cluster_id, channel))
    else:
        query = """
            SELECT m.id as id, m.date as date, 
                m.text_en as text_en, m.text_original as text
            FROM messages m
            INNER JOIN clustering c
            ON (m.id = c.id AND m.channel = c.channel) 
            WHERE c.cluster_id = %s
                AND c.channel = %s
            """
        cur.execute(query, (cluster_id, channel))
    messages = cur.fetchall()
    cur.close()
    conn.close()
    messages = pd.DataFrame(messages, columns=["id", "date", "text_en", "text"])
    return messages


def get_channel_messages(channel: str, columns: list[str] | None = None) -> pd.DataFrame:
    """
    Returns a DataFrame of messages for a given channel from the messages table in the database.
    :param channel: The name of the channel.
    :param columns: The columns to return. If None, returns ["id", "channel", "text"] columns.
    """
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


def get_number_of_msg(channel: str) -> int:
    """
    Returns the number of messages in the database for a given channel.
    :param channel: The name of the channel.
    :return: The number of messages in the database for the given channel.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM messages WHERE channel = %s", (channel,))
    count = cur.fetchone()[0]
    cur.close()
    conn.close()
    return count


def get_channel_first_msg(channel: str) -> str:
    """
    Returns the date of the first message in the database for a given channel.
    :param channel: The name of the channel.
    :return: The date of the first message in the database for the given channel.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT MIN(date) FROM messages WHERE channel = %s", (channel,))
    first_msg = cur.fetchone()[0]
    cur.close()
    conn.close()
    return first_msg


def get_channel_info(channel: str) -> dict[str, str | int]:
    """
    Returns the description, number of messages, and channel creation date for a given channel.
    :param channel: The name of the channel.
    :return: A dictionary with the description, number of messages, and channel creation date.
    """
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


def get_channel_message_histogram(channel: str) -> pd.DataFrame:
    """
    Returns a histogram of the number of messages per month for a given channel.
    The histogram is created by truncating the date to the month and counting the number of messages for each month.
    The computation is done in the database for efficiency.

    :param channel: The name of the channel.
    :return: A DataFrame with the month and the number of messages.
    """

    def get_benchmark_data():
        import re

        id = int(re.search(r"^Benchmark (\d+)", channel).group(1))
        query = f"""
        SELECT 
            DATE_TRUNC('month', date) AS month, 
            COUNT(*) AS message_count
        FROM (
            SELECT msg.date FROM messages as msg
            INNER JOIN benchmark_data_map as bdm ON msg.id = bdm.msg_id 
                                AND msg.channel = bdm.channel_msg 
                                AND bdm.benchmark_id = {id}
        )
        GROUP BY month
        ORDER BY month;
        """
        return query

    conn = get_connection()
    cur = conn.cursor()
    if "Benchmark" in channel:
        query_select = get_benchmark_data()
    else:
        query_select = f"""
        SELECT 
            DATE_TRUNC('month', date) AS month, 
            COUNT(*) AS message_count
        FROM messages
        WHERE channel = '{channel}'
        GROUP BY month
        ORDER BY month;
        """
    cur.execute(query_select)
    histogram = cur.fetchall()
    cur.close()
    conn.close()
    histogram = pd.DataFrame(histogram, columns=["month", "message_count"])
    return histogram


def get_cluster_embeddings(channel: str):
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
        """,
        (channel,),
    )
    embeddings = cur.fetchall()

    cur.close()
    conn.close()

    embeddings = pd.DataFrame(embeddings, columns=["id", "embedding", "cluster_id"])
    return embeddings


def check_if_clustering_exists(channel: str) -> bool:
    """
    Check if clustering exists for a given channel in the database.
    :param channel: The name of the channel.
    :return: True if clustering exists, False otherwise.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT 
            COUNT(*)
        FROM (SELECT DISTINCT channel FROM clustering
              UNION
              SELECT DISTINCT channel FROM benchmark_clustering
        ) as c
        WHERE c.channel = %s
        """,
        (channel,),
    )
    count = cur.fetchone()[0]
    cur.close()
    conn.close()
    return count > 0


def get_clustering_keywords(channel: str) -> pd.DataFrame:
    """
    Returns a DataFrame of keywords for each cluster in the database for a given channel.
    :param channel: The name of the channel.
    :return: A DataFrame with the cluster ID and keywords.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT cluster_id, keywords FROM cluster_summaries
        WHERE channel = %s AND keywords IS NOT NULL
        """,
        (channel,),
    )
    keywords = cur.fetchall()
    cur.close()
    conn.close()

    keywords = pd.DataFrame(keywords, columns=["cluster_id", "keywords"])

    return keywords
