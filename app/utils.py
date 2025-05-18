import logging
import os

import numpy as np
import tiktoken
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI(
    api_key=os.getenv("OPENAI_API"),
)


class Summarization(BaseModel):
    summary: str


class KeywordExtraction(BaseModel):
    keywords: list[str]


class LocationsAndDates(BaseModel):
    locations_dates: list[str]


class MainTopic(BaseModel):
    main_topic: str


class Country(BaseModel):
    country: str


encoding = tiktoken.encoding_for_model("gpt-4o-mini")


def fill_context(texts, context_limit=128000 * 0.75, separater="msg"):
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    tokens = 0
    for text in texts:
        tokens += len(encoding.encode(text))
    needed_context = np.ceil(tokens / context_limit)

    # sort texts by tokens descending
    texts = sorted(texts, key=lambda x: len(encoding.encode(x)), reverse=True)

    contexts = [[] for _ in range(int(needed_context))]
    indexes = [i for i in range(int(needed_context))]
    while texts:
        np.random.shuffle(indexes)
        for i in indexes:
            if len(texts) == 0:
                break
            text = texts.pop(0)
            contexts[i].append(text)
        if len(texts) == 0:
            break

    stats = {}
    for i, context in enumerate(contexts):
        s = 0
        t = 0
        for text in context:
            s += len(encoding.encode(text))
            t += 1
        stats[f"context_{i}"] = (s, t)

    for context in contexts:
        np.random.shuffle(context)

    start_sep = f"<{separater}>"
    end_sep = f"</{separater}>"
    texts = ""
    for i, context in enumerate(contexts):
        for text in context:
            texts += f"{start_sep}{text}{end_sep}\n"
        contexts[i] = texts
        texts = ""

    return contexts, stats


def describe_cluster(cluster_messages):
    def summarize_texts(context_text: list) -> str:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert in multilingual content analysis and intelligence 
gathering. Summarize the following non-English messages, which
are enclosed within <msg></msg> tags, into one concise and informative
paragraph in English for intelligence reporting. Prioritize names, 
locations, and dates while ensuring key topics and insights are retained.
Try to include the most important topics from the texts.
Avoid redundancy by merging overlapping information into a single 
coherent statement. Maintain a neutral and objective tone, 
focusing only on factual content.""",
                },
                {"role": "user", "content": f"Inserted texts: \n{context_text}"},
            ],
            response_format=Summarization,
            temperature=0.3,
            max_completion_tokens=700,
        )
        return completion.choices[0].message.parsed.summary

    def sumarization_of_sumarizations(responses):
        contexts, stats = fill_context(responses, separater="summary")
        summaries = []
        for context in contexts:
            completion = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """
Prompt:
You are an expert in content synthesis and intelligence analysis. Your task
is to combine the following summaries into a single, cohesive summary that
captures recurring themes, key events, and critical details while
eliminating redundancy. Ensure that the final summary:
- Synthesizes information to highlight overarching trends, connections,
and central ideas.
- Prioritizes significant events, locations, names, and key phrases
while preserving accuracy.
- Removes redundant or overlapping content, ensuring clarity and
conciseness.
Maintains a neutral, objective tone while delivering a structured and
well-organized output.The input summaries are enclosed in
<summary></summary> tags. The final output should be a single, concise
paragraph of plain text.""",
                    },
                    {"role": "user", "content": f"Summaries: \n{context}"},
                ],
                response_format=Summarization,
                temperature=0.3,
                max_completion_tokens=1000,
            )

            response = completion.choices[0].message.parsed
            summaries.append(response.summary)
        return summaries

    def topic_modeling(texts):
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """
You are an expert text analyzer. Your task is to determine the main topic of the following text. 
Please provide the topic in one concise sentence or phrase, focusing on the core subject or idea 
discussed. Avoid including unnecessary details or interpretations""",
                },
                {"role": "user", "content": f"Text:\n {texts}"},
            ],
            response_format=MainTopic,
        )
        return completion.choices[0].message.parsed.main_topic

    cluster_texts = cluster_messages
    contexts, stats = fill_context(cluster_texts, separater="msg")
    responses = []
    for context in contexts:
        response = summarize_texts(context)
        responses.append(response)

    while len(responses) > 1:
        responses = sumarization_of_sumarizations(responses)
    description = responses[0]
    topic = None  # topic_modeling(description)
    return description, topic


def extract_locations_from_clusters(texts):
    def locations_and_dates(texts):
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """Extract the most important countries from the following texts. 
                    If there are no countries return an empty list. Output countries must be in english.""",
                },
                {"role": "user", "content": f"Texts:\n {texts}"},
            ],
            response_format=LocationsAndDates,
        )

        return completion.choices[0].message.parsed.locations_dates

    contexts, stats = fill_context(texts, separater="msg")
    responses = []
    for context in contexts:
        response = locations_and_dates(context)
        responses.append(response)

    responses = [item for sublist in responses for item in sublist]

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """Return at most TOP 10 locations from the following list of locations. 
                Output must be in english.""",
            },
            {"role": "user", "content": f"Locations:\n {responses}"},
        ],
        response_format=KeywordExtraction,
    )
    return completion.choices[0].message.parsed.keywords


def describe_channel(texts):
    def summarize_texts(context: list) -> str:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert in multilingual content analysis and intelligence 
gathering. Summarize the following Russian-language messages, which
are enclosed within <msg></msg> tags, into one concise and informative
paragraph in English for intelligence reporting. Prioritize names, 
locations, and dates while ensuring key topics and insights are retained.
Avoid redundancy by merging overlapping information into a single 
coherent statement. Maintain a neutral and objective tone, 
focusing only on factual content.""",
                },
                {"role": "user", "content": f"Inserted texts: \n{context}"},
            ],
            response_format=Summarization,
            temperature=0.3,
            max_completion_tokens=800,
        )
        return completion.choices[0].message.parsed.summary

    def sumarization_of_sumarizations(responses):
        contexts, stats = fill_context(responses, context_limit=128000 * 0.6, separater="summary")
        summaries = []
        for context in contexts:
            completion = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """
Prompt:
You are an expert in content synthesis and intelligence analysis. Your task
is to combine the following summaries into a single, cohesive summary that
captures recurring themes, key events, and critical details while
eliminating redundancy. Ensure that the final summary:
- Synthesizes information to highlight overarching trends, connections,
and central ideas.
- Prioritizes significant events, locations, names, and key phrases
while preserving accuracy.
- Removes redundant or overlapping content, ensuring clarity and
conciseness.
Maintains a neutral, objective tone while delivering a structured and
well-organized output.The input summaries are enclosed in
<summary></summary> tags. The final output should be a single, concise
paragraph of plain text.""",
                    },
                    {"role": "user", "content": f"Summaries: \n{context}"},
                ],
                response_format=Summarization,
                temperature=0.3,
                max_completion_tokens=800,
            )

            response = completion.choices[0].message.parsed
            summaries.append(response.summary)
        return summaries

    logging.info("describing channel - going to fill context, messages length: %s", len(texts))
    contexts, _ = fill_context(texts, separater="msg")
    logging.info("describing channel - filled context, contexts length: %s", len(contexts))

    logging.info("describing channel - starting first stage of summarization")
    responses = []
    for context in contexts:
        response = summarize_texts(context)
        responses.append(response)
    logging.info("describing channel - finished first stage of summarization")

    while len(responses) > 1:
        logging.info("describing channel - second stage, responses length: %s", len(responses))
        responses = sumarization_of_sumarizations(responses)
    description = responses[0]
    logging.info("describing channel - finished second stage of summarization")
    return description


def extract_message_country(text):
    system_prompt = """
Extract the most important country from the following text; texts are non-english. If there are 
no clear country names, return an empty string. Output must be in English.
If there are multiple countries, return the most important one. Return just the country name.
"""

    user_prompt = f"Text:\n{text}"
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=Country,
        temperature=0.5,
    )
    country = completion.choices[0].message.parsed.country
    return country


def generate_keywords(text):
    system_prompt = """
    Extract the most important keywords from the following text.
    Use English language, return 4 to 7 keywords or phrases. Do not use names of people and point the most important topics.
    Keywords should be relevant to the text and should not be too general.
    You can use words directly from the text, but also you can use synonyms or phrases that are not in the text
    and are more suitable for the context. Sort the keywords by importance.
    """

    user_prompt = f"Text:\n{text}"
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=KeywordExtraction,
        temperature=0.5,
    )
    keywords = completion.choices[0].message.parsed.keywords
    return keywords


def batch_jobs_monitor(create_date):
    import datetime

    batches = client.batches.list(limit=50)
    batches_today = []
    for batch in batches:
        created_at = datetime.datetime.fromtimestamp(batch.created_at)
        if created_at.date() == create_date:
            id = batch.id
            status = batch.status
            completed = batch.request_counts.completed
            total = batch.request_counts.total
            batch_info = {
                "id": id,
                "status": status,
                "completed": completed,
                "total": total,
                "endpoint": batch.endpoint,
                "metadata": batch.metadata,
            }
            if status == "completed":
                batch_info["output_file"] = batch.output_file_id
            batches_today.append(batch_info)
    return batches_today
