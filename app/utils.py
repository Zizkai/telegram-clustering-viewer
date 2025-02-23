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


encoding = tiktoken.encoding_for_model("gpt-4o-mini")


def fill_context(texts, context_limit=128000 * 0.65):
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
    for i, context in enumerate(contexts):
        contexts[i] = "\n\n\n".join(context)
    return contexts, stats


def describe_cluster(df, cluster_id):
    def summarize_texts(texts: list) -> str:
        text_str = ""
        for text in texts:
            text_str += f"<msg>{text}</msg>\n"
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
                {"role": "user", "content": f"Inserted texts: \n{texts}"},
            ],
            response_format=Summarization,
            temperature=0.3,
            max_completion_tokens=500,
        )
        return completion.choices[0].message.parsed.summary

    def sumarization_of_sumarizations(responses):
        contexts, stats = fill_context(responses)
        summaries = []
        prompt_text = ""
        for context in contexts:
            for text in context:
                prompt_text += f"<summary>{text}</summary>\n"
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
                    {"role": "user", "content": f"Summaries: \n{prompt_text}"},
                ],
                response_format=Summarization,
                temperature=0.3,
                max_completion_tokens=550,
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
You are an expert text analyzer. Your task is to determine the main topic of the following text. Please provide the topic in
one concise sentence or phrase, focusing on the core subject or idea discussed. Avoid including unnecessary details or interpretations""",
                },
                {"role": "user", "content": f"Text:\n {texts}"},
            ],
            response_format=MainTopic,
        )
        return completion.choices[0].message.parsed.main_topic

    cluster_texts = df[df["cluster"] == cluster_id]
    cluster_texts = cluster_texts["text"].tolist()
    contexts, stats = fill_context(cluster_texts)
    responses = []
    for context in contexts:
        response = summarize_texts(context)
        responses.append(response)

    while len(responses) > 1:
        responses = sumarization_of_sumarizations(responses)
    description = responses[0]
    topic = topic_modeling(description)
    return description, topic


def extract_keywords(text):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Extract the most important topics and location (Top 10) from the following text:",
            },
            {"role": "user", "content": text},
        ],
        response_format=KeywordExtraction,
    )

    return completion.choices[0].message.parsed.keywords


def extract_locations_from_clusters(df, cluster_id):
    def locations_and_dates(texts):
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Extract the most important locations from the following texts. Group them by regions. Output will be in english.",
                },
                {"role": "user", "content": f"Texts:\n {texts}"},
            ],
            response_format=LocationsAndDates,
        )

        return completion.choices[0].message.parsed.locations_dates

    cluster_texts = df[df["cluster"] == cluster_id]["text"].tolist()
    contexts, stats = fill_context(cluster_texts)
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
                "content": "Return at most TOP 10 locations from the following list of locations.",
            },
            {"role": "user", "content": f"Locations:\n {responses}"},
        ],
        response_format=KeywordExtraction,
    )
    return completion.choices[0].message.parsed.keywords
