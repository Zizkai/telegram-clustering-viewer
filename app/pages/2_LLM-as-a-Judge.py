import db_utils
import streamlit as st

st.set_page_config(
    page_title="LLM-as-a-Judge",
    page_icon="⚖️",
)


st.title("LLM-as-a-Judge")
st.write(
    """
    This page shows the results of the LLM-as-a-Judge experiment. 
    The goal of this experiment was to evaluate the clustering methods using a large language model (LLM) as a judge.

    You can see three text when one is marked as anchor text and the other two are the positive and negative sample.
    Below the texts you can see the LLM's reasoning and the descision. You can also switch between LLMs models.
    The LLMs used in this experiment are:
    - OpenAI GPT-4o-mini
    - OpenAI o4-mini (reasoning model)
    """
)

with st.expander("System prompt for LLMs:"):
    st.write("""
    **Role:**  
    You are an expert linguist specializing in semantic and meaning-based comparison of texts across different 
            languages.

    **Task:**  
    You will receive three texts written in the same language (any language, not necessarily English):  
    - One **Anchor Text**  
    - Two **Candidate Texts** (Text 1 and Text 2)

    Your objective is to determine which candidate text (**Text 1** or **Text 2**) is **semantically and 
            meaningfully closer** to the Anchor Text.  
    **The language of the texts does not matter — focus solely on meaning and content.**

    **Instructions:**  
    1. **Carefully read and analyze** the content of all three texts.  
    2. **Compare** the core ideas, themes, and meaning between the Anchor Text and each Candidate Text individually.  
    3. **Decide** which Candidate Text is closer in meaning to the Anchor Text.  
    4. **Do not add** any introduction text.
    5. **Output** should be analysis written in english and the decision about candidates -> Text 1 or Text 2

    **Input Format:**
    - **Anchor Text:** `[Insert text here]`
    - **Text 1:** `[Insert text here]`
    - **Text 2:** `[Insert text here]`
    """)

with st.form("channel_selector"):
    channel = st.selectbox("Channel:", db_utils.llm_judge_channels(), help="Select a channel to view the data")
    selection_button = st.form_submit_button("Select")

if selection_button:
    data_text = db_utils.get_llm_judge_text_data(channel)
    data_decision_reasoning = db_utils.get_llm_judge_decision_data(channel, "o4-mini")
    data_decision_inteligence = db_utils.get_llm_judge_decision_data(channel, "gpt-4.1-nano")

    for _, row in data_text.iterrows():
        tab1, tab2 = st.tabs(["Eng", "Original Text"])
        with tab1:
            st.write(f"**Anchor text:** {row['anchor_translation']}")
            st.write(f"**Text 1:** {row['positive_translation']}")
            st.write(f"**Text 2:** {row['negative_translation']}")
        with tab2:
            st.write(f"**Anchor text:** {row['anchor']}")
            st.write(f"**Text 1:** {row['text1']}")
            st.write(f"**Text 2:** {row['text2']}")

        llm_row_resoning = data_decision_reasoning[data_decision_reasoning["id"] == row["id"]].iloc[0]
        llm_row_intelligence = data_decision_inteligence[data_decision_inteligence["id"] == row["id"]].iloc[0]

        tab_llm_1, tab_llm_2 = st.tabs(["gpt-4.1-nano", "o4-mini"])
        with tab_llm_1:
            st.write(f"**LLM Reasoning** : {llm_row_intelligence['reasoning']}")
            st.write(f"**LLM Decision** : {llm_row_intelligence['decision']}")
            st.write(f"**Does answer corespond with the clustering?**: {llm_row_intelligence['correct_decision']}")
        with tab_llm_2:
            st.write(f"**LLM Reasoning** : {llm_row_resoning['reasoning']}")
            st.write(f"**LLM Decision** : {llm_row_resoning['decision']}")
            st.write(f"**Does answer corespond with the clustering?**: {llm_row_resoning['correct_decision']}")

        st.write("---")
