import db_utils
import streamlit as st

st.set_page_config(
    page_title="Home",
    page_icon="👋",
)

# Initialize session state if not set
if "channel" not in st.session_state:
    st.session_state.channel = None


# debugging
st.write("## Debugging")
st.write(f"**Session state**: {st.session_state}")

st.write("# Telegram Data Clustering")

with st.form("channel_selector"):
    channel = st.selectbox("Channel:", db_utils.get_channel_names(), help="Select a channel to view the data")
    selection_button = st.form_submit_button("Select")


if selection_button:
    # save the selected channel in the session state
    st.session_state["channel"] = channel

    # Display the selected channel
    st.write(f"Selected channel: {channel}")
    channel_info = db_utils.get_channel_info(channel)
    st.write(f"**Number of messages**: {channel_info['messages']}")
    st.write(f"**Channel created**: {channel_info['channel_created']}")
    st.write(f"""**Channel description**: {channel_info["description"]}""")

    # create histogram of messages and use loading spinner
    st.write("## Channel activity over time")
    histogram = db_utils.get_channel_message_histogram(channel)
    st.line_chart(histogram.set_index("month"), x_label="Month", y_label="Number of messages")
