import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# --------------------------------------------------
# Load OpenAI API Key
# --------------------------------------------------
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY not found. Please check your .env file.")
    st.stop()

client = OpenAI(api_key=api_key)

# --------------------------------------------------
# Model Configuration
# --------------------------------------------------
MODEL = "gpt-5.4-nano"

# GPT-5.4 nano pricing per 1 Million Tokens
INPUT_PRICE_PER_1M = 0.2
OUTPUT_PRICE_PER_1M = 1.25


# --------------------------------------------------
# Streamlit Page
# --------------------------------------------------
st.set_page_config(
    page_title="LLM Token Usage Demo"
)

st.title("Customer Support CB")
st.caption(
    "LLM Token Usage & API Cost Demo - Developed by @GenAIWithDilip"
)


# --------------------------------------------------
# Initialize Session State
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_input_tokens" not in st.session_state:
    st.session_state.session_input_tokens = 0

if "session_output_tokens" not in st.session_state:
    st.session_state.session_output_tokens = 0

if "session_total_tokens" not in st.session_state:
    st.session_state.session_total_tokens = 0

if "session_total_cost" not in st.session_state:
    st.session_state.session_total_cost = 0.0

if "api_call_count" not in st.session_state:
    st.session_state.api_call_count = 0


# --------------------------------------------------
# Display Previous Conversation
# --------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# --------------------------------------------------
# User Input
# --------------------------------------------------
prompt = st.chat_input(
    "Ask a customer support question..."
)


if prompt:

    # --------------------------------------------------
    # Store User Message
    # --------------------------------------------------
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.write(prompt)


    # --------------------------------------------------
    # Developer Prompt
    # --------------------------------------------------
    conversation = [
        {
            "role": "developer",
            "content": """
                You are a customer support assistant for ABC Electronics.

                You help customers with:
                - Orders
                - Shipping
                - Returns
                - Refunds

                Keep your answers short, clear and helpful.
            """
        }
    ]

    # Add previous conversation summary instead of full history
    # if st.session_state.conversation_summary:

    #     conversation.append(
    #         {
    #             "role": "developer",
    #             "content": f"""
    #                 Previous Conversation Summary:

    #                 {st.session_state.conversation_summary}
    #             """
    #         }
    #     )
    # --------------------------------------------------
    # Add Complete Conversation History
    # --------------------------------------------------
    conversation.extend(
        st.session_state.messages
    )


    # --------------------------------------------------
    # Show Exactly What Our Application Sends
    # --------------------------------------------------
    with st.expander("Input Sent to OpenAI API"):

        st.json(conversation)


    # --------------------------------------------------
    # Call OpenAI API
    # --------------------------------------------------
    try:

        response = client.responses.create(
            model=MODEL,
            input=conversation
        )

        answer = response.output_text


        # --------------------------------------------------
        # Actual Token Usage Reported by OpenAI
        # --------------------------------------------------
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        total_tokens = response.usage.total_tokens


        # --------------------------------------------------
        # Cached Tokens - if any
        # --------------------------------------------------
        cached_tokens = 0

        if response.usage.input_tokens_details:
            cached_tokens = (
                response.usage.input_tokens_details.cached_tokens
                or 0
            )


        # --------------------------------------------------
        # Calculate Cost for THIS API Call
        #
        # This simplified demo calculates all input tokens
        # using the normal input rate.
        # --------------------------------------------------
        input_cost = (
            input_tokens / 1_000_000
        ) * INPUT_PRICE_PER_1M

        output_cost = (
            output_tokens / 1_000_000
        ) * OUTPUT_PRICE_PER_1M

        current_call_cost = (
            input_cost + output_cost
        )


        # --------------------------------------------------
        # Add Current Call to Session Totals
        # --------------------------------------------------
        st.session_state.session_input_tokens += input_tokens

        st.session_state.session_output_tokens += output_tokens

        st.session_state.session_total_tokens += total_tokens

        st.session_state.session_total_cost += current_call_cost

        st.session_state.api_call_count += 1


        # --------------------------------------------------
        # Store Assistant Response
        # --------------------------------------------------
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


        # --------------------------------------------------
        # Display Assistant Response
        # --------------------------------------------------
        with st.chat_message("assistant"):
            st.write(answer)


        # ==================================================
        # CURRENT API CALL
        # ==================================================
        st.divider()

        st.subheader("📊 Current API Call")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Input Tokens",
            f"{input_tokens:,}"
        )

        col2.metric(
            "Output Tokens",
            f"{output_tokens:,}"
        )

        col3.metric(
            "Total Tokens",
            f"{total_tokens:,}"
        )

        col4.metric(
            "Call Cost",
            f"${current_call_cost:.6f}"
        )


        # --------------------------------------------------
        # Additional Token Information
        # --------------------------------------------------
        st.write(
            f"Cached Input Tokens: **{cached_tokens:,}**"
        )


        # ==================================================
        # COMPLETE SESSION TOTAL
        # ==================================================
        st.subheader("📈 Complete Session Usage")

        session_col1, session_col2, session_col3, session_col4 = (
            st.columns(4)
        )

        session_col1.metric(
            "Session Input",
            f"{st.session_state.session_input_tokens:,}"
        )

        session_col2.metric(
            "Session Output",
            f"{st.session_state.session_output_tokens:,}"
        )

        session_col3.metric(
            "Session Total",
            f"{st.session_state.session_total_tokens:,}"
        )

        session_col4.metric(
            "Session Cost",
            f"${st.session_state.session_total_cost:.6f}"
        )


        st.caption(
            f"API Calls in this session: "
            f"{st.session_state.api_call_count}"
        )



    # --------------------------------------------------
    # Error Handling
    # --------------------------------------------------
    except Exception as e:

        st.error(
            f"OpenAI API Error: {e}"
        )



