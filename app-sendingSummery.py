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

INPUT_PRICE_PER_1M = 0.2
OUTPUT_PRICE_PER_1M = 1.25


# --------------------------------------------------
# Streamlit Page
# --------------------------------------------------
st.set_page_config(page_title="LLM Token Optimization Demo")

st.title("Customer Support CB")
st.caption(
    "Token Usage + Conversation Summarization Demo - @GenAIWithDilip"
)


# --------------------------------------------------
# Initialize Session State
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_summary" not in st.session_state:
    st.session_state.conversation_summary = ""

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
prompt = st.chat_input("Ask a customer support question...")


if prompt:

    # Save current user message for UI only
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.write(prompt)


    # --------------------------------------------------
    # Build Optimized Input
    # --------------------------------------------------
    conversation = [
        {
            "role": "developer",
            "content": """
You are a customer support executive for ABC Electronics.

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
    if st.session_state.conversation_summary:

        conversation.append(
            {
                "role": "developer",
                "content": f"""
Previous Conversation Summary:

{st.session_state.conversation_summary}
"""
            }
        )


    # Add ONLY current user question
    conversation.append(
        {
            "role": "user",
            "content": prompt
        }
    )


    # --------------------------------------------------
    # Show Input Sent to OpenAI
    # --------------------------------------------------
    with st.expander("📤 Input Sent to OpenAI"):

        st.json(conversation)


    try:

        # --------------------------------------------------
        # Main LLM Call
        # --------------------------------------------------
        response = client.responses.create(
            model=MODEL,
            input=conversation
        )

        answer = response.output_text


        # --------------------------------------------------
        # Usage for Main Call
        # --------------------------------------------------
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        total_tokens = response.usage.total_tokens


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
        # Store Assistant Response
        # --------------------------------------------------
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


        with st.chat_message("assistant"):
            st.write(answer)


        # --------------------------------------------------
        # Update Running Summary
        # --------------------------------------------------
        summary_prompt = f"""
You are maintaining a short conversation summary.

Existing Summary:
{st.session_state.conversation_summary}

New User Message:
{prompt}

New Assistant Response:
{answer}

Update the summary.

Rules:
- Keep only important facts.
- Preserve order numbers, product names, dates, issues and decisions.
- Remove greetings and unnecessary wording.
- Keep the summary concise.
"""

        summary_response = client.responses.create(
            model=MODEL,
            input=summary_prompt
        )

        st.session_state.conversation_summary = (
            summary_response.output_text
        )


        # --------------------------------------------------
        # Add Summary Call Usage Also
        # --------------------------------------------------
        summary_input_tokens = (
            summary_response.usage.input_tokens
        )

        summary_output_tokens = (
            summary_response.usage.output_tokens
        )

        summary_total_tokens = (
            summary_response.usage.total_tokens
        )


        summary_input_cost = (
            summary_input_tokens / 1_000_000
        ) * INPUT_PRICE_PER_1M

        summary_output_cost = (
            summary_output_tokens / 1_000_000
        ) * OUTPUT_PRICE_PER_1M

        summary_cost = (
            summary_input_cost + summary_output_cost
        )


        # --------------------------------------------------
        # Session Totals
        #
        # Important:
        # We include BOTH:
        # 1. Main chatbot call
        # 2. Summary-generation call
        # --------------------------------------------------
        st.session_state.session_total_tokens += (
            total_tokens + summary_total_tokens
        )

        st.session_state.session_total_cost += (
            current_call_cost + summary_cost
        )

        st.session_state.api_call_count += 2


        # --------------------------------------------------
        # Display Current Call Usage
        # --------------------------------------------------
        st.divider()

        st.subheader("📊 Customer Support Call")

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
            "Cost",
            f"${current_call_cost:.6f}"
        )

        # --------------------------------------------------
        # Complete Session Usage
        # --------------------------------------------------
        st.subheader("📈 Complete Session Usage")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "API Calls",
            st.session_state.api_call_count
        )

        col2.metric(
            "Total Tokens",
            f"{st.session_state.session_total_tokens:,}"
        )

        col3.metric(
            "Session Cost",
            f"${st.session_state.session_total_cost:.6f}"
        )


    except Exception as e:

        st.error(
            f"OpenAI API Error: {e}"
        )


