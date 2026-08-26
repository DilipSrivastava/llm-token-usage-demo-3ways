import os

import streamlit as st
import pandas as pd

from openai import OpenAI
from dotenv import load_dotenv


# ============================================================
# LOAD .ENV
# ============================================================

load_dotenv()


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="LLM Token Usage Experiment",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "gpt-5.4-nano"

INPUT_PRICE_PER_1M = 0.2
OUTPUT_PRICE_PER_1M = 1.25

SUMMARY_EVERY = 5


SYSTEM_PROMPT = """
You are a helpful customer support assistant.

Answer the user's questions using the conversation context.

Be accurate and concise.
Keep your answers under 80 words.
"""


# ============================================================
# QUESTIONS
# ============================================================

QUESTIONS = [
    "I placed an order for a laptop yesterday.",
     "The order number is ORD-1001.",
     "The laptop is a 16-inch model with 16GB RAM.",
     "I paid 1200 dollars for it.",
     "I paid using my credit card.",
    "I was told the delivery would take five days.",
    "The tracking page currently says delayed.",
    "Can you tell me my order number?",
    "What laptop did I order?",
    "How much did I pay?",
    "What payment method did I use?",
    "What was the original delivery estimate?",
    "Why is my order delayed?",
    "Can I change the delivery address?",
    "The new delivery address should be my office.",
    "My office is open from 9 AM to 6 PM.",
    "Please remember that delivery preference.",
    "I also want accidental damage protection.",
    "I prefer email communication.",
    "Give me a final summary of everything about my order."
]


# ============================================================
# SESSION STATE
# ============================================================

if "results_available" not in st.session_state:
    st.session_state.results_available = False

if "full_df" not in st.session_state:
    st.session_state.full_df = None

if "summary_df" not in st.session_state:
    st.session_state.summary_df = None


# ============================================================
# COST
# ============================================================

def calculate_cost(input_tokens, output_tokens):

    return (
        (input_tokens / 1_000_000) * INPUT_PRICE_PER_1M
        +
        (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_1M
    )


# ============================================================
# OPENAI CLIENT
# ============================================================

api_key = os.getenv("OPENAI_API_KEY")

if api_key:

    client = OpenAI(
        api_key=api_key
    )

else:

    client = None


# ============================================================
# LLM CALL
# ============================================================

def call_llm(messages):

    if client is None:

        raise Exception(
            "OPENAI_API_KEY is not available."
        )

    return client.responses.create(
        model=MODEL,
        input=messages
    )


# ============================================================
# FULL HISTORY
# ============================================================

def run_full_history(progress):

    history = []

    results = []

    total_input = 0
    total_output = 0

    for turn, question in enumerate(
        QUESTIONS,
        start=1
    ):

        history.append(
            {
                "role": "user",
                "content": question
            }
        )

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

        messages.extend(history)

        response = call_llm(
            messages
        )

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        total_input += input_tokens
        total_output += output_tokens

        history.append(
            {
                "role": "assistant",
                "content": response.output_text
            }
        )

        results.append(
            {
                "Turn": turn,
                "Question": question,
                "Input Tokens": input_tokens,
                "Output Tokens": output_tokens,
                "Total Tokens": (
                    input_tokens +
                    output_tokens
                ),
                "Cumulative Tokens": (
                    total_input +
                    total_output
                ),
                "Cumulative Cost": calculate_cost(
                    total_input,
                    total_output
                )
            }
        )

        progress.progress(
            turn / len(QUESTIONS)
        )

    return pd.DataFrame(results)


# ============================================================
# GENERATE SUMMARY : to generate a summary of the conversation history
# ============================================================

def generate_summary(
    old_summary,
    history
):

    conversation = ""

    for message in history:

        conversation += (
            f"{message['role']}: "
            f"{message['content']}\n"
        )

    prompt = f"""
You maintain memory for a customer support conversation.

Existing summary:

{old_summary}

Conversation:

{conversation}

Create a compact summary containing only
information important for future questions.

Keep:

- order details
- requirements
- preferences
- important facts
- decisions

Remove:

- repetition
- greetings
- unnecessary explanations

Return ONLY the summary.
"""

    return call_llm(
        [
            {
                "role": "user",
                "content": prompt
            }
        ]
    )


# ============================================================
# SMART SUMMARY : to run the smart summary strategy
# ============================================================

def run_smart_summary(progress):

    recent_history = []

    summary = ""

    results = []

    total_input = 0
    total_output = 0

    total_summary_input = 0
    total_summary_output = 0

    for turn, question in enumerate(
        QUESTIONS,
        start=1
    ):

        recent_history.append(
            {
                "role": "user",
                "content": question
            }
        )

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

        if summary:

            messages.append(
                {
                    "role": "system",
                    "content":
                    "Conversation memory:\n\n"
                    + summary
                }
            )

        messages.extend(
            recent_history
        )

        response = call_llm(
            messages
        )

        main_input = response.usage.input_tokens
        main_output = response.usage.output_tokens

        total_input += main_input
        total_output += main_output

        recent_history.append(
            {
                "role": "assistant",
                "content": response.output_text
            }
        )

        summary_input = 0
        summary_output = 0

        if turn % SUMMARY_EVERY == 0:

            summary_response = generate_summary(
                summary,
                recent_history
            )

            summary = (
                summary_response.output_text
            )

            summary_input = (
                summary_response
                .usage
                .input_tokens
            )

            summary_output = (
                summary_response
                .usage
                .output_tokens
            )

            total_summary_input += summary_input
            total_summary_output += summary_output

            recent_history = recent_history[-2:]

        results.append(
            {
                "Turn": turn,
                "Question": question,

                "Main Input Tokens":
                    main_input,

                "Main Output Tokens":
                    main_output,

                "Summary Input Tokens":
                    summary_input,

                "Summary Output Tokens":
                    summary_output,

                "Total Tokens":
                    (
                        main_input
                        + main_output
                        + summary_input
                        + summary_output
                    ),

                "Cumulative Tokens":
                    (
                        total_input
                        + total_output
                        + total_summary_input
                        + total_summary_output
                    ),

                "Cumulative Cost":
                    calculate_cost(
                        total_input + total_summary_input,
                        total_output + total_summary_output
                    )
            }
        )

        progress.progress(
            turn / len(QUESTIONS)
        )

    return pd.DataFrame(results)


# ============================================================
# HEADER
# ============================================================

st.title(
    "📊 LLM Token Usage Experiment"
)

st.write(
    """
## Does sending the entire conversation every time
increase token usage?

This experiment sends the **same 20 questions**
through two different conversation-management
strategies.
"""
)


# ============================================================
# API STATUS
# ============================================================

# if client:

#     st.success(
#         "🟢 OpenAI API Key Loaded"
#     )

# else:

#     st.warning(
#         "🟡 OpenAI API Key not loaded. "
#         "The UI is still available, but the experiment "
#         "cannot run until the API key is configured."
#     )


# ============================================================
# STRATEGIES
# ============================================================

col1, col2 = st.columns(2)

with col1:

    st.info(
        """
### 🔵 Full History

Every request receives:

**System Prompt + Entire Conversation**

The context becomes larger after every turn.
"""
    )


with col2:

    st.success(
        """
### 🟢 Smart Summary

Older conversation is summarized.

The LLM receives:

**System Prompt + Summary + Recent Messages**
"""
    )


# ============================================================
# EXPERIMENT INFORMATION
# ============================================================

st.divider()

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Questions",
        len(QUESTIONS)
    )

with col2:

    st.metric(
        "Model",
        MODEL
    )

with col3:

    st.metric(
        "Summary Every",
        f"{SUMMARY_EVERY} turns"
    )

with col4:

    st.metric(
        "Strategies",
        "2"
    )


# ============================================================
# RUN BUTTON
# ============================================================

st.divider()

# st.subheader(
#     "🚀 Run Experiment"
# )

# st.write(
#     "Click the button below to start the experiment."
# )

run_experiment = st.button(
    "🚀 Run Experiment",
    key="run_experiment_button",
    type="primary",
    use_container_width=True
)


# ============================================================
# EXPERIMENT :  to run the experiment if the button is clicked
# ============================================================

if run_experiment:

    if client is None:

        st.error(
            """
OpenAI API key is not available.

Please configure OPENAI_API_KEY and restart Streamlit.
"""
        )

        st.stop()


    # ========================================================
    # FULL HISTORY : to run the full history strategy
    # ========================================================

    st.divider()

    st.subheader(
        "🔵 Running Full History"
    )

    full_progress = st.progress(
        0
    )

    try:

        full_df = run_full_history(
            full_progress
        )

    except Exception as e:

        st.error(
            "Full History experiment failed."
        )

        st.exception(e)

        st.stop()

    st.success(
        "✅ Full History completed"
    )


    # ========================================================
    # SMART SUMMARY : to run the smart summary strategy
    # ========================================================

    st.subheader(
        "🟢 Running Smart Summary"
    )

    summary_progress = st.progress(
        0
    )

    try:

        summary_df = run_smart_summary(
            summary_progress
        )

    except Exception as e:

        st.error(
            "Smart Summary experiment failed."
        )

        st.exception(e)

        st.stop()

    st.success(
        "✅ Smart Summary completed"
    )


    # ========================================================
    # SAVE : to save the results in session state for later use
    # ========================================================

    st.session_state.full_df = full_df

    st.session_state.summary_df = summary_df

    st.session_state.results_available = True


# ============================================================
# RESULTS : to show the results of the experiment if available
# ============================================================

if st.session_state.results_available:

    full_df = st.session_state.full_df

    summary_df = st.session_state.summary_df


    # ========================================================
    # FULL TOTALS : to calculate the total tokens and cost for the full history strategy
    # ========================================================

    full_input = int(
        full_df["Input Tokens"].sum()
    )

    full_output = int(
        full_df["Output Tokens"].sum()
    )

    full_total = (
        full_input +
        full_output
    )

    full_cost = calculate_cost(
        full_input,
        full_output
    )


    # ========================================================
    # SUMMARY TOTALS : to calculate the total tokens and cost for the smart summary strategy
    # ========================================================

    summary_main_input = int(
        summary_df[
            "Main Input Tokens"
        ].sum()
    )

    summary_main_output = int(
        summary_df[
            "Main Output Tokens"
        ].sum()
    )

    summary_input = int(
        summary_df[
            "Summary Input Tokens"
        ].sum()
    )

    summary_output = int(
        summary_df[
            "Summary Output Tokens"
        ].sum()
    )

    summary_total = (
        summary_main_input
        + summary_main_output
        + summary_input
        + summary_output
    )

    summary_cost = calculate_cost(
        summary_main_input + summary_input,
        summary_main_output + summary_output
    )


    # ========================================================
    # FINAL RESULTS: to show the final results in a table
    # ========================================================

    st.divider()

    st.header(
        "🏆 Final Results"
    )


    comparison = pd.DataFrame(
        {
            "Metric": [
                "Input Tokens",
                "Output Tokens",
                "Summary Tokens",
                "TOTAL TOKENS",
                "TOTAL COST"
            ],

            "🔵 Full History": [
                f"{full_input:,}",
                f"{full_output:,}",
                "0",
                f"{full_total:,}",
                f"${full_cost:.6f}"
            ],

            "🟢 Smart Summary": [
                f"{summary_main_input:,}",
                f"{summary_main_output:,}",
                f"{summary_input + summary_output:,}",
                f"{summary_total:,}",
                f"${summary_cost:.6f}"
            ]
        }
    )

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # METRICS : to show the metrics for each strategy in a side-by-side layout 
    # ========================================================

    # col1, col2 = st.columns(2)

    # with col1:

    #     st.subheader(
    #         "🔵 Full History"
    #     )

    #     st.metric(
    #         "Input Tokens",
    #         f"{full_input:,}"
    #     )

    #     st.metric(
    #         "Output Tokens",
    #         f"{full_output:,}"
    #     )

    #     st.metric(
    #         "Total Tokens",
    #         f"{full_total:,}"
    #     )

    #     st.metric(
    #         "Cost",
    #         f"${full_cost:.6f}"
    #     )


    # with col2:

    #     st.subheader(
    #         "🟢 Smart Summary"
    #     )

    #     st.metric(
    #         "Main Input Tokens",
    #         f"{summary_main_input:,}"
    #     )

    #     st.metric(
    #         "Summary Input Tokens",
    #         f"{summary_input:,}"
    #     )

    #     st.metric(
    #         "Output Tokens",
    #         f"{summary_main_output + summary_output:,}"
    #     )

    #     st.metric(
    #         "Total Tokens",
    #         f"{summary_total:,}"
    #     )

    #     st.metric(
    #         "Cost",
    #         f"${summary_cost:.6f}"
    #     )


    # ========================================================
    # WINNER : to show which strategy used fewer tokens and cost
    # ========================================================

    st.divider()

    st.header(
        "🎯 Experiment Result"
    )

    if summary_total < full_total:

        saving = (
            (full_total - summary_total)
            / full_total
        ) * 100

        cost_saving = (
            (full_cost - summary_cost)
            / full_cost
        ) * 100

        st.success(
            f"""
## 🟢 Smart Summary Wins

**Full History:** {full_total:,} tokens

**Smart Summary:** {summary_total:,} tokens

### Token Reduction

**{saving:.2f}%**

### Cost Reduction

**{cost_saving:.2f}%**
"""
        )

    else:

        overhead = (
            (summary_total - full_total)
            / full_total
        ) * 100

        st.warning(
            f"""
## 🔵 Full History Used Fewer Tokens

**Full History:** {full_total:,} tokens

**Smart Summary:** {summary_total:,} tokens

### Summary Overhead

**{overhead:.2f}%**

The conversation may not yet be large enough
for summarization to offset the summary-generation
cost.
"""
        )


    # ========================================================
    # CUMULATIVE GRAPH : to show the cumulative token usage for each turn in a graph
    # ========================================================

    st.divider()

    st.header(
        "📈 Cumulative Token Usage"
    )

    graph = pd.DataFrame(
        {
            "🔵 Full History":
                full_df[
                    "Cumulative Tokens"
                ].values,

            "🟢 Smart Summary":
                summary_df[
                    "Cumulative Tokens"
                ].values
        }
    )

    graph.index = range(
        1,
        len(QUESTIONS) + 1
    )

    st.line_chart(
        graph
    )


    # ========================================================
    # INPUT TOKEN GRAPH : to show the input token usage for each turn in a graph
    # ========================================================

    st.header(
        "📊 Input Tokens Per Turn"
    )

    input_graph = pd.DataFrame(
        {
            "🔵 Full History":
                full_df[
                    "Input Tokens"
                ].values,

            "🟢 Smart Summary":
                summary_df[
                    "Main Input Tokens"
                ].values
        }
    )

    input_graph.index = range(
        1,
        len(QUESTIONS) + 1
    )

    st.line_chart(
        input_graph
    )


    # ========================================================
    # TURN BY TURN : to show the token usage for each turn in a table
    # ========================================================

    st.divider()

    st.header(
        "🔍 Turn-by-Turn Token Usage"
    )

    turn_data = pd.DataFrame(
        {
            "Turn":
                range(
                    1,
                    len(QUESTIONS) + 1
                ),

            "Question":
                QUESTIONS,

            "Full History Input":
                full_df[
                    "Input Tokens"
                ].values,

            "Smart Summary Input":
                summary_df[
                    "Main Input Tokens"
                ].values,

            "Summary Input":
                summary_df[
                    "Summary Input Tokens"
                ].values,

            "Full History Cumulative":
                full_df[
                    "Cumulative Tokens"
                ].values,

            "Smart Summary Cumulative":
                summary_df[
                    "Cumulative Tokens"
                ].values
        }
    )

    st.dataframe(
        turn_data,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # QUESTIONS  : To Sow all the questions asked in the experiment
    # ========================================================

    st.divider()

    with st.expander(
        "📝 View the 20 questions"
    ):

        for i, question in enumerate(
            QUESTIONS,
            start=1
        ):

            st.write(
                f"**{i}.** {question}"
            )