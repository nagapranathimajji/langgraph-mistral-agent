# app.py

import streamlit as st
import asyncio
from agent_graph import process_user_input

st.set_page_config(page_title="LangGraph Chat Agent", layout="centered")
st.title("💬 LangGraph + Mistral Agent")

# Clear chat
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "agent" else "🧑"):
        st.markdown(msg["content"])

# Input box
if prompt := st.chat_input("Type something (e.g. `5 + 4`, `summarize: ...`, `translate: Hello`)"):
    # Save user input
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    # Agent response
    with st.chat_message("agent", avatar="🤖"):
        with st.spinner("Thinking..."):
            response = asyncio.run(process_user_input(prompt))
            st.markdown(response)

    # Save agent response
    st.session_state.messages.append({"role": "agent", "content": response})
