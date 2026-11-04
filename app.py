"""Streamlit chat UI. Run with: streamlit run app.py"""
import streamlit as st

from rag import answer

st.set_page_config(page_title="Medical Chatbot", page_icon="🩺")
st.title("🩺 Medical Chatbot")
st.caption("Answers come from the documents it was given. Not a substitute for a doctor.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if q := st.chat_input("Ask a medical question"):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("Searching the documents..."):
            reply, hits = answer(q)
        st.markdown(reply)
        with st.expander("Sources"):
            for h in hits:
                st.markdown(f"**{h['source']}**, page {h['page']} (score {h['score']:.2f})")
                st.caption(h["text"][:300] + "...")
    st.session_state.messages.append({"role": "assistant", "content": reply})
