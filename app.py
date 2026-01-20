import streamlit as st
import google.generativeai as genai
import os

# Configure Gemini using Streamlit Secrets (not .env)
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("❌ Google API key not found. Set it in Streamlit Cloud secrets.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# Use the model you want (update if gemini-2.0-flash-001 becomes available)
MODEL_NAME = "gemini-2.0-flash-001"  

st.set_page_config(page_title="AI Log Analyzer", layout="wide")
st.title("🔍 AI Log Analyzer with Gemini")
st.caption(f"Powered by **{MODEL_NAME}**")

# Input method
input_type = st.radio("Input Method", ["Upload Log File", "Paste Log Text"], horizontal=True)

log_content = ""
if input_type == "Upload Log File":
    uploaded = st.file_uploader("📄 Upload .log, .txt, or similar", type=["log", "txt", "out"])
    if uploaded:
        try:
            log_content = uploaded.read().decode("utf-8")
        except Exception as e:
            st.error(f"Error reading file: {e}")
else:
    log_content = st.text_area("📝 Paste your logs here:", height=200, placeholder="Jan 01 12:00:00 server ERROR: Connection timeout...")

# Truncate to avoid token limits (Gemini-1.5-flash supports ~1M tokens, but long logs slow UI)
MAX_CHARS = 50000
if len(log_content) > MAX_CHARS:
    log_content = log_content[:MAX_CHARS]
    st.warning(f"⚠️ Log truncated to {MAX_CHARS:,} characters for performance.")

if log_content.strip():
    with st.expander("📋 Preview (first 1000 chars)"):
        st.code(log_content[:1000], language="text")

    user_question = st.text_input("💬 Ask a question about the logs:", 
                                  placeholder="e.g., What are the top 3 errors?")

    if user_question.strip():
        with st.spinner("🧠 Analyzing with Gemini..."):
            try:
                model = genai.GenerativeModel(MODEL_NAME)
                prompt = f"""You are a senior DevOps/SRE engineer.
Log data:
