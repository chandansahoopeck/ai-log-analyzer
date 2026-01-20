import streamlit as st
import google.generativeai as genai
import os

# Configure Gemini using Streamlit Secrets
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("❌ Google API key not found. Please add it to Streamlit Cloud secrets under 'GOOGLE_API_KEY'.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

MODEL_NAME = "gemini-2.0-flash-001"  

st.set_page_config(page_title="AI Log Analyzer", layout="wide")
st.title("🔍 AI Log Analyzer with Gemini")
st.caption(f"Powered by **{MODEL_NAME}**")

# Input method selection
input_type = st.radio("Input Method", ["Upload Log File", "Paste Log Text"], horizontal=True)

log_content = ""
if input_type == "Upload Log File":
    uploaded = st.file_uploader("📄 Upload .log, .txt, or similar", type=["log", "txt", "out", "json"])
    if uploaded:
        try:
            log_content = uploaded.read().decode("utf-8")
        except UnicodeDecodeError:
            st.error("⚠️ Could not decode file. Please upload a UTF-8 text file.")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
else:
    log_content = st.text_area(
        "📝 Paste your logs here:",
        height=200,
        placeholder="Example:\nJan 01 12:00:00 server ERROR: Connection timeout\nJan 01 12:01:00 server INFO: Retrying..."
    )

# Truncate very long logs to avoid performance issues (Gemini can handle long context, but UI may lag)
MAX_CHARS = 50000
if len(log_content) > MAX_CHARS:
    log_content = log_content[:MAX_CHARS]
    st.warning(f"⚠️ Log truncated to {MAX_CHARS:,} characters for responsiveness.")

# Show preview if content exists
if log_content.strip():
    with st.expander("📋 Log Preview (first 1,000 characters)"):
        st.code(log_content[:1000], language="text")

    user_question = st.text_input(
        "💬 Ask a question about the logs:",
        placeholder="e.g., What are the most frequent errors? Are there any timeouts?"
    )

    if user_question.strip():
        with st.spinner("🧠 Analyzing with Gemini..."):
            try:
                model = genai.GenerativeModel(MODEL_NAME)
                
                # Use triple SINGLE quotes to avoid SyntaxError with f-strings
                prompt = f'''You are a senior DevOps/SRE engineer analyzing system logs.
Log content:
