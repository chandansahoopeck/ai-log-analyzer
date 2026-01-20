import streamlit as st
import google.generativeai as genai

# Configure Gemini using Streamlit Secrets
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("❌ Google API key not found. Please add it to Streamlit Cloud secrets under 'GOOGLE_API_KEY'.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# Use gemini-1.5-flash for now; update when gemini-2.0-flash-001 is available
MODEL_NAME = "gemini-2.0-flash-001"

st.set_page_config(page_title="AI Log Analyzer", layout="wide")
st.title("🔍 AI Log Analyzer with Gemini")
st.caption(f"Powered by **{MODEL_NAME}**")

# Input method
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

# Truncate long logs
MAX_CHARS = 50000
if len(log_content) > MAX_CHARS:
    log_content = log_content[:MAX_CHARS]
    st.warning(f"⚠️ Log truncated to {MAX_CHARS:,} characters for responsiveness.")

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
                
                # SAFE PROMPT: No triple quotes → no SyntaxError
                prompt = (
                    "You are a senior DevOps/SRE engineer analyzing system logs.\n"
                    "Log content:\n"
                    "```\n"
                    + log_content + "\n"
                    "```\n\n"
                    "Question: " + user_question + "\n\n"
                    "Instructions:\n"
                    "- Be concise, technical, and accurate.\n"
                    "- Reference specific lines or patterns if they exist.\n"
                    "- If the log does not contain enough information to answer, say so clearly.\n"
                    "- Do not make up facts."
                )

                response = model.generate_content(prompt)
                
                if not response.text:
                    st.warning("⚠️ Gemini returned an empty response. The content may have been blocked.")
                else:
                    st.subheader("🤖 AI Analysis")
                    st.write(response.text)

            except Exception as e:
                st.error(f"❌ Error calling Gemini API: {str(e)}")
                st.info("💡 Make sure:\n- Your API key is valid\n- The model name is correct\n- You have quota remaining")
