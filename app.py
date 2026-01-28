"""
AI Log Analyzer - ROBUST MODE
Works with ANY log format: NDJSON, minified JSON, mixed text, nested structures
"""
import streamlit as st
import google.generativeai as genai
import re
import time
import json
from typing import List, Dict, Any
from datetime import datetime

# ============================================
# CONFIGURATION
# ============================================
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("❌ Google API key not found. Add to Streamlit Cloud secrets as 'GOOGLE_API_KEY'.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
MODEL_NAME = "gemini-2.0-flash-001"

# ============================================
# ROBUST SCANNER - WORKS WITH ANY FORMAT
# ============================================
def robust_scan(log_content: str, target_id: str, target_value: str = "15166.0") -> List[Dict[str, Any]]:
    """
    Find entries using LITERAL STRING SEARCH (works with ANY format):
    - NDJSON (one JSON object per line)
    - Minified JSON (single massive line)
    - Mixed text logs (timestamps + JSON)
    - Nested structures
    """
    results = []
    
    # Strategy 1: Search for BOTH values together (highest confidence)
    combined_pattern = re.compile(
        rf'.{{0,200}}{re.escape(target_id)}.{{0,200}}{re.escape(target_value)}.{{0,200}}',
        re.IGNORECASE
    )
    
    # Strategy 2: Search for value alone (fallback)
    value_pattern = re.compile(
        rf'.{{0,300}}{re.escape(target_value)}.{{0,300}}',
        re.IGNORECASE
    )
    
    # Try combined search first
    matches = combined_pattern.findall(log_content)
    
    if not matches:
        # Fallback to value-only search
        matches = value_pattern.findall(log_content)
        if matches:
            st.warning(f"⚠️ ID {target_id} not found near value. Showing entries with processing time {target_value} only.")
    
    # Extract clean JSON snippets from matches
    for i, match in enumerate(matches[:5]):  # Limit to 5 matches
        # Try to extract valid JSON object from surrounding text
        json_obj = extract_json_object(match)
        
        results.append({
            'raw_context': match,
            'json_object': json_obj,
            'match_type': 'combined' if target_id in match else 'value_only',
            'index': i
        })
    
    return results


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Extract first valid JSON object from messy text (handles timestamps/prefixes)
    """
    # Find first { ... } pair that looks like JSON
    start = text.find('{')
    if start == -1:
        return {}
    
    brace_count = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                try:
                    return json.loads(text[start:i+1])
                except:
                    # Try to clean/repair JSON
                    cleaned = text[start:i+1].replace('\n', ' ').replace('\r', ' ')
                    try:
                        return json.loads(cleaned)
                    except:
                        return {"raw_snippet": text[start:i+1][:500]}
    
    return {}


# ============================================
# STREAMLIT UI - ROBUST MODE
# ============================================
st.set_page_config(page_title="MS Graph Log Analyzer - Robust", layout="wide")
st.title("🔍 MS Graph Log Analyzer - ROBUST MODE")
st.caption("Works with ANY log format: NDJSON, minified JSON, mixed text logs")

# Sidebar configuration
with st.sidebar:
    st.header("🎯 Search Configuration")
    target_id = st.text_input("🔍 Target ID", value="798602", help="Your numeric ID")
    target_value = st.text_input("⏱️ Processing time value", value="15166.0", help="e.g., 15166.0")
    st.divider()
    st.info("💡 This tool searches for LITERAL STRINGS – works with any log format!")

# File upload
input_type = st.radio("Input Method", ["Upload Log File", "Paste Log Text"], horizontal=True)
log_content = ""

if input_type == "Upload Log File":
    uploaded = st.file_uploader("📄 Upload your log file (35MB+ supported)", type=["json", "txt", "log"])
    if uploaded:
        try:
            log_content = uploaded.read().decode("utf-8", errors='ignore')
            file_size_mb = len(log_content.encode('utf-8', errors='ignore')) / (1024 * 1024)
            st.success(f"✅ Loaded {file_size_mb:.1f}MB file")
        except Exception as e:
            st.error(f"❌ Error: {e}")
else:
    log_content = st.text_area("📝 Paste logs", height=200)

if log_content.strip():
    # Quick existence check
    with st.status("🔍 Checking if values exist in file...", expanded=False) as status:
        id_exists = target_id in log_content
        value_exists = target_value in log_content
        
        if id_exists and value_exists:
            status.update(label=f"✅ Both ID {target_id} and value {target_value} FOUND in file", state="complete")
            st.success(f"✅ Confirmed: ID `{target_id}` and value `{target_value}` exist in this file")
        elif value_exists:
            status.update(label=f"⚠️ Value {target_value} found, but ID {target_id} NOT found", state="complete")
            st.warning(f"⚠️ Value `{target_value}` found, but ID `{target_id}` NOT found. Showing all entries with this processing time.")
        else:
            status.update(label=f"❌ Neither ID nor value found in first 10MB of file", state="complete")
            st.error(f"❌ Neither `{target_id}` nor `{target_value}` found in scanned portion of file.")
            st.info("""
            🔍 **Diagnostic steps:**
            1. Run locally: `grep -a "15166.0" your_log.json`
            2. Check if file is truncated or from wrong time period
            3. Try searching for variations: `15166` (without decimal), `"processingTime"`
            """)
            st.stop()
    
    user_question = st.text_input(
        "💬 Question about logs",
        value=f"What caused processing time {target_value} for ID {target_id}?",
        help="Customize your question"
    )
    
    if st.button("🔍 Analyze with Robust Search", type="primary"):
        # STEP 1: Find matching entries
        with st.status("🔍 Extracting relevant log entries...", expanded=True) as status:
            start_time = time.time()
            matches = robust_scan(log_content, target_id, target_value)
            scan_time = time.time() - start_time
            
            if matches:
                status.update(label=f"✅ Extracted {len(matches)} relevant entries in {scan_time:.1f}s", state="complete")
                st.success(f"🎯 Found {len(matches)} log entries containing your target data")
            else:
                st.error("❌ No relevant entries extracted. Try adjusting search terms.")
                st.stop()
        
        # STEP 2: Show extracted entries for transparency
        st.subheader("📊 Extracted Log Entries")
        for match in matches:
            with st.expander(f"Entry {match['index'] + 1} ({match['match_type']})", expanded=True):
                if match['json_object']:
                    st.json(match['json_object'])
                else:
                    st.code(match['raw_context'][:800] + ("..." if len(match['raw_context']) > 800 else ""), language="json")
        
        # STEP 3: Analyze with Gemini
        st.subheader("🧠 AI Analysis")
        model = genai.GenerativeModel(MODEL_NAME)
        
        # Build analysis prompt from ALL matches
        context_snippets = []
        for i, match in enumerate(matches):
            snippet = match['json_object'] if match['json_object'] else match['raw_context'][:500]
            context_snippets.append(f"Entry {i+1}:\n{json.dumps(snippet, indent=2) if isinstance(snippet, dict) else snippet}")
        
        full_context = "\n\n---\n\n".join(context_snippets[:3])  # Limit to 3 entries
        
        prompt = (
            "You are a senior SRE analyzing application logs. BE PRECISE AND FACTUAL.\n\n"
            "Relevant log entries:\n"
            "```\n"
            f"{full_context}\n"
            "```\n\n"
            f"Question: {user_question}\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "- Quote EXACT field values from the logs\n"
            "- Identify root cause patterns (timeouts, retries, errors)\n"
            "- If insufficient data, say 'INSUFFICIENT_DATA_IN_PROVIDED_SNIPPETS'\n"
            "- DO NOT hallucinate values not present in the logs\n"
            "- Focus on operation type (e.g., MailAttachments), error codes, and timing patterns"
        )
        
        with st.spinner("🧠 Analyzing with Gemini..."):
            try:
                resp = model.generate_content(
                    prompt,
                    safety_settings={genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE}
                )
                st.write(resp.text)
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.info("💡 Common fixes: Check API quota, try simpler question, or reduce context size")
        
        # Export option
        with st.expander("📥 Export Results"):
            export_data = {
                'question': user_question,
                'target_id': target_id,
                'target_value': target_value,
                'matches_found': len(matches),
                'entries': [m['json_object'] if m['json_object'] else m['raw_context'][:1000] for m in matches],
                'timestamp': datetime.now().isoformat()
            }
            st.download_button(
                label="Download Analysis Results (JSON)",
                data=json.dumps(export_data, indent=2),
                file_name=f"log_analysis_{target_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

st.divider()
st.caption("✅ Robust mode: Searches for literal strings – works with NDJSON, minified JSON, mixed logs, and nested structures")
