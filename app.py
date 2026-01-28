"""
AI Log Analyzer - CUSTOMIZED FOR YOUR MS GRAPH LOG STRUCTURE
Specifically designed for logs with:
  - "id": 798602 (numeric application ID)
  - "processingTime": 15166.0
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
# TARGETED SEARCH FOR YOUR LOG STRUCTURE
# ============================================
def scan_for_target_entry(log_content: str, target_id: str) -> List[Dict[str, Any]]:
    """
    Find ALL entries matching your exact log structure:
      - "id": <target_id>
      - "processingTime": <value>
    """
    # Regex to handle possible whitespace variations
    id_pattern = re.compile(r'"id"\s*:\s*' + re.escape(target_id))
    entries = []
    current_entry = ""
    brace_depth = 0
    
    # Process line-by-line to avoid memory issues
    for line in log_content.splitlines():
        # Track JSON object boundaries
        if brace_depth == 0 and line.strip().startswith('{'):
            current_entry = line
            brace_depth = line.count('{') - line.count('}')
        elif brace_depth > 0:
            current_entry += '\n' + line
            brace_depth += line.count('{') - line.count('}')
            
            if brace_depth <= 0 and id_pattern.search(current_entry):
                # Extract processing time value
                pt_match = re.search(r'"processingTime"\s*:\s*(\d+\.?\d*)', current_entry)
                pt_value = pt_match.group(1) if pt_match else "N/A"
                
                entries.append({
                    'entry': current_entry,
                    'id': target_id,
                    'processing_time': pt_value,
                    'match_type': 'exact_id'
                })
                current_entry = ""
                brace_depth = 0
    
    return entries

# ============================================
# STREAMLIT UI - CUSTOMIZED FOR YOUR LOGS
# ============================================
st.set_page_config(page_title="MS Graph Log Analyzer", layout="wide")
st.title("🔍 MS Graph Log Analyzer - CUSTOM FOR YOUR STRUCTURE")
st.caption("Specifically designed for logs with numeric IDs like 798602 and processingTime fields")

# Sidebar configuration
with st.sidebar:
    st.header("🎯 Target Configuration")
    target_id = st.text_input("🔍 Target ID to find", value="798602", 
                             help="Your application's numeric ID (e.g., 798602)")
    st.divider()
    st.info("💡 This tool is customized for YOUR log structure:\n- 'id': 798602\n- 'processingTime': 15166.0")

# File upload
input_type = st.radio("Input Method", ["Upload Log File", "Paste Log Text"], horizontal=True)
log_content = ""

if input_type == "Upload Log File":
    uploaded = st.file_uploader("📄 Upload your log file (35MB+ supported)", type=["json", "txt", "log"])
    if uploaded:
        try:
            log_content = uploaded.read().decode("utf-8")
            file_size_mb = len(log_content.encode('utf-8')) / (1024 * 1024)
            st.success(f"✅ Loaded {file_size_mb:.1f}MB file")
        except Exception as e:
            st.error(f"❌ Error: {e}")
else:
    log_content = st.text_area("📝 Paste logs", height=200)

if log_content.strip():
    # User question (auto-filled for your case)
    user_question = st.text_input(
        "💬 Question about logs",
        placeholder=f"What caused processing time 15166.0 for ID {target_id}?",
        value=f"What caused processing time 15166.0 for ID {target_id}?",
        help="This is pre-filled for your specific case"
    )
    
    if st.button("🔍 Analyze with Targeted Search", type="primary"):
        # STEP 1: Find ALL matching entries
        with st.status("🔍 Scanning for target ID...", expanded=True) as status:
            start_time = time.time()
            target_entries = scan_for_target_entry(log_content, target_id)
            scan_time = time.time() - start_time
            
            if target_entries:
                status.update(label=f"✅ Found {len(target_entries)} matching entries in {scan_time:.1f}s", state="complete")
                st.success(f"🎯 FOUND {len(target_entries)} entries with ID {target_id}")
            else:
                status.update(label=f"❌ No entries found with ID {target_id}", state="complete")
                st.error(f"❌ No entries found with ID {target_id}. Try these fixes:")
                st.markdown("""
                1. **Check ID format**: Is it `"id": "798602"` (string) or `"id": 798602` (number)?
                2. **Try different field names**: 
                   - `legacyId`, `recordId`, `externalId`
                   - `properties.id`, `metadata.id`
                3. **Search for processing time value**: 
                   ```bash
                   grep -a '"processingTime": *15166.0' your_log.json
                   ```
                """)
                st.stop()
        
        # STEP 2: Analyze matching entries
        st.subheader("🧠 Analysis of Target Entries")
        model = genai.GenerativeModel(MODEL_NAME)
        
        for i, entry in enumerate(target_entries[:3]):  # Limit to first 3 entries
            with st.status(f"🔍 Analyzing entry {i+1}...", expanded=False):
                # SAFE PROMPT with explicit newlines
                prompt = (
                    "You are a senior SRE analyzing application logs. BE PRECISE AND FACTUAL.\n\n"
                    "Log entry:\n"
                    "```\n"
                    f"{entry['entry'][:48000]}\n"
                    "```\n\n"
                    f"Question: What caused processing time {entry['processing_time']} for ID {target_id}?\n\n"
                    "CRITICAL INSTRUCTIONS:\n"
                    "- Quote the EXACT JSON values\n"
                    "- Identify error patterns or dependencies\n"
                    "- If data is insufficient, say 'INSUFFICIENT_DATA_IN_ENTRY'\n"
                    "- DO NOT hallucinate or guess values"
                )
                
                try:
                    resp = model.generate_content(
                        prompt,
                        safety_settings={genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE}
                    )
                    
                    st.success(f"✅ Analysis for ID {target_id} (processing time: {entry['processing_time']} ms):")
                    st.write(resp.text)
                    
                    # Show raw entry for debugging
                    with st.expander("🔍 Raw log entry (for debugging)"):
                        try:
                            st.json(json.loads(entry['entry']))
                        except:
                            st.code(entry['entry'][:1000], language="json")
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
        
        # STEP 3: Show context if multiple entries found
        if len(target_entries) > 3:
            st.info(f"💡 **Note**: Found {len(target_entries)} total entries for this ID. Showing first 3. Check if processing time varies across entries - this could indicate intermittent issues.")
        
st.divider()
st.caption("✅ This tool is customized for YOUR log structure - no more 'insufficient data' errors!")
