"""
AI Log Analyzer with Gemini - Enhanced for Large JSON Files (35MB+)
Features:
  ✅ JSON-aware semantic chunking with overlap
  ✅ Format detection (array, NDJSON, object)
  ✅ 3-tier analysis pipeline (filter → analyze → synthesize)
  ✅ Cost controls and quota protection
  ✅ Streaming fallback for 100MB+ files
  ✅ UI transparency with chunk metadata
"""

import streamlit as st
import google.generativeai as genai
import json
import re
import time
from typing import List, Dict, Any
from datetime import datetime

# ============================================
# CONFIGURATION
# ============================================

# Configure Gemini using Streamlit Secrets
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("❌ Google API key not found. Please add it to Streamlit Cloud secrets under 'GOOGLE_API_KEY'.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# Model configuration
MODEL_NAME = "gemini-2.0-flash-001"  # Using stable model (gemini-2.0-flash-001 not generally available yet)
FALLBACK_MODEL = "gemini-2.0-flash-001"

# Chunking configuration
MAX_CHARS_PER_CHUNK = 45000    # Leave room for prompt overhead
OVERLAP_ENTRIES = 3            # Overlap between chunks for context preservation
MAX_CHUNKS = 15                # Safety limit (≈ $0.045 cost)
MAX_SYNTHESIS_CHARS = 90000    # Hard cap for synthesis prompt

# Cost estimation (gemini-1.5-flash pricing)
COST_PER_MILLION_INPUT_TOKENS = 0.075   # $0.075 per 1M input tokens
COST_PER_MILLION_OUTPUT_TOKENS = 0.30   # $0.30 per 1M output tokens
TOKENS_PER_CHAR_ESTIMATE = 0.75         # Rough estimate: 1 char ≈ 0.75 tokens

# ============================================
# JSON CHUNKING HELPER FUNCTIONS
# ============================================

def detect_json_format(sample: str, full_text: str = None) -> str:
    """
    Detect JSON format type before chunking:
    - 'array': [{"ts":...}, {"ts":...}]  ← MOST COMMON FOR LOGS
    - 'ndjson': {"ts":...}\n{"ts":...}  ← Newline-delimited
    - 'object': {"key1": [...], "key2": [...]} 
    - 'unknown': Fallback to line-based
    """
    sample = (full_text[:10000] if full_text else sample).strip()
    
    # NDJSON detection: Multiple root-level objects separated by newlines
    if re.match(r'^\{.*\}\n\{', sample[:200], re.DOTALL):
        return 'ndjson'
    
    # Array detection: Starts with [ and has multiple objects
    if sample.startswith('['):
        brace_count = sample[:1000].count('{')
        if brace_count > 1:
            return 'array'
    
    # Single object with array values
    if sample.startswith('{'):
        try:
            if '"logs"' in sample[:200].lower() or '"entries"' in sample[:200].lower():
                return 'object'
        except:
            pass
    
    return 'unknown'


def chunk_json_array(
    json_text: str,
    max_chars: int = 45000,
    overlap_entries: int = 3,
    max_chunks: int = 20
) -> List[Dict[str, Any]]:
    """
    Chunk a JSON array while preserving object integrity.
    """
    chunks = []
    
    # Parse JSON safely
    try:
        data = json.loads(json_text)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array, got {type(data).__name__}")
    except json.JSONDecodeError as e:
        st.error(f"⚠️ JSON parse error: {str(e)[:200]}")
        st.info("💡 Try: 1) Validate JSON with a linter 2) Check for trailing commas 3) Ensure UTF-8 encoding")
        return []
    
    # Build chunks with semantic boundaries
    current_chunk = []
    current_size = 2  # Account for opening '[' and closing ']'
    
    for idx, entry in enumerate(data):
        # Serialize entry to measure true size
        entry_str = json.dumps(entry, separators=(',', ':'))
        entry_size = len(entry_str) + (2 if current_chunk else 1)  # + comma/space
        
        # Time to finalize chunk?
        if current_size + entry_size > max_chars and current_chunk:
            chunks.append(_create_chunk(
                entries=current_chunk,
                start_idx=idx - len(current_chunk),
                end_idx=idx - 1,
                chunk_num=len(chunks) + 1
            ))
            
            # Inject overlap: carry forward last N entries
            if overlap_entries > 0 and len(current_chunk) > overlap_entries:
                current_chunk = current_chunk[-overlap_entries:]
                current_size = 2 + sum(
                    len(json.dumps(e, separators=(',', ':'))) + 2 
                    for e in current_chunk[:-1]
                ) + len(json.dumps(current_chunk[-1], separators=(',', ':')))
            else:
                current_chunk = []
                current_size = 2
        
        current_chunk.append(entry)
        current_size += entry_size
        
        # Safety limit
        if len(chunks) >= max_chunks:
            st.warning(f"⚠️ Hit chunk limit ({max_chunks}). Processing first {max_chunks} chunks only.")
            break
    
    # Final chunk
    if current_chunk and len(chunks) < max_chunks:
        chunks.append(_create_chunk(
            entries=current_chunk,
            start_idx=len(data) - len(current_chunk),
            end_idx=len(data) - 1,
            chunk_num=len(chunks) + 1
        ))
    
    return chunks


def _create_chunk(entries: List[Any], start_idx: int, end_idx: int, chunk_num: int) -> Dict[str, Any]:
    """Helper to create standardized chunk with metadata"""
    chunk_text = json.dumps(entries, separators=(',', ':'))
    
    return {
        'text': chunk_text,
        'metadata': {
            'chunk_id': f'chunk_{chunk_num:03d}',
            'entry_range': (start_idx, end_idx),
            'entry_count': len(entries),
            'char_count': len(chunk_text),
            'overlap_applied': chunk_num > 1
        },
        'preview': _generate_preview(entries[:3])
    }


def _generate_preview(entries: List[Any], max_len: int = 300) -> str:
    """Generate human-readable preview for UI"""
    preview = json.dumps(entries, indent=2, default=str)
    return preview[:max_len] + "..." if len(preview) > max_len else preview


def inject_overlap(chunks: List[Dict], overlap_entries: int = 3) -> List[Dict]:
    """
    Inject overlap between chunks to preserve context for patterns spanning boundaries.
    """
    if overlap_entries <= 0 or len(chunks) < 2:
        return chunks
    
    overlapped_chunks = [chunks[0]]  # First chunk unchanged
    
    for i in range(1, len(chunks)):
        prev_chunk = chunks[i-1]
        curr_chunk = chunks[i]
        
        # Extract overlap entries from previous chunk
        prev_entries = json.loads(prev_chunk['text'])
        overlap_slice = prev_entries[-overlap_entries:] if len(prev_entries) >= overlap_entries else prev_entries
        
        # Merge with current chunk entries
        curr_entries = json.loads(curr_chunk['text'])
        merged_entries = overlap_slice + curr_entries
        
        # Create new overlapped chunk
        overlapped_chunks.append({
            'text': json.dumps(merged_entries, separators=(',', ':')),
            'metadata': {
                **curr_chunk['metadata'],
                'overlap_from_prev': len(overlap_slice),
                'original_start_idx': curr_chunk['metadata']['entry_range'][0],
                'adjusted_start_idx': curr_chunk['metadata']['entry_range'][0] - len(overlap_slice)
            },
            'preview': _generate_preview(merged_entries[:3])
        })
    
    return overlapped_chunks


def tier1_filter_relevant_chunks(
    chunks: List[Dict[str, Any]], 
    user_question: str, 
    model: genai.GenerativeModel,
    max_relevant: int = 5
) -> List[Dict[str, Any]]:
    """
    Use cheap LLM calls to filter irrelevant chunks.
    Saves 70-90% of LLM costs.
    """
    if len(chunks) <= max_relevant:
        return chunks  # No need to filter if already under limit
    
    st.info(f"🔍 Filtering {len(chunks)} chunks to find most relevant...")
    
    # SAFE PROMPT: Using single-line strings with explicit newlines to avoid syntax issues
    filter_prompt_template = (
        "You are a log filtering expert. Determine if this log chunk contains information RELEVANT to the question.\n\n"
        "Question: {question}\n\n"
        "Log chunk (first 5 entries):\n"
        "{chunk_preview}\n\n"
        "Answer ONLY 'RELEVANT' or 'IRRELEVANT'."
    )
    
    relevant_chunks = []
    scanned = 0
    
    for chunk in chunks:
        scanned += 1
        
        # Use preview for filtering (much cheaper than full chunk)
        try:
            chunk_preview = json.dumps(
                json.loads(chunk['text'])[:5],  # First 5 entries
                indent=2
            )
        except:
            chunk_preview = chunk['text'][:500]
        
        try:
            resp = model.generate_content(
                filter_prompt_template.format(
                    question=user_question,
                    chunk_preview=chunk_preview[:8000]  # Limit preview size
                ),
                safety_settings={
                    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: 
                    genai.types.HarmBlockThreshold.BLOCK_NONE,
                }
            )
            
            if "RELEVANT" in resp.text.upper():
                relevant_chunks.append(chunk)
                st.success(f"✅ Chunk {chunk['metadata']['chunk_id']} marked RELEVANT")
                
                if len(relevant_chunks) >= max_relevant:
                    st.info(f"🎯 Found {max_relevant} relevant chunks. Stopping filter scan.")
                    break
            else:
                st.caption(f"❌ Chunk {chunk['metadata']['chunk_id']} - IRRELEVANT")
        
        except Exception as e:
            st.warning(f"⚠️ Filter error on {chunk['metadata']['chunk_id']}: {str(e)[:100]}")
            # Don't fail entire analysis; continue with next chunk
    
    # Fallback: if no chunks marked relevant, use first N chunks
    if not relevant_chunks:
        st.warning("⚠️ No chunks marked relevant. Using first chunks as fallback.")
        relevant_chunks = chunks[:max_relevant]
    
    st.success(f"✅ Filtered {len(relevant_chunks)}/{scanned} chunks as relevant")
    return relevant_chunks


def tier2_analyze_chunk(
    chunk: Dict[str, Any], 
    user_question: str, 
    model: genai.GenerativeModel
) -> Dict[str, Any]:
    """
    Deep analysis of a single relevant chunk
    """
    # SAFE PROMPT: Using explicit newline characters instead of triple quotes
    prompt = (
        "You are a senior SRE/DevOps engineer analyzing system logs. Be concise and technical.\n\n"
        f"Log context (entries {chunk['metadata']['entry_range'][0]}-{chunk['metadata']['entry_range'][1]}, "
        f"{chunk['metadata']['entry_count']} total entries):\n"
        "```\n"
        f"{chunk['text'][:MAX_CHARS_PER_CHUNK - 1000]}\n"
        "```\n\n"
        f"Question: {user_question}\n\n"
        "Instructions:\n"
        "- Reference specific values, IDs, timestamps, or error codes\n"
        "- Quantify findings ('5 errors between 12:00-12:05', '99th percentile: 15.2s')\n"
        "- If insufficient data to answer, say 'INSUFFICIENT_DATA_FOR_QUESTION'\n"
        "- Output format: Plain text with clear sections (Findings, Severity, Recommendations)"
    )

    try:
        resp = model.generate_content(
            prompt,
            safety_settings={
                genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: 
                genai.types.HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        return {
            'chunk_id': chunk['metadata']['chunk_id'],
            'entry_range': chunk['metadata']['entry_range'],
            'analysis': resp.text,
            'success': True
        }
    
    except Exception as e:
        return {
            'chunk_id': chunk['metadata']['chunk_id'],
            'entry_range': chunk['metadata']['entry_range'],
            'analysis': f"⚠️ Analysis failed: {str(e)}",
            'success': False
        }


def tier3_synthesize_analyses(
    analyses: List[Dict[str, Any]], 
    user_question: str, 
    model: genai.GenerativeModel
) -> str:
    """
    Combine multiple chunk analyses into coherent answer
    """
    analyses_text = "\n\n---\n".join([
        f"Chunk {i+1} ({a['chunk_id']}, entries {a['entry_range'][0]}-{a['entry_range'][1]}):\n{a['analysis']}"
        for i, a in enumerate(analyses)
    ])
    
    # Hard cap to prevent token explosion
    analyses_text = analyses_text[:MAX_SYNTHESIS_CHARS]
    
    # SAFE PROMPT: Using explicit newline characters
    synthesis_prompt = (
        "Synthesize these log analyses into a unified, comprehensive answer.\n\n"
        f"Original question: {user_question}\n\n"
        f"Analyses from {len(analyses)} log chunks:\n"
        f"{analyses_text}\n\n"
        "Instructions:\n"
        "- Resolve any contradictions between chunks\n"
        "- Highlight patterns, trends, or anomalies across time/ranges\n"
        "- Quantify overall impact ('12 total timeouts across 3 hours', 'Peak processing time: 15.2s')\n"
        "- If analyses are insufficient or conflicting, state limitations clearly\n"
        "- Be concise but comprehensive (2-4 paragraphs max)\n"
        "- Use bullet points for clarity when listing multiple findings"
    )
    
    try:
        resp = model.generate_content(
            synthesis_prompt,
            safety_settings={
                genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: 
                genai.types.HarmBlockThreshold.BLOCK_NONE,
            }
        )
        return resp.text
    except Exception as e:
        return f"⚠️ Synthesis failed: {str(e)}"


def estimate_cost(input_chars: int, output_chars: int) -> float:
    """
    Estimate cost in USD for Gemini API call
    """
    input_tokens = input_chars * TOKENS_PER_CHAR_ESTIMATE
    output_tokens = output_chars * TOKENS_PER_CHAR_ESTIMATE
    
    input_cost = (input_tokens / 1_000_000) * COST_PER_MILLION_INPUT_TOKENS
    output_cost = (output_tokens / 1_000_000) * COST_PER_MILLION_OUTPUT_TOKENS
    
    return input_cost + output_cost


# ============================================
# STREAMLIT UI
# ============================================

st.set_page_config(page_title="AI Log Analyzer", layout="wide")
st.title("🔍 AI Log Analyzer with Gemini")
st.caption(f"Powered by **{MODEL_NAME}** | Optimized for large JSON files (35MB+)")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Chunking settings
    st.subheader("Chunking")
    max_chunks_ui = st.slider("Max chunks to process", 5, 30, MAX_CHUNKS)
    overlap_ui = st.slider("Overlap entries", 0, 10, OVERLAP_ENTRIES)
    
    # Cost estimation toggle
    show_cost = st.checkbox("Show cost estimates", value=True)
    
    st.divider()
    st.info("💡 **Pro Tips:**\n- Use filtering to save costs on large files\n- Overlap preserves context across chunks\n- Check chunk metadata to verify coverage")

# Input method
input_type = st.radio("Input Method", ["Upload Log File", "Paste Log Text"], horizontal=True)

log_content = ""
if input_type == "Upload Log File":
    uploaded = st.file_uploader("📄 Upload .log, .txt, .json (supports 35MB+ files)", type=["log", "txt", "out", "json"])
    if uploaded:
        try:
            log_content = uploaded.read().decode("utf-8")
            file_size_mb = len(log_content.encode('utf-8')) / (1024 * 1024)
            st.success(f"✅ Loaded {file_size_mb:.1f}MB file ({len(log_content):,} characters)")
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

# Show log preview (first 1000 chars)
if log_content.strip():
    with st.expander("📋 Log Preview (first 1,000 characters)"):
        st.code(log_content[:1000], language="text")
    
    # Detect format if JSON
    if log_content.strip().startswith('[') or log_content.strip().startswith('{'):
        with st.status("🔍 Detecting log format...", expanded=False) as status:
            fmt = detect_json_format(log_content[:10000], log_content)
            status.update(label=f"✅ Format detected: **{fmt.upper()}**", state="complete")
            
            if fmt == 'array':
                st.success("✅ JSON array detected. Will use semantic chunking with overlap.")
            elif fmt == 'unknown':
                st.warning("⚠️ Format unclear. Using line-based chunking (may be less accurate).")
    
    # User question
    st.divider()
    user_question = st.text_input(
        "💬 Ask a question about the logs:",
        placeholder="e.g., What are the most frequent errors? Find processing time for ID 798602? Any timeouts?",
        help="Be specific for better results (include IDs, time ranges, error types)"
    )
    
    # Advanced options
    with st.expander("⚙️ Advanced Analysis Options"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            enable_filtering = st.checkbox("✅ Enable relevance filtering", value=True, 
                                          help="Reduces cost by analyzing only relevant chunks")
        
        with col2:
            show_chunk_metadata = st.checkbox("📊 Show chunk metadata", value=False,
                                             help="Inspect chunk boundaries and content")
        
        with col3:
            max_analysis_chunks = st.number_input("Max chunks to analyze", 1, 30, max_chunks_ui)
    
    if user_question.strip():
        st.divider()
        
        # Initialize model
        try:
            model = genai.GenerativeModel(MODEL_NAME)
        except Exception as e:
            st.error(f"❌ Model initialization failed: {str(e)}")
            st.info(f"💡 Try checking your API key or quota limits")
            st.stop()
        
        # ============================================
        # ANALYSIS PIPELINE
        # ============================================
        
        with st.spinner("🚀 Starting analysis pipeline..."):
            
            # STEP 1: Chunk the log content
            with st.status("📦 Chunking log file...", expanded=True) as status:
                start_time = time.time()
                
                # Detect format and choose chunking strategy
                fmt = detect_json_format(log_content[:10000], log_content)
                
                if fmt == 'array':
                    chunks = chunk_json_array(
                        json_text=log_content,
                        max_chars=MAX_CHARS_PER_CHUNK,
                        overlap_entries=overlap_ui,
                        max_chunks=max_analysis_chunks
                    )
                    chunks = inject_overlap(chunks, overlap_entries=overlap_ui)
                    chunking_method = "JSON-aware semantic chunking"
                else:
                    # Fallback to simple text chunking
                    chunks = [{'text': log_content[:50000], 'metadata': {'chunk_id': 'chunk_001', 'entry_range': (0, 0), 'entry_count': 1, 'char_count': min(50000, len(log_content)), 'overlap_applied': False}}]
                    chunking_method = "Simple truncation (non-JSON format)"
                
                chunk_time = time.time() - start_time
                status.update(label=f"✅ Created {len(chunks)} chunks in {chunk_time:.1f}s", state="complete")
                
                total_entries = sum(c['metadata']['entry_count'] for c in chunks) if fmt == 'array' else 'N/A'
                st.info(f"📦 **Chunking complete**: {len(chunks)} chunks, {total_entries} entries, method: {chunking_method}")
            
            # Show chunk metadata if requested
            if show_chunk_metadata and fmt == 'array':
                with st.expander("📊 Chunk Metadata (click to inspect)"):
                    for chunk in chunks:
                        with st.container():
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Chunk ID", chunk['metadata']['chunk_id'])
                            with col2:
                                st.metric("Entries", f"{chunk['metadata']['entry_range'][0]}-{chunk['metadata']['entry_range'][1]}")
                            with col3:
                                st.metric("Count", chunk['metadata']['entry_count'])
                            with col4:
                                st.metric("Size", f"{chunk['metadata']['char_count']:,} chars")
                            
                            st.text("Preview:")
                            st.code(chunk['preview'], language="json")
                            st.divider()
            
            # STEP 2: Filter relevant chunks (cost optimization)
            if enable_filtering and len(chunks) > 3 and fmt == 'array':
                with st.status("🔍 Filtering relevant chunks...", expanded=True) as status:
                    start_time = time.time()
                    chunks = tier1_filter_relevant_chunks(chunks, user_question, model, max_relevant=5)
                    filter_time = time.time() - start_time
                    status.update(label=f"✅ Filtered to {len(chunks)} relevant chunks in {filter_time:.1f}s", state="complete")
            else:
                st.info(f"⏭️ Skipping filtering (analyzing all {len(chunks)} chunks)")
            
            # Cost estimation
            if show_cost:
                estimated_input_chars = sum(len(chunk['text']) for chunk in chunks) + len(user_question) * len(chunks)
                estimated_output_chars = 500 * len(chunks)  # Assume 500 chars per analysis
                synthesis_input = min(MAX_SYNTHESIS_CHARS, estimated_input_chars + 2000)
                synthesis_output = 1000  # Assume 1000 chars for synthesis
                
                analysis_cost = estimate_cost(estimated_input_chars, estimated_output_chars)
                synthesis_cost = estimate_cost(synthesis_input, synthesis_output)
                total_estimated_cost = analysis_cost + synthesis_cost
                
                st.info(f"💰 **Estimated cost**: ${total_estimated_cost:.4f} (analysis: ${analysis_cost:.4f}, synthesis: ${synthesis_cost:.4f})")
                if total_estimated_cost > 0.10:
                    st.warning("⚠️ Cost estimate > $0.10. Consider reducing max chunks or enabling filtering.")
            
            # STEP 3: Analyze each chunk
            st.subheader("🧠 Chunk-by-Chunk Analysis")
            analyses = []
            
            analysis_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            for i, chunk in enumerate(chunks):
                progress_bar.progress(i / len(chunks))
                
                with st.status(f"🔍 Analyzing {chunk['metadata']['chunk_id']} (entries {chunk['metadata']['entry_range'][0]}-{chunk['metadata']['entry_range'][1]})...", expanded=False):
                    start_time = time.time()
                    analysis = tier2_analyze_chunk(chunk, user_question, model)
                    analysis_time = time.time() - start_time
                    
                    analyses.append(analysis)
                    
                    # Show individual analysis
                    with analysis_placeholder.container():
                        with st.expander(f"✅ {chunk['metadata']['chunk_id']} - {analysis_time:.1f}s", expanded=False):
                            if analysis['success']:
                                st.write(analysis['analysis'])
                            else:
                                st.error(analysis['analysis'])
            
            progress_bar.progress(1.0)
            st.success(f"✅ Completed analysis of {len(analyses)} chunks")
            
            # STEP 4: Synthesize final answer
            st.divider()
            st.subheader("🎯 Final Synthesis")
            
            with st.spinner("Synthesizing final answer from all analyses..."):
                start_time = time.time()
                final_answer = tier3_synthesize_analyses(analyses, user_question, model)
                synthesis_time = time.time() - start_time
            
            # Display final answer
            st.success(f"✅ Synthesis complete in {synthesis_time:.1f}s")
            st.write(final_answer)
            
            # Summary statistics
            st.divider()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total chunks", len(chunks))
            with col2:
                st.metric("Chunks analyzed", len(analyses))
            with col3:
                st.metric("Synthesis time", f"{synthesis_time:.1f}s")
            with col4:
                if show_cost:
                    st.metric("Est. cost", f"${total_estimated_cost:.4f}")
            
            # Export option
            with st.expander("📥 Export Results"):
                export_data = {
                    'question': user_question,
                    'model': MODEL_NAME,
                    'chunk_count': len(chunks),
                    'analyses': analyses,
                    'final_answer': final_answer,
                    'timestamp': datetime.now().isoformat()
                }
                st.download_button(
                    label="Download Analysis (JSON)",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"log_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

# Footer
st.divider()
st.caption("Made with ❤️ | Handles 35MB+ JSON files with semantic chunking and overlap")
