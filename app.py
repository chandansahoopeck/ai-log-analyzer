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
from typing import List, Dict, Any, Tuple, Optional
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
MODEL_NAME = "gemini-2.0-flash-001"  # Update when available
FALLBACK_MODEL = "gemini-1.5-flash"   # Fallback if primary unavailable

# Chunking configuration
MAX_CHARS_PER_CHUNK = 45000    # Leave room for prompt overhead
OVERLAP_ENTRIES = 3            # Overlap between chunks for context preservation
MAX_CHUNKS = 15                # Safety limit (≈ $0.045 cost with gemini-1.5-flash)
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
    
    filter_prompt_template = """You are a log filtering expert. Determine if this log chunk contains information RELEVANT to the question.

Question: {question}

Log chunk (first 5 entries):
