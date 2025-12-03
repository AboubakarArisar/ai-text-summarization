"""
AI Agent for Text Summarization
Deep Learning-based Text Summarization using BART Transformer Model
"""

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
import torch

# Page configuration
st.set_page_config(
    page_title="AI Text Summarization Agent",
    layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .model-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .summary-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #ffffff;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'summarizer_loaded' not in st.session_state:
    st.session_state.summarizer_loaded = False
if 'summarizer_model' not in st.session_state:
    st.session_state.summarizer_model = None

@st.cache_resource
def load_summarizer():
    """Load pre-trained BART model for text summarization"""
    try:
        with st.spinner("Loading AI Summarization Model (this may take a minute on first run)..."):
            # Use BART model specifically trained for summarization
            model_name = "facebook/bart-large-cnn"
            
            # Load the summarization pipeline
            summarizer = pipeline(
                "summarization",
                model=model_name,
                tokenizer=model_name,
                device=0 if torch.cuda.is_available() else -1  # Use GPU if available
            )
            
            st.success("AI Model loaded successfully!")
            return summarizer, None
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def generate_summary_bart(text, summarizer, max_length=130, min_length=30):
    """Generate summary using BART model"""
    try:
        # BART has a token limit, so we need to chunk long texts
        max_input_length = 1024
        
        if len(text.split()) > max_input_length:
            # Split into chunks and summarize each, then combine
            words = text.split()
            chunks = []
            current_chunk = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 1 <= max_input_length:
                    current_chunk.append(word)
                    current_length += len(word) + 1
                else:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [word]
                    current_length = len(word)
            
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            
            # Summarize each chunk
            summaries = []
            for chunk in chunks:
                if len(chunk.strip()) > 50:  # Only summarize meaningful chunks
                    summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
                    summaries.append(summary[0]['summary_text'])
            
            # Combine summaries
            combined_text = " ".join(summaries)
            
            # If combined is still too long, summarize again
            if len(combined_text.split()) > max_input_length:
                final_summary = summarizer(combined_text, max_length=max_length, min_length=min_length, do_sample=False)
                return final_summary[0]['summary_text']
            else:
                return combined_text
        else:
            # Direct summarization for shorter texts
            summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
            return summary[0]['summary_text']
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def calculate_metrics(original_text, summary_text):
    """Calculate basic metrics for the summary"""
    original_words = len(original_text.split())
    summary_words = len(summary_text.split())
    compression_ratio = summary_words / original_words if original_words > 0 else 0
    
    return {
        'original_length': original_words,
        'summary_length': summary_words,
        'compression_ratio': f"{compression_ratio:.2%}"
    }

def main():
    # Header
    st.markdown('<h1 class="main-header">AI Text Summarization Agent</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("Agent Configuration")
        st.markdown("""
        **Model Information:**
        - **Model**: BART (Bidirectional and Auto-Regressive Transformers)
        - **Architecture**: Transformer-based Encoder-Decoder
        - **Training**: Pre-trained on CNN/DailyMail dataset
        - **Type**: Abstractive Summarization (generates new sentences)
        
        **Features:**
        - Deep Learning powered
        - State-of-the-art transformer model
        - Handles long documents automatically
        - Fast and accurate summarization
        """)
        
        st.markdown("---")
        st.markdown("**Instructions:**")
        st.markdown("1. Enter or paste your article/text")
        st.markdown("2. Adjust summary length (optional)")
        st.markdown("3. Click 'Generate Summary'")
        st.markdown("4. View your summarized text")
        
        st.markdown("---")
        st.markdown("**Summary Length Settings:**")
        max_len = st.slider("Max Length", 50, 200, 130, help="Maximum length of summary")
        min_len = st.slider("Min Length", 10, 100, 30, help="Minimum length of summary")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Text")
        article_input = st.text_area(
            "Enter or paste your article/text here:",
            height=400,
            placeholder="Paste your text here... The AI will summarize it for you.",
            help="Enter any text you want to summarize. Can be articles, documents, or any long text."
        )
        
        # Example article button
        if st.button("Load Example Article", use_container_width=True):
            example_article = """
            Most of us will have felt the pain of a bee sting. Luckily most of us will have avoided the dreaded pain of a tarantula hawk or a fire ant. 
            Justin Schmidt felt all three of these - and 147 other horrible, burning sensations - after a dedicated life-long career devoted to insects. 
            On numerous fieldwork trips, The University of Arizona entomologist would find himself digging up living colonies of creatures, who in turn 
            were not happy with this destructive human scooping them into bags - and promptly sank their fangs, stingers or pincers into him. 
            Still, no pain, no gain, and Schmidt turned his experiences into the Schmidt Sting Pain Index, ranking 78 species in a list which, 
            while subjective, was put together by the man who must surely know best, ranking their pain on a scale of 1 to 4.
            The index has become a valuable resource for entomologists and researchers studying insect behavior and venom. 
            Schmidt's work has helped scientists understand the evolutionary purposes of different types of stings and bites.
            His research has also contributed to medical science, helping doctors understand pain mechanisms and develop better treatments.
            """
            st.session_state.example_article = example_article
            st.rerun()
        
        if 'example_article' in st.session_state:
            article_input = st.text_area(
                "Enter or paste your article/text here:",
                value=st.session_state.example_article,
                height=400
            )
    
    with col2:
        st.subheader("AI Agent Processing")
        
        # Model status
        if st.session_state.summarizer_model is None:
            st.info("**Status:** Model not loaded. Click 'Load AI Model' first.")
        else:
            st.success("**Status:** AI Model Ready!")
        
        # Load model button
        if st.button("Load AI Model", use_container_width=True, type="secondary"):
            with st.spinner("Loading deep learning model..."):
                summarizer, error = load_summarizer()
                if error:
                    st.error(f"{error}")
                else:
                    st.session_state.summarizer_model = summarizer
                    st.session_state.summarizer_loaded = True
                    st.success("Model loaded! Ready to summarize.")
                    st.rerun()
        
        # Generate summary button
        if st.button("Generate Summary", type="primary", use_container_width=True, disabled=st.session_state.summarizer_model is None):
            if not article_input or len(article_input.strip()) < 50:
                st.error("Please enter text with at least 50 characters.")
            else:
                with st.spinner("AI Agent is processing your text..."):
                    # Step 1: Loading model (if not loaded)
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    if st.session_state.summarizer_model is None:
                        status_text.text("Step 1/4: Loading AI model...")
                        progress_bar.progress(10)
                        summarizer, error = load_summarizer()
                        if error:
                            st.error(f"Error: {error}")
                            st.stop()
                        st.session_state.summarizer_model = summarizer
                    else:
                        summarizer = st.session_state.summarizer_model
                    
                    progress_bar.progress(25)
                    
                    # Step 2: Preprocessing
                    status_text.text("Step 2/4: Preprocessing text...")
                    progress_bar.progress(50)
                    
                    # Step 3: Generating summary
                    status_text.text("Step 3/4: Generating summary using deep learning...")
                    progress_bar.progress(75)
                    
                    try:
                        summary_text = generate_summary_bart(article_input, summarizer, max_length=max_len, min_length=min_len)
                    except Exception as e:
                        st.error(f"Error generating summary: {str(e)}")
                        st.stop()
                    
                    # Step 4: Post-processing
                    status_text.text("Step 4/4: Finalizing results...")
                    progress_bar.progress(100)
                    
                    # Calculate metrics
                    metrics = calculate_metrics(article_input, summary_text)
                    
                    # Store results
                    st.session_state.article = article_input
                    st.session_state.summary = summary_text
                    st.session_state.metrics = metrics
                    st.session_state.processed = True
                    
                    status_text.text("Processing complete!")
                    progress_bar.progress(100)
    
    # Results section
    if st.session_state.get('processed', False):
        st.markdown("---")
        st.subheader("AI Agent Output - Summary Results")
        
        # Display summary
        st.markdown('<div class="model-card">', unsafe_allow_html=True)
        st.markdown("### Generated Summary")
        
        st.markdown('<div class="summary-box">', unsafe_allow_html=True)
        st.write(st.session_state.summary)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display metrics
        st.markdown("### Summary Metrics")
        col_met1, col_met2, col_met3 = st.columns(3)
        
        with col_met1:
            st.metric("Original Length", f"{st.session_state.metrics['original_length']} words")
        with col_met2:
            st.metric("Summary Length", f"{st.session_state.metrics['summary_length']} words")
        with col_met3:
            st.metric("Compression Ratio", st.session_state.metrics['compression_ratio'])
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Copy button
        st.markdown("---")
        if st.button("Copy Summary to Clipboard", use_container_width=True):
            st.code(st.session_state.summary, language=None)
            st.success("Summary displayed above. You can copy it manually.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p><strong>AI Text Summarization Agent</strong> | Powered by BART Transformer Model</p>
        <p>Built with Streamlit & Hugging Face Transformers | Deep Learning Technology</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
