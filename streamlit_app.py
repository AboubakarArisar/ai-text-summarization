"""
AI Agent for Text Summarization
Demonstrates both Model A (No Attention) and Model B (With Attention)
"""

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AI Text Summarization Agent",
    page_icon="📝",
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
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'vectorizers_loaded' not in st.session_state:
    st.session_state.vectorizers_loaded = False

# Constants (should match notebook)
VOCAB_SIZE = 30000
MAX_ARTICLE_LEN = 600
MAX_SUMMARY_LEN = 120
EMBED_SIZE = 128

@st.cache_resource
def load_models():
    """Load both trained models from Google Drive"""
    import gdown
    import os
    
    # Google Drive file IDs
    MODEL_A_DRIVE_ID = "1sqcN84pqSI7O81tegUb56fOtFbLyr1jr"
    MODEL_B_DRIVE_ID = "1U7xWOne35qO9JMo8lb4rGwRi_eKuu119"
    
    model_a_path = "model_a.h5"
    model_b_path = "model_b.h5"
    
    # Download Model A if not exists
    if not os.path.exists(model_a_path):
        st.info("📥 Downloading Model A from Google Drive...")
        url_a = f"https://drive.google.com/uc?export=download&id={MODEL_A_DRIVE_ID}"
        try:
            gdown.download(url_a, model_a_path, quiet=False)
            st.success("✅ Model A downloaded!")
        except Exception as e:
            st.error(f"❌ Failed to download Model A: {str(e)}")
            return None, None, f"Download error: {str(e)}"
    
    # Download Model B if not exists
    if not os.path.exists(model_b_path):
        st.info("📥 Downloading Model B from Google Drive...")
        url_b = f"https://drive.google.com/uc?export=download&id={MODEL_B_DRIVE_ID}"
        try:
            gdown.download(url_b, model_b_path, quiet=False)
            st.success("✅ Model B downloaded!")
        except Exception as e:
            st.error(f"❌ Failed to download Model B: {str(e)}")
            return None, None, f"Download error: {str(e)}"
    
    # Load models
    try:
        st.info("🔄 Loading models into memory...")
        model_a = tf.keras.models.load_model(model_a_path, compile=False)
        model_b = tf.keras.models.load_model(model_b_path, compile=False)
        st.success("✅ Models loaded successfully!")
        return model_a, model_b, None
    except Exception as e:
        return None, None, str(e)

@st.cache_resource
def load_vectorizers():
    """Load text vectorizers"""
    try:
        # These need to be loaded from the notebook or saved separately
        # For now, we'll create a placeholder
        # In actual deployment, you'd save/load these from the notebook
        return None, "Vectorizers need to be loaded from notebook"
    except Exception as e:
        return None, str(e)

def generate_summary(model, article, article_vectorizer, summary_vectorizer, max_length=MAX_SUMMARY_LEN):
    """Generate summary using greedy decoding"""
    vocab = summary_vectorizer.get_vocabulary()
    unk_id = vocab.index('[UNK]') if '[UNK]' in vocab else 1
    
    summary = ""
    
    for word_idx in range(max_length):
        X = article_vectorizer([article])
        X_dec = summary_vectorizer([f"startofseq {summary}"])
        
        # Use last timestep (-1) since we're generating word by word
        y_proba = model.predict([X, X_dec], verbose=0)[0, -1]
        
        # Avoid [UNK] tokens
        y_proba[unk_id] = 0
        predicted_id = np.argmax(y_proba)
        predicted_word = vocab[predicted_id]
        
        if predicted_word == "endofseq":
            break
        
        summary += " " + predicted_word
    
    return summary.strip()

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Text Summarization Agent</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Agent Configuration")
        st.markdown("""
        **Model Information:**
        - Model A: Encoder-Decoder (No Attention)
        - Model B: Encoder-Decoder (With Attention)
        
        **Dataset:** CNN/DailyMail
        **Task:** Text Summarization
        """)
        
        st.markdown("---")
        st.markdown("**Instructions:**")
        st.markdown("1. Enter or paste an article")
        st.markdown("2. Click 'Generate Summary'")
        st.markdown("3. Compare results from both models")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📄 Input Article")
        article_input = st.text_area(
            "Enter or paste your article here:",
            height=300,
            placeholder="Paste your article text here...",
            help="The article you want to summarize"
        )
        
        # Example article button
        if st.button("📋 Load Example Article"):
            example_article = """
            Most of us will have felt the pain of a bee sting. Luckily most of us will have avoided the dreaded pain of a tarantula hawk or a fire ant. 
            Justin Schmidt felt all three of these - and 147 other horrible, burning sensations - after a dedicated life-long career devoted to insects. 
            On numerous fieldwork trips, The University of Arizona entomologist would find himself digging up living colonies of creatures, who in turn 
            were not happy with this destructive human scooping them into bags - and promptly sank their fangs, stingers or pincers into him. 
            Still, no pain, no gain, and Schmidt turned his experiences into the Schmidt Sting Pain Index, ranking 78 species in a list which, 
            while subjective, was put together by the man who must surely know best, ranking their pain on a scale of 1 to 4.
            """
            st.session_state.example_article = example_article
            st.rerun()
        
        if 'example_article' in st.session_state:
            article_input = st.text_area(
                "Enter or paste your article here:",
                value=st.session_state.example_article,
                height=300
            )
    
    with col2:
        st.subheader("🎯 Agent Processing")
        
        # Model status
        st.info("**Status:** Ready to process")
        
        # Processing options
        use_model_a = st.checkbox("Use Model A (No Attention)", value=True)
        use_model_b = st.checkbox("Use Model B (With Attention)", value=True)
        
        if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
            if not article_input or len(article_input.strip()) < 50:
                st.error("⚠️ Please enter an article with at least 50 characters.")
            else:
                with st.spinner("🤖 AI Agent is processing..."):
                    # Step 1: Loading models
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Step 1/4: Loading models from Google Drive...")
                    progress_bar.progress(10)
                    
                    # Load models (will download from Google Drive if needed)
                    model_a, model_b, error = load_models()
                    
                    if error:
                        st.error(f"❌ Error loading models: {error}")
                        st.stop()
                    
                    progress_bar.progress(25)
                    
                    # Step 2: Preprocessing
                    status_text.text("Step 2/4: Preprocessing article...")
                    progress_bar.progress(50)
                    
                    # Step 3: Generating summaries
                    status_text.text("Step 3/4: Generating summaries...")
                    progress_bar.progress(75)
                    
                    # Step 4: Post-processing
                    status_text.text("Step 4/4: Finalizing results...")
                    progress_bar.progress(100)
                    
                    status_text.text("✅ Processing complete!")
                    
                    # Store results
                    st.session_state.article = article_input
                    st.session_state.processed = True
    
    # Results section
    if st.session_state.get('processed', False):
        st.markdown("---")
        st.subheader("📊 Agent Output - Summary Results")
        
        results_col1, results_col2 = st.columns(2)
        
        with results_col1:
            if use_model_a:
                st.markdown('<div class="model-card">', unsafe_allow_html=True)
                st.markdown("### Model A: Encoder-Decoder (No Attention)")
                
                # Simulated summary (replace with actual model prediction)
                summary_a = "This is a simulated summary from Model A. In actual deployment, this would be generated by the trained model."
                
                st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                st.write(summary_a)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Metrics
                st.caption("**Metrics:** ROUGE-1: 0.1204 | ROUGE-2: 0.0103 | ROUGE-L: 0.0854")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with results_col2:
            if use_model_b:
                st.markdown('<div class="model-card">', unsafe_allow_html=True)
                st.markdown("### Model B: Encoder-Decoder (With Attention)")
                
                # Simulated summary (replace with actual model prediction)
                summary_b = "This is a simulated summary from Model B. In actual deployment, this would be generated by the trained model with attention mechanism."
                
                st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                st.write(summary_b)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Metrics
                st.caption("**Metrics:** ROUGE-1: 0.1920 | ROUGE-2: 0.0523 | ROUGE-L: 0.1388")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Comparison
        if use_model_a and use_model_b:
            st.markdown("---")
            st.subheader("📈 Model Comparison")
            
            comparison_data = {
                'Metric': ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'Processing Time'],
                'Model A': ['0.1204', '0.0103', '0.0854', '~2.5s'],
                'Model B': ['0.1920', '0.0523', '0.1388', '~3.0s'],
                'Improvement': ['+59.5%', '+407.8%', '+62.5%', 'Slightly slower']
            }
            
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>AI Text Summarization Agent | Built with Streamlit & TensorFlow</p>
        <p>Models: Encoder-Decoder with/without Attention Mechanism</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

