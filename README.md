# AI Text Summarization Agent

**Deep Learning-powered Text Summarization Application** using state-of-the-art BART Transformer model.

## Features

- 🤖 **Real AI Summarization**: Uses BART (Bidirectional and Auto-Regressive Transformers) model
- 🚀 **Deep Learning Powered**: Pre-trained on CNN/DailyMail dataset
- 📝 **Abstractive Summarization**: Generates new sentences, not just extracts
- ⚡ **Fast Processing**: Optimized for quick results
- 🎯 **Customizable**: Adjust summary length (min/max)
- 📊 **Metrics Display**: Shows compression ratio and word counts
- 💻 **Easy to Use**: Simple web interface - just paste and click!

## Setup

### Prerequisites
- Python 3.8 or higher (tested with Python 3.12)
- pip package manager

### Installation Steps

1. **Create a virtual environment** (recommended):
```bash
python -m venv venv
```

2. **Activate the virtual environment**:
   - **Windows (PowerShell)**:
     ```bash
     .\venv\Scripts\Activate.ps1
     ```
   - **Windows (Command Prompt)**:
     ```bash
     venv\Scripts\activate.bat
     ```
   - **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Run the app**:
```bash
streamlit run streamlit_app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

### Troubleshooting

**If pip is not found:**
```bash
python -m ensurepip --upgrade
```

**If you encounter permission errors:**
- Make sure you're using the virtual environment
- On Windows, you may need to run PowerShell as Administrator for the first activation

## How It Works

1. **Load the AI Model**: Click "Load AI Model" button (first time will download the model - takes ~1-2 minutes)
2. **Enter Your Text**: Paste any article, document, or text you want to summarize
3. **Generate Summary**: Click "Generate Summary" and the AI will create a summary
4. **View Results**: See your summarized text with metrics

## Model Information

- **Model**: facebook/bart-large-cnn
- **Type**: Transformer-based Encoder-Decoder
- **Training**: Pre-trained on CNN/DailyMail dataset
- **Download**: Automatically downloads on first use (~1.6GB)

## Requirements

- Python 3.8+ (tested with Python 3.12)
- Internet connection (for first-time model download)
- ~2GB free disk space (for model storage)
- See `requirements.txt` for package versions

