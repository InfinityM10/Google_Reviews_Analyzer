# Google Reviews Analyzer

## Overview

Google Reviews Analyzer is a Streamlit application that helps businesses extract meaningful insights from their Google Reviews data. The tool uses Natural Language Processing (NLP) techniques and optional Large Language Model (LLM) enhancement to identify key themes, track sentiment trends, and provide actionable recommendations based on customer feedback.

## Features

- **Theme Extraction**: Automatically identifies major themes and topics from your review data
- **Sentiment Analysis**: Tracks sentiment trends over time to identify periods of improvement or decline
- **Actionable Recommendations**: Provides business-specific recommendations based on review themes
- **Interactive Visualizations**: Explore your data through rating distributions, sentiment trends, and term frequency charts
- **LLM Enhancement**: Optional integration with Llama 3 via Ollama for more nuanced and human-like analysis

## Requirements

- Python 3.8+
- Streamlit
- Pandas
- Plotly
- scikit-learn
- NumPy
- Requests (for Ollama integration)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/google-reviews-analyzer.git
   cd google-reviews-analyzer
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Launch the application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`

3. Upload your Google Reviews CSV file or use the sample data option

4. Click "Run Analysis" to process the reviews and generate insights

## CSV Format

Your Google Reviews CSV file should contain at least these columns:
- `rating`: Numerical rating (typically 1-5)
- `text`: Review content text
- `time`: Review date (YYYY-MM-DD format)

## LLM Enhancement (Optional)

The application can use Llama 3 via [Ollama](https://ollama.ai/) for enhanced analysis:

1. Install Ollama from [ollama.ai](https://ollama.ai/)
2. Pull the Llama 3 model:
   ```bash
   ollama pull llama3
   ```
3. Run Ollama in the background before starting the application
4. The application will automatically detect if Ollama is available

If Ollama is not available, the application will fall back to basic NLP techniques.

## How It Works

1. **Topic Modeling**: The application uses Non-negative Matrix Factorization (NMF) with TF-IDF vectorization to extract key topics from review text
2. **Theme Extraction**: Topics are converted to business-relevant themes with sample phrases
3. **Sentiment Analysis**: Review ratings and text are analyzed to track sentiment trends over time
4. **LLM Enhancement**: When available, Llama 3 is used to improve theme titles, provide better recommendations, and enhance sentiment analysis

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)

## Acknowledgements

- [Streamlit](https://streamlit.io/) for the web application framework
- [scikit-learn](https://scikit-learn.org/) for NLP capabilities
- [Plotly](https://plotly.com/) for interactive visualizations
- [Ollama](https://ollama.ai/) for local LLM inference
