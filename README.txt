YouTube Comment Sentiment Analyzer 📊

A full-stack Python web application built with Streamlit that extracts real user comments from any YouTube video, analyzes their emotional tone using the Llama-3 LLM (via Groq), and provides visual and textual insights into audience feedback.

🚀 Features

• Automated Data Extraction: Securely fetches up to 500 top-level comments and video metadata using the YouTube Data API v3. Includes a basic spam-filter to remove self-promotional comments.
• AI Sentiment Classification: Leverages llama-3.3-70b-versatile through the Groq API to categorize every comment as Positive, Negative, or Neutral.
• Sentiment Trend Visualization: Uses Plotly to render an interactive, time-series line chart showing how audience sentiment changes over time.
• Top Comment Highlighting: Automatically sorts and displays the most-liked positive and negative comments.
• Generative AI Summaries: Reads through the categorized comments and writes a natural, 2-3 sentence summary of what the audience specifically loved and disliked about the video.

🛠️ Prerequisites

Before you begin, ensure you have met the following requirements:
• Python 3.8 or higher installed.
• A YouTube Data API v3 Key (Get this from the Google Cloud Console).
• A Groq API Key (Get this from the Groq Developer Console).

⚙️ Installation

Clone the repository:
git clone https://github.com/yourusername/youtube-sentiment-analyzer.git
cd youtube-sentiment-analyzer

Create a virtual environment (Recommended):
python -m venv venv
source venv/bin/activate (On Windows use: venv\Scripts\activate)

Install the dependencies:
pip install -r requirements.txt

Set up Environment Variables:
Create a file named .env in the root directory of the project and add your API keys:
YOUTUBE_API_KEY=your_youtube_api_key_here
GROQ_API_KEY=your_groq_api_key_here

💻 Running the App

Execute the following command in your terminal to start the Streamlit server:
streamlit run app.py

The application will automatically open in your default web browser at http://localhost:8501.