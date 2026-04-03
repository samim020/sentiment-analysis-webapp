📊 YouTube Comment Sentiment Analyzer

A Streamlit-based web application that extracts YouTube comments, performs Natural Language Processing (NLP) for sentiment classification, and generates AI-driven summaries of audience feedback. 

This tool automates the process of gauging viewer reception by categorizing comments into Positive, Neutral, and Negative sentiments, and leveraging Generative AI to provide concise overviews of what audiences loved or disliked.

✨ Features

Automated Data Extraction: Scrapes up to 500 top-level comments from any public YouTube video using the YouTube Data API v3. Handles pagination and includes built-in spam filtering (removes links, promotional codes, etc.).
Video Metadata Display: Fetches and displays real-time video statistics, including thumbnails, view counts, likes, and total comment counts.
Sentiment Classification: Utilizes a pre-trained Hugging Face transformer model (`cardiffnlp/twitter-roberta-base-sentiment-latest`) to classify the emotional tone of each comment.
Generative AI Summarization: Integrates Google's Gemini LLM (`gemini-3-flash-preview`) to read through the categorized comments and generate natural, human-readable summaries of the overall audience vibe.
Interactive UI: Built with Streamlit for a clean, responsive, and user-friendly web interface featuring metric dashboards and progress indicators.

 🛠️ Tech Stack

Language: Python
Frontend: Streamlit
Machine Learning / NLP: Hugging Face `transformers` (PyTorch)
Generative AI: Google GenAI SDK (`gemini-3-flash-preview`)
APIs: Google API Client (YouTube Data API v3)
Environment Management: `python-dotenv`

 🚀 Installation & Setup

Follow these steps to run the application locally.

 1. Clone the repository

git clone [https://github.com/YourUsername/youtube-sentiment-analyzer.git](https://github.com/YourUsername/youtube-sentiment-analyzer.git)
cd youtube-sentiment-analyzer

2. Create a Virtual Environment (Recommended)

python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

3. Install Dependencies
Create a requirements.txt file in your repository containing the following, then install them:

Plaintext
streamlit
transformers
google-api-python-client
python-dotenv
google-genai
torch

pip install -r requirements.txt

4. Configure Environment Variables
This app requires API keys for both YouTube and Google Gemini.
Create a .env file in the root directory of the project and add your keys:


YOUTUBE_API_KEY="your_youtube_api_key_here"
GEMINI_API_KEY="your_gemini_api_key_here"

5. Run the Application

streamlit run app.py



🧠 How It Works under the Hood
Input Validation: The app extracts the specific 11-character video_id from standard or shortened YouTube URLs.

Resource Caching: The Hugging Face sentiment pipeline is decorated with @st.cache_resource to ensure the model is loaded into memory only once, preventing long load times on app reruns.

Data Pipeline: - Comments are fetched in batches of 100 using page tokens.

Text is normalized to lowercase and filtered against a list of common spam flags.

Kept comments are truncated to 500 characters to respect transformer token limits.

Analysis & Aggregation: Comments are processed by the RoBERTa model, bucketed by sentiment, and then the raw text is concatenated (up to 3,500 characters) and sent to the Gemini API for final summarization.