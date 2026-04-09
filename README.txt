# YouTube Comment Sentiment Analyzer 🎯

A Python web app that scrapes YouTube comments, classifies their sentiment using an LLM, and surfaces audience insights through summaries, charts, and topic clusters.

---

## What It Does

Paste any YouTube video URL and the app will:

- **Fetch video metadata** — thumbnail, title, channel, views, likes, comment count
- **Scrape up to 500 top comments** ordered by relevance (highest engagement first)
- **Classify each comment** as positive, negative, or neutral using Llama 3 via Groq API — in batches of 50 for speed
- **Show sentiment breakdown** — percentage split across all three categories
- **Plot sentiment over time** — line chart showing how audience mood shifted month by month since the video was published
- **Surface top liked comments** — top 3 most liked comments per sentiment category
- **Generate AI summaries** — one paragraph each for what people loved, disliked, and observed neutrally
- **Generate an overall verdict** — single punchy paragraph summarizing the full audience reception
- **Identify recurring topics** — 4-5 theme clusters pulled from the comment pool (e.g. "audio quality", "emotional reactions")

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Comment Scraping | YouTube Data API v3 |
| Sentiment Classification | Llama 3 8B via Groq API |
| Summarization & Topics | Llama 3 8B via Groq API |
| Charts | Plotly |
| Language | Python 3.9+ |

---

## Project Structure

```
├── app.py               # main application file
├── requirements.txt     # all dependencies
├── .env                 # secret api keys (never commit this)
└── README.md
```

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get your API keys

**YouTube Data API v3**
1. Go to [https://console.cloud.google.com](https://console.cloud.google.com)
2. Create a new project
3. Enable the **YouTube Data API v3** from the API library
4. Go to Credentials → Create Credentials → API Key
5. Copy the key

**Groq API**
1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up and go to API Keys
3. Click Create Key, give it a name, copy the key
4. Free tier gives 14,400 requests/day — more than enough for this app

### 4. Create your `.env` file
Create a file called `.env` in the root of the project and add:
```
YOUTUBE_API_KEY=your_youtube_key_here
GROQ_API_KEY=your_groq_key_here
```

### 5. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## How to Use

1. Paste any YouTube video URL into the input box — both formats work:
   - `https://www.youtube.com/watch?v=VIDEO_ID`
   - `https://youtu.be/VIDEO_ID`
2. Click **Analyze Video Comments**
3. Wait for comment scraping (~5 seconds) and batch classification (~30-60 seconds depending on comment count)
4. Scroll through the full analysis

---

## Key Design Decisions

**Why Groq instead of a local model?**
Small local sentiment models (like `cardiffnlp/twitter-roberta`) are trained on tweets and fail badly on YouTube comments — they flag emotionally complex comments as negative based on individual sad words without understanding context. Llama 3 via Groq understands context, sarcasm, and nuance. Groq's free tier is generous enough that API cost is not a concern for project use.

**Why batch classification?**
Sending 500 individual API calls one by one means 500 sequential network round trips. Batching 50 comments per call reduces that to 10 calls, cutting classification time from ~8 minutes to under a minute with no added complexity.

**Why `order=relevance` when fetching comments?**
YouTube's default order is newest first. With a 500 comment limit on a video with 10,000 comments, fetching newest-first means you miss the most engaged comments entirely. `order=relevance` sends YouTube's own engagement-ranked comments first, giving a far more representative sample.

---

## Limitations

- YouTube API quota is 10,000 units/day on the free tier. Each comment fetch uses ~3 units per 100 comments, so you can run roughly 30 full analyses per day before hitting limits.
- Comments are capped at 500 per analysis due to API and LLM cost considerations.
- The sentiment over time chart requires comments spanning at least 2 different months to render — very new videos may not show it.
- Spam filtering is keyword-based and may occasionally remove legitimate comments containing flagged words.

---

## Built For

Minor project submission — demonstrating practical integration of LLM APIs, third-party data APIs, and interactive data visualization in a real-world Python web application.
