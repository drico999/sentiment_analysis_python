# SentimentAI — Sentiment Analysis Dashboard

A fully functional sentiment analysis tool and interactive dashboard built with Python. Analyse individual pieces of text or upload a CSV dataset to get a full sentiment breakdown with charts and insights.

## Features

1. **Single Text Analysis** — Paste any text and instantly see its sentiment classification (Positive, Negative, or Neutral), compound score, and a visual gauge.
2. **Batch Dashboard** — Upload a CSV file to analyse hundreds or thousands of entries at once, with a sentiment distribution pie chart, compound score histogram, and entry count bar chart.
3. **Insights Report** — Automatically generated summary based on the aggregate sentiment of your dataset.
4. **Export Results** — Download the full analysed dataset as a CSV after batch processing.

## Tech Stack

- **Streamlit** — Interactive web dashboard, built entirely in Python.
- **NLTK VADER** (Valence Aware Dictionary and sEntiment Reasoner) — The core sentiment engine. Optimised for short-form and social media text, using a lexicon and rule-based approach to calculate positive, negative, neutral, and compound scores without requiring a GPU or large language model.
- **Pandas** — Data structure management and CSV processing.
- **Plotly** — Interactive charts and visualisations.

## Installation & Setup

### Prerequisites
- Python 3.8 or higher — [Download here](https://www.python.org/downloads/). During installation, check **"Add Python to PATH"**.

### Steps

1. **Clone or download** this project and place all files in the same folder:
   ```
   app.py
   sentiment_utils.py
   requirements.txt
   ```

2. **Open a terminal** (Command Prompt on Windows, Terminal on Mac/Linux) and navigate to the project folder:
   ```bash
   cd path/to/your/project-folder
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

A browser tab will open automatically at `http://localhost:8501`.

## Using the App

### Single Text Analysis
1. Select **"Analyze Single Text"** in the sidebar.
2. Type or paste any text into the input box.
3. Click **Analyze →** to see the score, classification, and gauge chart.

### Batch / Dashboard Mode
1. Select **"Dashboard & Batch"** in the sidebar.
2. Upload a CSV file that contains a column named `text`, `review`, `tweet`, `comment`, `message`, or `content` — the app detects it automatically.
3. Charts and insights will generate instantly after upload.
4. Click **Download full results CSV** to save the scored dataset.

> **No CSV?** Expand the "Need a sample file?" section inside the app to generate and download a demo dataset.

## How Sentiment Scoring Works

| Score Range | Classification |
|---|---|
| ≥ 0.05 | Positive |
| −0.05 to 0.05 | Neutral |
| ≤ −0.05 | Negative |

The **compound score** is a single normalised value between −1.0 (most negative) and +1.0 (most positive) that summarises the overall sentiment of the text.
