# Sentiment Analysis Dashboard (Python) - Week 3 AI Bootcamp Project

This project is a fully functional Sentiment Analysis Tool and Dashboard, built to fulfill the Week 3 deliverables for the Clickatell 1-Month AI Bootcamp.

## Features & Deliverables Met

1. **Sentiment Analysis Tool**: An interactive UI allowing users to input text and immediately see the sentiment classification (Positive, Negative, Neutral) and score. It also supports batch processing via CSV upload.
2. **Dashboard Interpreting Data**: A real-time updating dashboard featuring a Sentiment Distribution pie chart and a Compound Score histogram, powered by Plotly.
3. **Insights Report**: Automatically generated, dynamic insights based on the aggregate sentiment data analyzed from the CSV.

## Technical Explanation

This application is built using a modern Python data stack:

- **Streamlit**: Used for building the interactive web dashboard and handling the frontend user interface entirely in Python.
- **NLTK VADER (Valence Aware Dictionary and sEntiment Reasoner)**: The core sentiment analysis engine. VADER is highly optimized for social media text and works excellently without requiring intensive GPU resources or massive pre-trained language models. It uses a lexicon and rule-based sentiment analysis approach to calculate positive, negative, neutral, and compound scores.
- **Pandas**: Used for managing the data structures and processing the uploaded CSV files.
- **Plotly**: Used for rendering the interactive charts on the dashboard.

## Installation and Setup

Since you will be installing Python later, follow these exact steps once Python is installed on your machine:

1. **Install Python**: Ensure Python (3.8 or higher) is installed. Check the box to "Add Python to PATH" during installation.
2. **Open Terminal/Command Prompt**: Navigate to this project folder.
3. **Install Dependencies**: Run the following command to install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Dashboard**: Start the Streamlit application by running:
   ```bash
   streamlit run app.py
   ```

A browser window will automatically open with your Sentiment Analysis Dashboard ready to use!
