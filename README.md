# 📰 News Summarizer

A Flask-based web app that fetches news articles from a given URL, generates **AI-powered summaries** (abstractive + extractive), extracts key topics, and even reads the summary aloud using **gTTS (Google Text-to-Speech)**.

---

## ✨ Features
- 🌐 Fetch and parse news articles from URLs  
- 🤖 Abstractive summarization (BART / mT5 multilingual models)  
- 📌 Extractive summarization (important sentences)  
- 🔑 Key topic extraction (RAKE or fallback frequency analysis)  
- 🗣️ Text-to-speech support (English + Tamil)  
- 🕒 Stores recent URL history in session  
- 🎨 Dark/Light theme toggle with history dropdown  

---

## 🚀 Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/kamalikaprabakaran/News-Summarizer.git
cd news-summarizer
