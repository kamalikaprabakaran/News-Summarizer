Project Overview

This project is a web-based News Summarizer developed using Flask and Natural Language Processing (NLP) techniques.
The application accepts a news article URL, extracts the content, and generates a concise summary using AI models.
It also supports Tamil and English news articles and can read the summarized content aloud using text-to-speech.

The main goal of this project is to demonstrate the practical use of AI models, multilingual NLP, and web development in solving a real-world problem.

---

Features
1.Fetch and parse news articles from URLs  
2.Abstractive summarization (BART / mT5 multilingual models)  
3.Extractive summarization (important sentences)  
4.Key topic extraction (RAKE or fallback frequency analysis)  
5.Text-to-speech support (English + Tamil)  
6.Stores recent URL history in session  
7.Dark/Light theme toggle with history dropdown  

---
Clone the repository
```bash
git clone https://github.com/kamalikaprabakaran/News-Summarizer.git
cd news-summarizer
