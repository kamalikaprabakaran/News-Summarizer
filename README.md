<div align="center">

# 📰 Smart News Summarizer

### A Flask web app that extracts news articles from a URL and summarizes them using AI — with support for English and Tamil, and text-to-speech playback.

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-FFD21E?style=for-the-badge)](https://huggingface.co/docs/transformers)

</div>

---

## 📌 Overview

**Smart News Summarizer** is a web app that takes a news article URL, extracts its content, and generates a summary using pretrained NLP models. It detects whether the article is in **English or Tamil**, produces both **abstractive** (AI-generated) and **extractive** (key-sentence) summaries, pulls out key topics, and can read the summary aloud via text-to-speech. Recent URLs are kept in a session-based history for quick access.

---

## ✨ Features

- 🔗 **Article extraction from a URL**, using `newspaper3k` with a BeautifulSoup-based fallback if extraction fails
- 🌐 **Automatic language detection** (English / Tamil) via `langdetect`
- 🤖 **Abstractive summarization** — `facebook/bart-large-cnn` for English, `csebuetnlp/mT5_multilingual_XLSum` for Tamil/multilingual content
- 📑 **Extractive summarization** — ranks and selects the most important sentences using word-frequency scoring
- 🔑 **Key topic extraction** — via RAKE (`rake-nltk`) for English, with a frequency-based fallback for other languages
- 🔊 **Text-to-speech** — converts the summary to an audio file using `gTTS`, in either English or Tamil
- 🕘 **Session-based history** — keeps track of the last 20 summarized URLs
- 🎛️ **Adjustable summary length** (short / default / long) and mode (abstractive / extractive / both)
- 🌗 **Dark / light theme toggle** in the UI

---

## 🧠 How It Works

1. **Article extraction** — `get_article_text()` fetches and parses the article using `newspaper3k`; if that fails or returns empty text, it falls back to a raw `requests` + `BeautifulSoup` scrape of all `<p>` tags.
2. **Language detection** — `detect_language()` uses `langdetect` to classify the text as Tamil (`ta`) or English (`en`), defaulting to English on failure.
3. **Summarization** — `make_summary_package()` builds the response:
   - **Abstractive**: the appropriate Hugging Face summarization pipeline (BART for English, mT5 for Tamil) is run on the article text, chunked to the first 2000 characters, with length bounds set by the selected summary length (short/default/long).
   - **Extractive**: `extract_key_sentences()` scores each sentence by the combined frequency of its words and returns the top-ranked sentences as bullet points.
4. **Key topics** — `extract_key_topics()` uses RAKE for English text, or a stopword-filtered word-frequency count as a fallback for other languages / when RAKE is unavailable.
5. **Text-to-speech** — `/speak` (and the `text_to_speech()` helper) generates an `.mp3` file with `gTTS` and saves it to the `static/` folder for playback.
6. **History** — `push_history()` stores each summarized URL and its result in the Flask session, capping history at the 20 most recent entries.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|--------------|
| `GET`  | `/` | Renders the main UI, along with any existing session history |
| `POST` | `/summary` | Accepts a JSON body (`url`, `language`, `length`, `mode`) and returns the summary package (title, abstractive summary, extractive summary, key topics, detected language) |
| `POST` | `/speak` | Accepts form data (`text`, `lang`) and returns a path to a generated `.mp3` audio file |

**Example `/summary` request:**
```json
{
  "url": "https://example.com/news-article",
  "language": "en",
  "length": "default",
  "mode": "both"
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Python, Flask |
| **Article Extraction** | newspaper3k, BeautifulSoup4, Requests |
| **NLP / Summarization** | Hugging Face Transformers (BART, mT5), NLTK, RAKE |
| **Language Detection** | langdetect |
| **Text-to-Speech** | gTTS |
| **Frontend** | HTML, CSS, JavaScript (sidebar UI with theme toggle) |

---

## 📁 Project Structure

```
News-Summarizer/
├── app.py                 # Flask app: article extraction, summarization, TTS, API routes
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html         # Main UI
├── static/                # Generated audio files (output.mp3, etc.)
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/kamalikaprabakaran/News-Summarizer.git
   cd News-Summarizer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   python app.py
   ```

4. **Open in browser**
   Visit `http://127.0.0.1:5000/`

---

## 🎯 Example Usage

1. Paste a news article URL into the input field.
2. Choose the summary **length** (short / default / long) and **mode** (abstractive / extractive / both).
3. Click summarize — view the AI-generated summary, key sentences, and extracted key topics.
4. Optionally, click the text-to-speech option to have the summary read aloud.
5. Revisit any previously summarized article from the session history sidebar.

---

## 🙋‍♀️ Author

**Kamalika Prabakaran**
📎 [GitHub Profile](https://github.com/kamalikaprabakaran)

If you found this project interesting, consider giving it a ⭐ — it helps a lot!

</div>
