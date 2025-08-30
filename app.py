from flask import Flask, render_template, request, session, jsonify
from newspaper import Article
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import requests
from bs4 import BeautifulSoup
import nltk
import re
from collections import Counter
from langdetect import detect, DetectorFactory
from gtts import gTTS
import os

DetectorFactory.seed = 0  # make detection deterministic

app = Flask(__name__)
app.secret_key = "supersecretkey"  # change for production

# Download NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# Lazy load summarization models
summarizer_en = None
summarizer_multi = None

def text_to_speech(text, lang="en"):
    # Make sure static folder exists (even if you already created it)
    os.makedirs("static", exist_ok=True)

    # Path to save mp3 file
    file_path = os.path.join("static", "output.mp3")

    # Generate speech
    tts = gTTS(text=text, lang=lang)
    tts.save(file_path)

    return file_path

def get_summarizer(source_lang: str, target_lang: str):
    global summarizer_en, summarizer_multi
    if source_lang == "en" and target_lang:
        if summarizer_en is None:
            summarizer_en = pipeline("summarization", model="facebook/bart-large-cnn")
        return summarizer_en
    if summarizer_multi is None:
        model_name = "csebuetnlp/mT5_multilingual_XLSum"
        summarizer_multi = pipeline(
            "summarization",
            model=AutoModelForSeq2SeqLM.from_pretrained(model_name),
            tokenizer=AutoTokenizer.from_pretrained(model_name),
        )
    return summarizer_multi

def get_article_text(url):
    try:
        a = Article(url)
        a.download()
        a.parse()
        if a.text and a.text.strip():
            return (a.title or "Untitled"), a.text
    except Exception:
        pass

    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
        title = soup.title.string.strip() if soup.title else "Untitled"
        return title, text
    except Exception as e:
        return "Error", f"Failed to fetch article: {e}"

def detect_language(text: str) -> str:
    try:
        code = detect(text)
        if code.startswith("ta"):
            return "ta"
        if code.startswith("en"):
            return "en"
        return "en"
    except Exception:
        return "en"

def split_sentences(text: str, lang: str):
    if lang == "ta":
        return [s.strip() for s in re.split(r'(?<=[.?!])\s+|\n+', text) if s.strip()]
    else:
        return nltk.sent_tokenize(text)

def extract_key_sentences(text, lang="en", num_sentences=5):
    sentences = split_sentences(text, lang)
    if not sentences:
        return []
    words = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    freq = Counter(words)

    scores = {}
    for s in sentences:
        for w in re.findall(r"\w+", s.lower(), flags=re.UNICODE):
            scores[s] = scores.get(s, 0) + freq.get(w, 0)

    ranked = sorted(scores, key=scores.get, reverse=True)
    return ranked[:num_sentences]

def bullets_from_sentences(sentences):
    return [f"• {s.strip()}" for s in sentences if s and s.strip()]

def abstractive_bullets(text, source_lang="en", target_lang="en", max_len=130, min_len=50):
    source = text[:2000]
    p = get_summarizer(source_lang, target_lang)
    out = p(source, max_length=max_len, min_length=min_len, do_sample=False)
    summary = out[0]["summary_text"] if isinstance(out, list) else out["summary_text"]
    sentences = split_sentences(summary, target_lang)
    return bullets_from_sentences(sentences)

def extract_key_topics(text, lang="en", top_n=6):
    try:
        from rake_nltk import Rake
        HAS_RAKE = True
    except Exception:
        HAS_RAKE = False

    if lang == "en" and HAS_RAKE:
        rake = Rake()
        rake.extract_keywords_from_text(text)
        phrases = rake.get_ranked_phrases()[:top_n]
        return phrases

    if lang == "en":
        stop = set(nltk.corpus.stopwords.words("english"))
        words = [
            w for w in re.findall(r"\w+", text.lower(), flags=re.UNICODE)
            if w not in stop and len(w) > 3 and not w.isdigit()
        ]
    else:
        words = [
            w for w in re.findall(r"\w+", text.lower(), flags=re.UNICODE)
            if len(w) > 2 and not w.isdigit()
        ]

    counts = Counter(words)
    return [w for w, _ in counts.most_common(top_n)]

def make_summary_package(url, length="default", mode="both", target_lang="en"):
    title, text = get_article_text(url)
    if not text or not text.strip() or title == "Error":
        return {
            "url": url,
            "title": title,
            "summary": ["Could not fetch or summarize this article."],
            "extractive": [],
            "key_topics": [],
            "lang": "en"
        }

    lang = detect_language(text)
    if length == "short":
        max_len, min_len = 80, 30
    elif length == "long":
        max_len, min_len = 200, 80
    else:
        max_len, min_len = 130, 50

    package = {
        "url": url,
        "title": title,
        "summary": [],
        "extractive": [],
        "key_topics": [],
        "lang": lang
    }

    if mode in ("abstractive", "both"):
    # if user selected Tamil, force mT5 model to output Tamil
        summary_lang = target_lang if target_lang in ("ta", "en") else lang
        package["summary"] = abstractive_bullets(
            text,
            source_lang=lang,
            target_lang=summary_lang,
            max_len=max_len,
            min_len=min_len
        )


    if mode in ("extractive", "both"):
        package["extractive"] = bullets_from_sentences(extract_key_sentences(text, lang=lang, num_sentences=5))

    package["key_topics"] = extract_key_topics(text, lang=lang, top_n=6)
    return package

def push_history(url, package):
    session.setdefault("history", [])
    session.setdefault("summaries", {})
    if url in session["history"]:
        session["history"].remove(url)
    session["history"].insert(0, url)
    session["history"] = session["history"][:20]
    session["summaries"][url] = package
    session.modified = True

@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        history=session.get("history", []),
        title=None,
        summary=None,
        extractive=None,
        key_topics=None
    )

# Fix here: accept POST method and parse request JSON properly
@app.route("/summary", methods=["POST"])
def get_summary_api():
    data = request.get_json(force=True)
    url = data.get("url", "")
    language = data.get("language", "en")
    length = data.get("length", "default")
    mode = data.get("mode", "both")
    target_lang = data.get("language", "en")  # default English

    cache = session.get("summaries", {})
    if url in cache:
        return jsonify(cache[url])

    package = make_summary_package(url, length=length, mode=mode, target_lang=target_lang)
    push_history(url, package)
    return jsonify(package)

@app.route('/speak', methods=['POST'])
def speak():
    text = request.form.get('text') or ""   # whatever key you used
    lang = request.form.get('lang', 'en')
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400
    
    os.makedirs("static", exist_ok=True)
    file_path = os.path.join("static", f"output_{hash(text)}.mp3")

    tts = gTTS(text=text, lang=lang)
    tts.save(file_path)
    return jsonify({"audio_file": file_path})

if __name__ == "__main__":
    app.run(debug=True)
