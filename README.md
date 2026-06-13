# YouTutor — Multilingual AI Video Tutor

YouTutor transforms any YouTube video into an interactive multilingual tutor, enabling users to ask questions in their preferred language and receive grounded answers from the video's content. The app extracts audio from YouTube videos, generates transcripts, retrieves relevant context using FAISS, and produces grounded answers using Groq LLMs. It also supports multilingual translation and optional voice replies.

It is designed for students, multilingual learners, and users who want to interact with educational video content more effectively through grounded AI-based question answering.

---
## Demo

Watch the project demonstration here:

https://drive.google.com/file/d/17r831je4IdrPgwLr-mboCxrQy6vDB8qR/view?usp=sharing

---
## Features

* YouTube audio extraction using `yt-dlp` and `ffmpeg`
* Speech-to-text using Sarvam AI with Groq Whisper fallback
* RAG pipeline using MiniLM embeddings and FAISS
* Grounded answers generated using Groq `llama-3.3-70b-versatile`
* Multilingual translation with fallback support
* Optional voice replies using Sarvam TTS and gTTS
* Multilingual support for major Indian languages
* Chat-style Streamlit interface
* Automatic audio chunking for long videos

---

## Tech Stack

* Streamlit
* yt-dlp
* ffmpeg
* Sarvam AI
* Groq
* FAISS
* Sentence Transformers

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
SARVAM_API_KEY=your_key
GROQ_API_KEY=your_key
```

Run the app:

```bash
streamlit run app.py
```

---

## Core Pipeline

```text
YouTube Video
    ↓
Audio Extraction
    ↓
Transcription
    ↓
Chunking
    ↓
Embeddings
    ↓
FAISS Retrieval
    ↓
Groq LLM
    ↓
Multilingual Response
```

The app uses fallback systems for transcription, translation, and text-to-speech to improve reliability.

---

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    ├── config.toml
    └── secrets.toml.example
```
---

## Future Improvements

- PDF and document support
- Chat history persistence
- Video timestamp citations
- Multi-video knowledge base

---
## Important Notes

* Remove hardcoded API keys before deployment
* The embedding model downloads automatically during first run
* Long videos are processed in smaller chunks when needed

---

## Troubleshooting

| Problem               | Solution                                   |
| --------------------- | ------------------------------------------ |
| `ffmpeg not found`    | Install ffmpeg and add it to PATH          |
| Empty transcript      | Change the selected video language         |
| Slow first launch     | Embedding model downloads during first run |
| Authentication errors | Verify API keys                            |

---

## License

This project is intended for educational purposes.
