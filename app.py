import streamlit as st
import os
import tempfile
import time
import base64
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="YouTutor · Multilingual AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Dependency checks ────────────────────────────────────────────────────────
try:
    import yt_dlp
except ImportError:
    st.error("Run: pip install yt-dlp"); st.stop()

try:
    import numpy as np
    import faiss
except ImportError:
    st.error("Run: pip install faiss-cpu numpy"); st.stop()

try:
    from groq import Groq
except ImportError:
    st.error("Run: pip install groq"); st.stop()

try:
    from sarvamai import SarvamAI
except ImportError:
    st.error("Run: pip install sarvamai"); st.stop()

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    st.error("Run: pip install sentence-transformers"); st.stop()

import requests

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --ink:    #0d0f1a;
    --paper:  #f5f3ee;
    --accent: #e85d26;
    --muted:  #8a8880;
    --card:   #ffffff;
    --border: #e2dfd8;
    --green:  #2d7a4f;
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background: var(--paper); color: var(--ink); }
h1,h2,h3 { font-family: 'Syne', sans-serif; }

.app-header { display:flex; align-items:center; gap:14px; padding:1.5rem 0 1rem; border-bottom:2px solid var(--ink); margin-bottom:1.5rem; }
.app-header .logo { width:44px; height:44px; background:var(--accent); border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:22px; }
.app-header h1 { font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800; margin:0; letter-spacing:-0.5px; }
.app-header .tagline { font-size:0.78rem; color:var(--muted); text-transform:uppercase; letter-spacing:1.5px; margin-top:2px; }

.pill { display:inline-block; padding:3px 10px; border-radius:20px; font-size:0.72rem; font-weight:600; letter-spacing:0.5px; }
.pill-green  { background:#d4f0e2; color:var(--green); }
.pill-orange { background:#fde8dc; color:var(--accent); }
.pill-grey   { background:#ebebeb; color:var(--muted); }

.chat-wrap { display:flex; flex-direction:column; gap:12px; margin-top:8px; min-height:80px; }
.bubble-user { align-self:flex-end; background:var(--ink); color:#fff; padding:10px 16px; border-radius:18px 18px 4px 18px; max-width:75%; font-size:0.9rem; line-height:1.5; }
.bubble-bot  { align-self:flex-start; background:var(--card); border:1px solid var(--border); color:var(--ink); padding:10px 16px; border-radius:18px 18px 18px 4px; max-width:80%; font-size:0.9rem; line-height:1.6; }
.bubble-lang { font-size:0.68rem; color:var(--muted); margin-top:4px; }

.transcript-box { background:#fafaf8; border:1px solid var(--border); border-radius:8px; padding:0.8rem 1rem; max-height:180px; overflow-y:auto; font-size:0.82rem; line-height:1.7; color:#444; }
.info-box { background:#fff8f4; border-left:3px solid var(--accent); padding:0.6rem 1rem; border-radius:0 8px 8px 0; font-size:0.82rem; color:#666; margin:8px 0; }
.divider { border:none; border-top:1px solid var(--border); margin:1rem 0; }

section[data-testid="stSidebar"] { background:var(--ink) !important; }
section[data-testid="stSidebar"] * { color:#e0ddd5 !important; }
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color:#fff !important; font-family:'Syne',sans-serif !important; }

.stButton > button { background:var(--accent) !important; color:#fff !important; border:none !important; border-radius:8px !important; font-family:'Syne',sans-serif !important; font-weight:700 !important; padding:0.5rem 1.2rem !important; }
.stButton > button:hover { opacity:0.88 !important; }
.stTextInput > div > div > input { border:1.5px solid var(--border) !important; border-radius:8px !important; font-size:0.9rem !important; }
.stTextInput > div > div > input:focus { border-color:var(--accent) !important; box-shadow:0 0 0 2px rgba(232,93,38,0.15) !important; }

#MainMenu, footer, header { visibility:hidden; }
.block-container { padding-top:1rem !important; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
LANGUAGES = {
    "English":   ("en-IN", "en"),
    "Hindi":     ("hi-IN", "hi"),
    "Telugu":    ("te-IN", "te"),
    "Tamil":     ("ta-IN", "ta"),
    "Kannada":   ("kn-IN", "kn"),
    "Malayalam": ("ml-IN", "ml"),
    "Bengali":   ("bn-IN", "bn"),
    "Marathi":   ("mr-IN", "mr"),
    "Gujarati":  ("gu-IN", "gu"),
    "Punjabi":   ("pa-IN", "pa"),
    "Odia":      ("or-IN", "or"),
}

CHUNK_SIZE    = 350
CHUNK_OVERLAP = 50
TOP_K         = 4

# ─── Session state ────────────────────────────────────────────────────────────
for k, v in {
    "transcript": None, "chunks": [], "index": None,
    "embedder": None,   "chat": [],   "video_title": "",
    "q_counter": 0,                          # CHANGE 1: counter to clear input box
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Helpers ──────────────────────────────────────────────────────────────────
def get_key(name):
    if name == "GROQ_API_KEY":
        return st.secrets.get(name, "") or os.getenv(name, "")

    if name == "SARVAM_API_KEY":
        return st.secrets.get(name, "") or os.getenv(name, "")

    return ""

def get_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"

# ─── Audio download ───────────────────────────────────────────────────────────
def download_audio(url, out_dir):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(out_dir, "audio.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
        "ffmpeg_location": get_ffmpeg(),
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "Video")
    return os.path.join(out_dir, "audio.wav"), title

# ─── Audio chunking (for large files) ───────────────────────────────────────
def split_audio(audio_path, chunk_minutes=4):
    import subprocess
    ffmpeg_exe = get_ffmpeg()
    probe = subprocess.run(
        [ffmpeg_exe, "-i", audio_path, "-f", "null", "-"],
        capture_output=True, text=True
    )
    duration = 0.0
    for line in probe.stderr.split("\n"):
        if "Duration" in line:
            try:
                t = line.split("Duration:")[1].split(",")[0].strip().split(":")
                duration = float(t[0])*3600 + float(t[1])*60 + float(t[2])
            except:
                pass
            break
    if duration == 0:
        return [audio_path]

    chunk_secs = chunk_minutes * 60
    chunks, start, idx = [], 0, 0
    base = audio_path.replace(".wav", "")
    while start < duration:
        out = f"{base}_chunk{idx}.wav"
        subprocess.run(
            [ffmpeg_exe, "-y", "-i", audio_path,
             "-ss", str(start), "-t", str(chunk_secs),
             "-ar", "16000", "-ac", "1", out],
            capture_output=True
        )
        if os.path.exists(out):
            chunks.append(out)
        start += chunk_secs
        idx   += 1
    return chunks if chunks else [audio_path]

# ─── STT — Sarvam saaras:v3 with Groq Whisper fallback ──────────────────────
def sarvam_stt(audio_path, lang_code, api_key):
    sarvam_key = get_key("SARVAM_API_KEY")
    groq_key   = get_key("GROQ_API_KEY")

    # Try Sarvam SDK first
    if sarvam_key:
        try:
            from sarvamai import SarvamAI
            client = SarvamAI(api_subscription_key=sarvam_key)
            with open(audio_path, "rb") as f:
                resp = client.speech_to_text.transcribe(
                    file=f,
                    model="saaras:v3",
                    mode="transcribe",
                )
            text = getattr(resp, "transcript", None) or getattr(resp, "text", None) or str(resp)
            if text and text.strip():
                return text.strip()
        except Exception as e:
            st.warning(f"Sarvam STT failed ({e}), falling back to Groq Whisper…")

    # Fallback: Groq Whisper with chunking
    if not groq_key:
        raise Exception("No GROQ_API_KEY found in .env — needed as STT fallback.")

    client   = Groq(api_key=groq_key)
    size_mb  = os.path.getsize(audio_path) / (1024 * 1024)
    chunks   = split_audio(audio_path, chunk_minutes=4) if size_mb > 20 else [audio_path]

    full_transcript = []
    for chunk_path in chunks:
        with open(chunk_path, "rb") as f:
            result = client.audio.transcriptions.create(
                file=(os.path.basename(chunk_path), f.read()),
                model="whisper-large-v3",
                response_format="text",
            )
        text = result if isinstance(result, str) else getattr(result, "text", str(result))
        full_transcript.append(text.strip())
        if chunk_path != audio_path and os.path.exists(chunk_path):
            os.unlink(chunk_path)

    return " ".join(full_transcript)

# ─── TTS — Sarvam bulbul:v3 with gTTS fallback ───────────────────────────────
def sarvam_tts(text, lang_locale, api_key):
    sarvam_key = get_key("SARVAM_API_KEY")

    # Try Sarvam SDK first
    if sarvam_key:
        try:
            from sarvamai import SarvamAI
            client = SarvamAI(api_subscription_key=sarvam_key)
            resp = client.text_to_speech.convert(
                model="bulbul:v3",
                text=text[:500],
                target_language_code=lang_locale,
                speaker="anand",
            )
            if hasattr(resp, "audios") and resp.audios:
                return base64.b64decode(resp.audios[0])
            if hasattr(resp, "audio"):
                raw = resp.audio
                return raw if isinstance(raw, bytes) else base64.b64decode(raw)
        except Exception:
            pass

    # Fallback: gTTS
    try:
        from gtts import gTTS
        import io
        lang_short = lang_locale.split("-")[0]
        tts = gTTS(text=text[:500], lang=lang_short, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        return buf.getvalue()
    except Exception:
        return None

# ─── Translate — Sarvam with deep-translator fallback ────────────────────────
def sarvam_translate(text, src_iso, tgt_iso, api_key):
    if src_iso == tgt_iso or not text.strip():
        return text
    sarvam_key = get_key("SARVAM_API_KEY")

    # Try Sarvam SDK first
    if sarvam_key:
        try:
            from sarvamai import SarvamAI
            src_locale = next((v[0] for v in LANGUAGES.values() if v[1] == src_iso), src_iso)
            tgt_locale = next((v[0] for v in LANGUAGES.values() if v[1] == tgt_iso), tgt_iso)
            client = SarvamAI(api_subscription_key=sarvam_key)
            resp = client.text.translate(
                input=text[:4000],
                source_language_code=src_locale,
                target_language_code=tgt_locale,
            )
            result = getattr(resp, "translated_text", None) or str(resp)
            if result and result.strip():
                return result.strip()
        except Exception:
            pass

    # Fallback: deep-translator
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source=src_iso, target=tgt_iso).translate(text[:4500]) or text
    except Exception:
        return text

# ─── Chunking ─────────────────────────────────────────────────────────────────
def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+size]))
        i += size - overlap
    return chunks

# ─── Embeddings + FAISS ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

def build_index(chunks, embedder):
    vecs = embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs.astype("float32"))
    return index

def retrieve(query, chunks, index, embedder, k=TOP_K):
    qv = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    _, ids = index.search(qv.astype("float32"), k)
    return [chunks[i] for i in ids[0] if i < len(chunks)]

# ─── Groq LLM ────────────────────────────────────────────────────────────────
def groq_answer(question_en, context_chunks, groq_key):
    context = "\n\n---\n\n".join(context_chunks)
    client = Groq(api_key=groq_key)
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": (
                "You are a precise video-content tutor. "
                "Answer ONLY using the transcript excerpts provided. "
                "Do NOT use any external knowledge. "
                "If the answer is not in the excerpts, reply EXACTLY: "
                "'This topic was not covered in the video.'"
            )},
            {"role": "user", "content": f"Transcript excerpts:\n\n{context}\n\nQuestion: {question_en}\n\nAnswer:"},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    return completion.choices[0].message.content.strip()

# ─── Full pipeline ────────────────────────────────────────────────────────────
def process_video(url, sarvam_key, src_lang_code):
    bar = st.progress(0, text="Starting…")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            bar.progress(10, text="⬇️ Downloading audio…")
            wav_path, title = download_audio(url, tmp)
            st.session_state.video_title = title

            bar.progress(40, text="🎙️ Transcribing…")
            transcript = sarvam_stt(wav_path, src_lang_code, sarvam_key)
            if not transcript.strip():
                st.error("Transcription returned empty. Try changing the Video Language.")
                return False
            st.session_state.transcript = transcript

            bar.progress(70, text="✂️ Chunking & building FAISS index…")
            embedder = load_embedder()
            chunks   = chunk_text(transcript)
            index    = build_index(chunks, embedder)

            st.session_state.chunks   = chunks
            st.session_state.index    = index
            st.session_state.embedder = embedder
            st.session_state.chat     = []

            bar.progress(100, text="✅ Ready!")
            time.sleep(0.4)
            bar.empty()
            return True
    except Exception as e:
        bar.empty()
        st.error(f"Error: {e}")
        return False

# ─── UI ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="logo">🎓</div>
  <div>
    <h1>YouTutor</h1>
    <div class="tagline">Multilingual AI · Sarvam SDK + Groq</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Settings row — all inline, no sidebar ────────────────────────────────────
c1, c2, c3, c4 = st.columns([1.3, 1.3, 0.9, 0.9])
with c1:
    video_lang  = st.selectbox("🎬 Video Language", list(LANGUAGES.keys()), index=0)
with c2:
    reply_lang  = st.selectbox("🌐 Reply Language", list(LANGUAGES.keys()), index=0)
with c3:
    voice_reply = st.toggle("🔊 Voice reply", value=False)
voice_input = False  # voice input removed
st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# Read keys
sarvam_key = get_key("SARVAM_API_KEY")
groq_key   = get_key("GROQ_API_KEY")

# Main columns
col_left, col_right = st.columns([1, 1.4], gap="large")

# LEFT — video input
with col_left:
    st.markdown('<div class="card-title" style="font-family:Syne,sans-serif;font-size:0.82rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#8a8880;margin-bottom:0.6rem">📹 Load a YouTube Video</div>', unsafe_allow_html=True)

    yt_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=…", label_visibility="collapsed")
    process_btn = st.button("🚀 Process Video", use_container_width=True)

    if process_btn:
        if not groq_key:
            st.warning("GROQ_API_KEY not found in .env file.")
        elif not yt_url.strip():
            st.warning("Paste a YouTube URL above.")
        else:
            ok = process_video(yt_url.strip(), sarvam_key, LANGUAGES[video_lang][0])
            if ok:
                st.success(f"✅ **{st.session_state.video_title}** — {len(st.session_state.chunks)} chunks indexed.")

    if st.session_state.transcript:
        st.markdown(f"""
<div style="margin-top:8px">
  <span class="pill pill-green">● Transcript ready</span>&nbsp;
  <span class="pill pill-grey">{len(st.session_state.chunks)} chunks</span>&nbsp;
  <span class="pill pill-orange">FAISS indexed</span>
</div>""", unsafe_allow_html=True)
        with st.expander("📄 Transcript preview"):
            preview = st.session_state.transcript[:1200] + ("…" if len(st.session_state.transcript) > 1200 else "")
            st.markdown(f'<div class="transcript-box">{preview}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">Paste a YouTube URL and click <b>Process Video</b>.<br>Audio is extracted, transcribed and indexed for Q&amp;A.</div>', unsafe_allow_html=True)


# RIGHT — chat
with col_right:
    st.markdown('<div style="font-family:Syne,sans-serif;font-size:0.82rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#8a8880;margin-bottom:0.6rem">💬 Ask the Video</div>', unsafe_allow_html=True)

    # Chat bubbles
    chat_html = '<div class="chat-wrap">'
    if not st.session_state.chat:
        chat_html += '<div style="color:#aaa;font-size:0.85rem;text-align:center;padding:2rem 0">Process a video to start asking questions…</div>'
    else:
        for msg in st.session_state.chat:
            if msg["role"] == "user":
                chat_html += f'<div class="bubble-user">{msg["content"]}</div>'
            else:
                chat_html += f'<div class="bubble-bot">{msg["content"]}</div>'
                if msg.get("lang"):
                    chat_html += f'<div class="bubble-lang">Replied in {msg["lang"]}</div>'
    chat_html += "</div>"
    st.markdown(chat_html, unsafe_allow_html=True)

    # Auto-playing audio for last bot reply
    if st.session_state.chat:
        last_bot = next((m for m in reversed(st.session_state.chat) if m["role"] == "assistant"), None)
        if last_bot and last_bot.get("audio"):
            audio_b64 = base64.b64encode(last_bot["audio"]).decode()
            autoplay_html = f"""
<audio id="reply-audio" autoplay style="display:none">
  <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
</audio>
<div style="display:flex;align-items:center;gap:10px;margin:8px 0 4px">
  <button onclick="document.getElementById('reply-audio').play()" style="
    background:#0d0f1a;color:#fff;border:none;border-radius:20px;
    padding:6px 16px;font-size:0.8rem;cursor:pointer">▶ Replay</button>
  <span style="font-size:0.75rem;color:#8a8880">Voice reply ready</span>
</div>"""
            st.components.v1.html(autoplay_html, height=50)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Input — CHANGE 1: use q_counter as key so box clears after submit
    prefill  = st.session_state.pop("prefill_question", "")
    question = st.text_input(
        "Your question",
        value=prefill,
        placeholder="Ask anything about the video…",
        label_visibility="collapsed",
        key=f"q_{st.session_state.q_counter}",
    )
    ask_btn  = st.button("Ask ▶", use_container_width=True)

    if ask_btn and question.strip():
        if not st.session_state.transcript:
            st.warning("Process a video first.")
        elif not groq_key:
            st.warning("Groq API key required.")
        else:
            with st.spinner("Thinking…"):
                reply_locale, reply_iso = LANGUAGES[reply_lang]

                # 1. Translate question → English for retrieval
                if reply_lang != "English" and sarvam_key:
                    try:    q_en = sarvam_translate(question, reply_iso, "en", sarvam_key)
                    except: q_en = question
                else:
                    q_en = question

                # 2. Retrieve top-k chunks
                hits = retrieve(q_en, st.session_state.chunks, st.session_state.index, st.session_state.embedder)

                # 3. LLM answer (English)
                answer_en = groq_answer(q_en, hits, groq_key)

                # 4. Translate answer → reply language
                if reply_lang != "English" and sarvam_key:
                    try:    answer_out = sarvam_translate(answer_en, "en", reply_iso, sarvam_key)
                    except: answer_out = answer_en
                else:
                    answer_out = answer_en

                # 5. Optional TTS
                audio_bytes = None
                if voice_reply:
                    audio_bytes = sarvam_tts(answer_out, reply_locale, sarvam_key)

                st.session_state.chat.append({"role": "user",      "content": question})
                st.session_state.chat.append({"role": "assistant", "content": answer_out, "lang": reply_lang, "audio": audio_bytes})
                st.session_state.q_counter += 1  # CHANGE 1: increment clears the input box

            st.rerun()

    if st.session_state.chat:
        if st.button("🗑️ Clear chat"):
            st.session_state.chat = []
            st.rerun()