# main.py
import os, io, re, math, typing
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydub import AudioSegment
from openai import OpenAI

# ===== 基本設定 =====
APP_TITLE = "Timeout-ComboA-API (WAV conversion)"
STT_MODEL = "gpt-4o-transcribe"           # or gpt-4o-mini-transcribe
EMB_MODEL = "text-embedding-3-large"
CANONICAL_THRESHOLD = 0.80

CANONICAL_SENTENCES: List[str] = [
    "Is everyone ready to begin the timeout?",
    "Do we have the consent form in front of us?",
    "Please introduce yourselves.",
    "Is the attending physician present?",
    "Who is the anesthesia attending?",
    "What are the names and roles of the other team members?",
    "What is your full name?",
    "What is your date of birth?",
    "What is your Medical Record Number?",
    "Is the consent form signed?",
    "What surgery and/or block is being performed?",
    "Which side?",
    "Is it marked?",
    "Which local anesthetic will be used?",
    "What is the intended concentration?",
    "What is the intended volume?",
    "Are you currently taking any anticoagulants?",
    "Do you have any clotting disorders?",
    "Do you have any known drug allergies?",
    "Do you have a history of systemic neuropathy and/or neuropathy at the surgical site?",
    "Is NIBP monitoring ready?",
    "Is ECG monitoring ready?",
    "Is SpO₂ ready?",
    "Is EtCO₂ monitoring ready?",
    "Does anyone have any other question or concern?",
    "Timeout completed."
]

DOMAIN_PROMPT = (
    "Transcribe in English only. This is an operating room timeout checklist. "
    "Each spoken sentence corresponds closely to one of the following lines. "
    "Never output fragments; always transcribe until a full sentence is heard. "
    "If uncertain, wait for more audio before finalizing.\n\n"
    + "\n".join(f"- {s}" for s in CANONICAL_SENTENCES)
)

SENT_END_PATTERN = re.compile(r'(.+?[\.!\?][\"\')\]]?\s+)')

# ===== App / CORS =====
app = FastAPI(title=APP_TITLE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 上線後建議改成你的網域
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== OpenAI =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing env OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ===== Embeddings cache =====
_canon_embeds: List[List[float]] = []

def _embed_texts(texts: List[str]) -> List[List[float]]:
    r = client.embeddings.create(model=EMB_MODEL, input=texts)
    return [d.embedding for d in r.data]

def _ensure_canon_embeds():
    global _canon_embeds
    if not _canon_embeds:
        _canon_embeds = _embed_texts(CANONICAL_SENTENCES)

def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def _segment_sentences(text: str) -> List[str]:
    outs, buf = [], text.strip()
    while True:
        m = SENT_END_PATTERN.match(buf)
        if not m: break
        s = m.group(1).strip()
        outs.append(s)
        buf = buf[len(m.group(1)):]
    # 過濾太短的半句，避免誤判
    return [s for s in outs if len(s.split()) >= 4]

# ====== 型別 ======
class ChunkResponse(BaseModel):
    hits: List[str]
    raw: List[str]
    suggestions: List[Dict[str, typing.Any]]

# ====== 轉檔：任意瀏覽器錄音 -> WAV bytes ======
# 需要 ffmpeg（我們會在 apt.txt 安裝）
EXT_BY_MIME = {
    "audio/webm": "webm",
    "audio/ogg": "ogg",
    "audio/mp4": "mp4",
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
}

def bytes_to_wav_bytes(data: bytes, content_type: str | None, filename: str | None) -> io.BytesIO:
    if not data or len(data) < 1000:
        raise HTTPException(status_code=400, detail="empty or too small audio")
    # 推斷格式
    fmt = EXT_BY_MIME.get(content_type or "", None)
    if not fmt and filename and "." in filename:
        fmt = filename.rsplit(".", 1)[-1].lower()
    # 嘗試解碼
    src = io.BytesIO(data)
    try:
        if fmt:
            seg = AudioSegment.from_file(src, format=fmt)
        else:
            seg = AudioSegment.from_file(src)  # 讓 ffmpeg 自動偵測
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"cannot decode audio: {e}")

    # 輸出為 WAV（16-bit PCM），Whisper/STT 最穩
    wav_io = io.BytesIO()
    seg.export(wav_io, format="wav")
    wav_io.seek(0)
    return wav_io

# ====== 路由 ======
@app.get("/health")
def health():
    return {"ok": True, "model": STT_MODEL, "emb": EMB_MODEL}

@app.get("/canon")
def canon():
    return {"sentences": CANONICAL_SENTENCES}

@app.post("/transcribe-chunk", response_model=ChunkResponse)
async def transcribe_chunk(audio: UploadFile = File(...)):
    # 1) 讀 bytes
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="no audio uploaded")

    # 2) 轉成 WAV
    wav_bytes = bytes_to_wav_bytes(
        data=data,
        content_type=audio.content_type,
        filename=audio.filename
    )

    # 3) OpenAI STT（帶防半句的 prompt）
    with wav_bytes as f:
        f.name = "chunk.wav"
        stt = client.audio.transcriptions.create(
            model=STT_MODEL,
            file=f,
            temperature=0,
            language="en",
            prompt=DOMAIN_PROMPT
        )
    rough = (stt.text or "").strip()
    if not rough:
        return {"hits": [], "raw": [], "suggestions": []}

    # 4) 本地斷句
    sents = _segment_sentences(rough)
    if not sents:
        return {"hits": [], "raw": [], "suggestions": []}

    # 5) Embeddings 映射到標準句
    _ensure_canon_embeds()
    q_embs = _embed_texts(sents)
    hits, suggestions = [], []
    for s, q in zip(sents, q_embs):
        best_idx, best_score = -1, -1.0
        for i, c in enumerate(_canon_embeds):
            sim = _cosine(q, c)
            if sim > best_score:
                best_idx, best_score = i, sim
        if best_score >= CANONICAL_THRESHOLD:
            hits.append(CANONICAL_SENTENCES[best_idx])
        else:
            suggestions.append({
                "raw": s,
                "best": CANONICAL_SENTENCES[best_idx],
                "score": round(best_score, 2)
            })

    return {"hits": hits, "raw": sents, "suggestions": suggestions}
