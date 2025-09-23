# app.py
import os, io, re, math
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# ===== 設定 =====
STT_MODEL = "gpt-4o-transcribe"
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

# ===== FastAPI & CORS =====
app = FastAPI(title="Timeout-ComboA-API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 上線後建議改成你的前端網域
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== OpenAI Client =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing env OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ===== Embeddings Cache =====
_canon_embeds: List[List[float]] = []

def _embed_texts(texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(model=EMB_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def _ensure_canon_embeds():
    global _canon_embeds
    if not _canon_embeds:
        _canon_embeds = _embed_texts(CANONICAL_SENTENCES)

def _cos(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0: return 0.0
    return dot/(na*nb)

def _segment_sentences(text: str) -> List[str]:
    outs, buf = [], text.strip()
    while True:
        m = SENT_END_PATTERN.match(buf)
        if not m: break
        s = m.group(1).strip()
        outs.append(s)
        buf = buf[len(m.group(1)):]
    # 過短片段忽略，避免半句
    return [s for s in outs if len(s.split()) >= 4]

class ChunkResp(BaseModel):
    hits: list[str]
    raw: list[str]
    suggestions: list[dict]

# ====== 路由 ======
@app.get("/")
def root():
    return {
        "message": "Timeout Checklist Server is running.",
        "try": ["/health", "/canon", "POST /transcribe-chunk"]
    }

@app.get("/health")
def health():
    return {"ok": True, "model": STT_MODEL, "emb": EMB_MODEL}

@app.get("/canon")
def canon():
    return {"sentences": CANONICAL_SENTENCES}

@app.post("/transcribe-chunk", response_model=ChunkResp)
async def transcribe_chunk(audio: UploadFile = File(...)):
    content = await audio.read()
    if not content:
        raise HTTPException(status_code=400, detail="empty audio")

    # 1) STT
    with io.BytesIO(content) as f:
        f.name = audio.filename or "chunk.webm"
        r = client.audio.transcriptions.create(
            model=STT_MODEL,
            file=f,
            temperature=0,
            language="en",
            prompt=DOMAIN_PROMPT
        )
    rough = (r.text or "").strip()
    if not rough:
        return {"hits": [], "raw": [], "suggestions": []}

    # 2) 斷句
    sents = _segment_sentences(rough)
    if not sents:
        return {"hits": [], "raw": [], "suggestions": []}

    # 3) Embeddings 對映
    _ensure_canon_embeds()
    q_embs = _embed_texts(sents)
    hits, suggestions = [], []
    for s, q in zip(sents, q_embs):
        best_idx, best_score = -1, -1.0
        for i, c in enumerate(_canon_embeds):
            sim = _cos(q, c)
            if sim > best_score:
                best_idx, best_score = i, sim
        if best_score >= CANONICAL_THRESHOLD:
            hits.append(CANONICAL_SENTENCES[best_idx])
        else:
            suggestions.append({"raw": s, "best": CANONICAL_SENTENCES[best_idx], "score": round(best_score, 2)})

    return {"hits": hits, "raw": sents, "suggestions": suggestions}
