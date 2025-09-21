# app.py — FastAPI + OpenAI STT + Embeddings（Render 可用｜方案A）
import os, io, re, uuid
from typing import Dict, List, Tuple
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from openai import OpenAI
from math import sqrt

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("⚠️  OPENAI_API_KEY is not set")

client = OpenAI(api_key=OPENAI_API_KEY)

# ======= 方案A的模型與門檻 =======
STT_MODEL = "gpt-4o-transcribe"
EMB_MODEL = "text-embedding-3-large"
CANONICAL_THRESHOLD = 0.85
TERMINATION_SENTENCE = "Timeout completed."

CANONICAL_SENTENCES = [
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
    "Prefer complete sentences; avoid partial fragments. "
    "If the spoken sentence is semantically equivalent to ANY of the following, "
    "output EXACTLY that canonical sentence, word-for-word:\n\n"
    + "\n".join(f"- {s}" for s in CANONICAL_SENTENCES)
)

# ======= FastAPI + CORS =======
app = FastAPI()
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======= Session 狀態（記憶體；正式可換 Redis）=======
class SessionState:
    def __init__(self):
        self.buffer = ""          # 滾動粗轉錄（用來切句）
        self.printed = set()      # 已命中的標準句（避免重複）
        self.ok_lines: List[str] = []  # 命中的標準句（匯出用）

sessions: Dict[str, SessionState] = {}

# ======= Embeddings 工具 =======
def embed_texts(texts: List[str]):
    r = client.embeddings.create(model=EMB_MODEL, input=texts)
    return [d.embedding for d in r.data]

def cosine_sim(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = sqrt(sum(x*x for x in a)); nb = sqrt(sum(y*y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot/(na*nb)

# 啟動時把 canonical embeddings 先算好（冷啟動一次）
CANON_EMB = embed_texts(CANONICAL_SENTENCES)

def best_match(sentence: str) -> Tuple[str, float]:
    """對一句粗轉錄做語意比對，回 (最佳標準句, 相似度)"""
    q = embed_texts([sentence])[0]
    best_i, best_s = -1, -1.0
    for i, c in enumerate(CANON_EMB):
        s = cosine_sim(q, c)
        if s > best_s:
            best_i, best_s = i, s
    return CANONICAL_SENTENCES[best_i], best_s

# ======= 句界（. ? ! 後允許引號/括號）=======
SENT_END = re.compile(r'(.+?[\.!\?][\"\')\]]?\s+)')

def take_sentences(state: SessionState, new_text: str) -> List[str]:
    outs = []
    state.buffer = (state.buffer + " " + new_text.strip()).strip() if new_text else state.buffer
    while True:
        m = SENT_END.match(state.buffer)
        if not m: break
        sent = m.group(1).strip()
        outs.append(sent)
        state.buffer = state.buffer[len(m.group(1)):]
    if len(state.buffer) > 1000:
        state.buffer = state.buffer[-1000:]
    return outs

# ======= Routes =======
@app.get("/")
def root():
    return {"ok": True, "service": "timeout-checklist-server", "mode": "ComboA"}

@app.post("/start")
def start_session():
    sid = str(uuid.uuid4())
    sessions[sid] = SessionState()
    return {"session_id": sid}

@app.post("/chunk")
async def process_chunk(session_id: str = Form(...), audio: UploadFile = Form(...)):
    """前端每 2.5s 丟一段 audio/webm；回本次新命中的標準句（hits），以及是否終止。"""
    st = sessions.get(session_id)
    if not st:
        return JSONResponse({"error": "invalid session"}, status_code=400)

    data = await audio.read()
    if not data or len(data) < 600:
        return {"hits": [], "terminate": False, "warn": "empty-or-too-small"}

    # 丟給 STT（OpenAI 支援 webm/opus，無需轉檔）
    bio = io.BytesIO(data); bio.name = audio.filename or "chunk.webm"
    try:
        r = client.audio.transcriptions.create(
            model=STT_MODEL, file=bio, language="en", temperature=0, prompt=DOMAIN_PROMPT
        )
        rough = (r.text or "").strip()
    except Exception as e:
        # 避免 500，回 200 + error，前端不中斷
        return {"hits": [], "terminate": False, "error": str(e)}

    hits = []
    if rough:
        # 切句 → 過短句先跳過（多半是碎片）
        for s in take_sentences(st, rough):
            if len(s.split()) < 4:
                continue
            best, score = best_match(s.lower().replace("  ", " "))
            if score >= CANONICAL_THRESHOLD and best not in st.printed:
                st.printed.add(best)
                st.ok_lines.append(best)
                hits.append({"sentence": best, "score": round(score, 2)})

    terminate = (TERMINATION_SENTENCE in st.printed)
    return {"hits": hits, "terminate": terminate}

@app.get("/export/{session_id}")
def export_text(session_id: str):
    st = sessions.get(session_id)
    if not st:
        return JSONResponse({"error": "invalid session"}, status_code=400)
    text = "\n".join(st.ok_lines) + ("\n" if st.ok_lines else "")
    return PlainTextResponse(text, media_type="text/plain; charset=utf-8")

@app.post("/reset")
def reset_session(session_id: str = Form(...)):
    if session_id in sessions:
        del sessions[session_id]
    return {"ok": True}
