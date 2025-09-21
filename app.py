# app.py  — FastAPI + OpenAI STT（Render 可用）
import os
import io
import uuid
from typing import Dict, List

from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from openai import OpenAI

# 讀環境變數（請在 Render 的 Environment Variables 設 OPENAI_API_KEY）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # 在雲端若沒設好，至少回可讀錯誤
    print("⚠️  OPENAI_API_KEY is not set")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

# CORS：開發先 *，上線改成你的網域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Canonical sentences（先保留；此版只做 STT，不做 embeddings 對映）=====
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
    "Timeout completed.",
]

DOMAIN_PROMPT = (
    "Transcribe in English only. This is an operating room timeout checklist. "
    "Prefer complete sentences; avoid partial fragments. "
    "If the spoken sentence is semantically equivalent to ANY of the following, "
    "output EXACTLY that canonical sentence, word-for-word:\n\n"
    + "\n".join(f"- {s}" for s in CANONICAL_SENTENCES)
)

# ====== 簡單的 session 記憶體存放（開發用；正式可換 Redis）======
SESSIONS: Dict[str, List[str]] = {}

@app.get("/")
def root():
    return {"message": "Server running OK 🚀"}

@app.post("/start")
def start_session():
    sid = str(uuid.uuid4())
    SESSIONS[sid] = []
    return {"session_id": sid}

@app.post("/chunk")
async def process_chunk(session_id: str = Form(...), audio: UploadFile = Form(...)):
    """
    接收前端 1 段音訊（建議 webm/opus）
    送到 OpenAI STT → 回傳文字
    """
    if session_id not in SESSIONS:
        return JSONResponse({"error": "invalid session"}, status_code=400)

    try:
        data = await audio.read()
    except Exception as e:
        return JSONResponse({"error": f"read-audio-failed: {e}"}, status_code=200)

    if not data or len(data) < 600:
        return {"recognized": "", "session_id": session_id, "warn": "empty-or-too-small"}

    bio = io.BytesIO(data)
    bio.name = "chunk.webm"  # 讓 SDK 知道是 webm/opus

    try:
        r = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=bio,
            language="en",
            temperature=0,
            prompt=DOMAIN_PROMPT,
        )
        text = (r.text or "").strip()
        # Debug 看看返回
        # print("STT:", text)
    except Exception as e:
        # 不要 500，回 200 + 錯誤訊息，前端不中斷
        return {"recognized": "", "session_id": session_id, "error": str(e)}

    if text:
        SESSIONS[session_id].append(text)

    return {"recognized": text, "session_id": session_id}

@app.post("/reset")
def reset_session(session_id: str = Form(...)):
    SESSIONS[session_id] = []
    return {"ok": True}

@app.get("/export/{session_id}")
def export_text(session_id: str):
    if session_id not in SESSIONS:
        return JSONResponse({"error": "invalid session"}, status_code=400)
    content = "\n".join(SESSIONS[session_id])
    return Response(content, media_type="text/plain")
