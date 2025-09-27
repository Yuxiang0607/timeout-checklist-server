import io, os, time, math, re, uuid
from typing import List, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from openai import OpenAI

# ====== 配置 ======
API_KEY = os.environ.get("OPENAI_API_KEY")  # 在 Render Dashboard 設定環境變數
STT_MODEL = "gpt-4o-transcribe"
EMB_MODEL = "text-embedding-3-large"
CANONICAL_THRESHOLD = 0.80
SAVE_DIR = "transcripts"
os.makedirs(SAVE_DIR, exist_ok=True)

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
    "Each spoken sentence may match one of the lines below.\n"
    "Do NOT guess or infer. If no clear speech is present, return exactly: [NO_SPEECH]. "
    "If the speech is not clearly one of the lines, transcribe literally; do not normalize.\n\n"
    + "\n".join(f"- {s}" for s in CANONICAL_SENTENCES)
)


client = OpenAI(api_key=API_KEY)

app = FastAPI(title="OR Timeout STT")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 部署後可改成你的前端網域
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def embed_texts(texts: List[str]) -> List[List[float]]:
    r = client.embeddings.create(model=EMB_MODEL, input=texts)
    return [d.embedding for d in r.data]

def cosine_sim(a, b) -> float:
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0: return 0.0
    return dot / (na*nb)

# 預先算 canonical embeddings（啟動時）
CANONICAL_EMB = embed_texts(CANONICAL_SENTENCES)

SENT_END_PATTERN = re.compile(r'(.+?[\.!\?][\"\')\]]?\s+)')

def split_sentences(text: str) -> List[str]:
    outs, buf = [], text.strip()
    while True:
        m = SENT_END_PATTERN.match(buf)
        if not m: break
        sent = m.group(1).strip()
        outs.append(sent)
        buf = buf[len(m.group(1)):]
    if buf.strip():
        outs.append(buf.strip())
    return outs

def best_match(sentence: str):
    q = embed_texts([sentence])[0]
    best_i, best_s = -1, -1.0
    for i, c in enumerate(CANONICAL_EMB):
        s = cosine_sim(q, c)
        if s > best_s:
            best_i, best_s = i, s
    return CANONICAL_SENTENCES[best_i], float(best_s)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    """
    接收前端錄音（webm/ogg/wav/m4a/mp3…），丟給 OpenAI STT。
    回傳：
      - matched: 命中的 canonical 清單（去重、保持首次出現順序）
      - coverage: {canonical_sentence: true/false}
      - transcript_url: 純淨稿（只含命中句）的 txt 下載位址
      - raw_text: STT 原文（除錯用）
    """
    if not API_KEY:
        raise HTTPException(500, "Missing OPENAI_API_KEY")

    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file")

    # 直接把 WebM/OGG/WAV bytes 丟給 STT；OpenAI 會自動解碼
    try:
        f = io.BytesIO(content); f.name = file.filename or "audio.webm"
        r = client.audio.transcriptions.create(
            model=STT_MODEL,
            file=f,
            temperature=0,
            language="en",
            prompt=DOMAIN_PROMPT
        )
        rough_text = (r.text or "").strip()
    except Exception as e:
        raise HTTPException(500, f"STT error: {e}")

    # === 低資訊保護：太短或沒有關鍵詞就直接視為沒有命中 ===
    MIN_WORDS = 3
    KEYWORDS = {"timeout","consent","introduce","attending","anesthesia",
                "name","birth","record","signed","surgery","block","side",
                "marked","local","concentration","volume","anticoagulants",
                "clotting","allergies","neuropathy","nibp","ecg","spo2","etco2",
                "question","concern","completed"}
    words = rough_text.lower().split()
    if len(words) < MIN_WORDS or not (set(words) & KEYWORDS):
        return JSONResponse({
            "matched": [],
            "coverage": {s: False for s in CANONICAL_SENTENCES},
            "transcript_url": None,
            "raw_text": rough_text
        })

    # 句界切分 + 映射到 canonical
    printed, ok_lines = set(), []
    coverage = {s: False for s in CANONICAL_SENTENCES}

    for s in split_sentences(rough_text):
        # 已與 canonical 完全相同直接接受
        if s in CANONICAL_SENTENCES:
            cand, score, matched = s, 1.0, True
        else:
            cand, score = best_match(s)
            matched = (score >= CANONICAL_THRESHOLD)

        if matched and (cand not in printed):
            printed.add(cand)
            ok_lines.append(cand)
            coverage[cand] = True

    # 儲存純淨稿 txt
    uid = uuid.uuid4().hex[:12]
    txt_path = os.path.join(SAVE_DIR, f"{uid}.txt")
    with open(txt_path, "w", encoding="utf-8") as w:
        w.write("\n".join(ok_lines) + "\n")

    return JSONResponse({
        "matched": ok_lines,
        "coverage": coverage,
        "transcript_url": f"/transcript/{uid}.txt",
        "raw_text": rough_text
    })

@app.get("/transcript/{name}")
def get_transcript(name: str):
    path = os.path.join(SAVE_DIR, name)
    if not os.path.isfile(path):
        raise HTTPException(404, "Not found")
    return FileResponse(path, media_type="text/plain", filename=name)
