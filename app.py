let mediaRecorder, chunks = [];
let recTimer = null;

async function startRec(){
  try{
    try { if ("speechSynthesis" in window) speechSynthesis.cancel(); } catch(e) {}

    const stream = await navigator.mediaDevices.getUserMedia({audio:true});
    chunks = [];
    mediaRecorder = new MediaRecorder(stream, {mimeType: "audio/webm"});
    mediaRecorder.ondataavailable = e => { if (e.data.size>0) chunks.push(e.data); };
    mediaRecorder.onstop = onStopAndUpload;
    mediaRecorder.start();
    log("🎙️ Recording started… (max 20s)");
    document.getElementById("btnStart").disabled = true;
    document.getElementById("btnStop").disabled = false;

    // 最多 20 秒自動停止
    recTimer = setTimeout(() => {
      if (mediaRecorder && mediaRecorder.state !== "inactive") {
        log("⏱️ Auto stop (20s limit)");
        stopRec();
      }
    }, 20000);

  }catch(err){
    log("Microphone error: " + err);
  }
}
document.getElementById("btnStart").onclick = startRec;

function stopRec(){
  if (recTimer) {
    clearTimeout(recTimer);
    recTimer = null;
  }
  if (mediaRecorder && mediaRecorder.state !== "inactive"){
    mediaRecorder.stop();
    document.getElementById("btnStop").disabled = true;
    document.getElementById("btnStart").disabled = false;
  }
}
