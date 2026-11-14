"""Generate a private offline waveform editor for human word timestamps.

Audio-first, no word timing inference; local WAV and source transcript only.
The rendered HTML is private and must never enter the public repository.
"""
from __future__ import annotations

from array import array
import hashlib
import json
import os
from pathlib import Path
import sys
import wave

from .boundary_review import validate_human_interval


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for data in iter(lambda: file.read(1048576), b""):
            h.update(data)
    return h.hexdigest()


def wave_envelope(path: Path, *, clip_sha256: str,
                  duration_ms: int, bin_ms: int = 5) -> list[list[float]]:
    if bin_ms <= 0 or 1000 % bin_ms:
        raise ValueError("Waveform bin must divide 1000 ms")
    if sha256(path) != clip_sha256:
        raise ValueError("Audio clip SHA mismatch")
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(),
                wav.getcomptype()) != (1, 2, 16000, "NONE"):
            raise ValueError("Expected mono 16-kHz PCM16")
        if wav.getnframes() != duration_ms * 16:
            raise ValueError("WAV duration and review interval disagree")
        pcm = array("h")
        pcm.frombytes(wav.readframes(wav.getnframes()))
    if sys.byteorder != "little":
        pcm.byteswap()
    bins = []
    samples = 16 * bin_ms
    for i in range(0, len(pcm), samples):
        section = pcm[i:i+samples]
        bins.append([round(min(section)/32768, 5),
                     round(max(section)/32768, 5)])
    if sha256(path) != clip_sha256:
        raise RuntimeError("Audio changed during waveform extraction")
    return bins


def create_offline_workbench(
    human_review_json: Path,
    clip_wav: Path,
    destination: Path,
    *,
    expected_review_sha256: str,
    expected_clip_sha256: str,
    expected_locked_human_sha256: str,
    original_source_duration_ms: int,
) -> dict:
    if destination.exists():
        raise FileExistsError("Do not overwrite a word-alignment workbench")
    if destination.resolve() in (human_review_json.resolve(), clip_wav.resolve()):
        raise ValueError("Workbench destination cannot be an input")
    if sha256(human_review_json) != expected_review_sha256:
        raise ValueError("Review source hash changed")
    review = json.loads(human_review_json.read_text(encoding="utf-8"))
    if review.get("locked_reference_sha256") != expected_locked_human_sha256:
        raise ValueError("Locked human transcript reference mismatch")
    verified = validate_human_interval(
        review, source_duration_ms=original_source_duration_ms
    )
    if verified["human_timed_word_count"]:
        raise ValueError("Use a new version for already reviewed word timings")
    waveform = wave_envelope(
        clip_wav, clip_sha256=expected_clip_sha256,
        duration_ms=verified["end_ms"]-verified["start_ms"]
    )
    local_path = Path(os.path.relpath(clip_wav, destination.parent)).as_posix()
    if local_path.startswith("http:") or local_path.startswith("https:"):
        raise ValueError("Remote audio URLs forbidden")
    payload = {
        "review": review, "waveform": waveform,
        "review_sha256": expected_review_sha256,
        "clip_sha256": expected_clip_sha256,
        "bin_ms": 5,
    }
    script_data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    script_data = (script_data.replace("&", "\\u0026")
                   .replace("<", "\\u003c").replace(">", "\\u003e"))
    safe_url = (local_path.replace("&", "&amp;").replace('"', "&quot;")
                .replace("<", "&lt;"))
    html = PAGE.replace("__DATA__", script_data).replace("__LOCAL_AUDIO__", safe_url)
    with destination.open("x", encoding="utf-8", newline="\n") as file:
        file.write(html)
    if (sha256(human_review_json) != expected_review_sha256
            or sha256(clip_wav) != expected_clip_sha256):
        raise RuntimeError("Review inputs changed during HTML creation")
    return {
        "schema": "whispercpp/offline-human-word-timing-workbench/v1",
        "html_sha256": sha256(destination),
        "clip_sha256": expected_clip_sha256,
        "frozen_human_reference_sha256": expected_locked_human_sha256,
        "human_word_count": verified["human_word_count"],
        "word_times_prepopulated": 0,
        "source_interval_ms": [verified["start_ms"], verified["end_ms"]],
        "network_requests": 0,
        "automatic_alignment": False,
        "auto_release_authorized": False,
    }


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="referrer" content="no-referrer"><title>Audio-first word alignment</title>
<style>
body{font:15px/1.5 system-ui,Arial,sans-serif;background:#f6f8fa;color:#213047;max-width:1000px;margin:1.2rem auto;padding:0 1rem}
h1{font-size:1.4rem}h2{font-size:1.1rem}
.panel{background:white;border:1px solid #d8e0e7;border-radius:8px;padding:1rem;margin:1rem 0}
.warn{padding:.7rem;border-left:4px solid #b9711a;background:#fff6e6}
.small{color:#526579;font-size:.87rem}
audio{width:100%}canvas{width:100%;height:180px;border:1px solid #a4b2c0;background:#f5faff;cursor:crosshair}
button{border:1px solid #8298ac;border-radius:5px;background:white;padding:.45rem .65rem;cursor:pointer}
button:focus-visible{outline:3px solid #3173a8}
.primary{background:#245c89;color:white}
.controls{display:flex;gap:.5rem;flex-wrap:wrap;align-items:center;margin:.6rem 0}
.wordrow{display:grid;grid-template-columns:2.4rem minmax(110px,1fr) 110px 110px 110px;gap:.4rem;align-items:center;padding:.5rem;border-bottom:1px solid #e2e7ec}
.wordrow.active{background:#edf5fc}.wordrow.partial{border-left:4px solid #be852d}
.wordrow input[type=number]{width:98px;padding:.3rem;border:1px solid #93a3b1;border-radius:3px}
@media(max-width:630px){.wordrow{grid-template-columns:2rem 1fr 1fr}.wordname{grid-column:2/4}}
</style></head>
<body>
<h1>Word-level alignment · audio-first</h1>
<p class="small">Local, offline review. No external libraries, uploads, or network requests.</p>
<p class="warn"><strong>Timing is not prefilled or inferred.</strong> Listen to the
audio to mark each onset/end, in milliseconds on the original recording clock.
The final word is context-completed from a partial fragment; do not label it
fully heard just because its spelling is known. Export does not authorize release.</p>
<div class="panel">
<p id="meta"></p>
<audio id="audio" src="__LOCAL_AUDIO__" controls preload="auto"></audio>
<canvas id="wave" width="1000" height="180" aria-label="Click waveform to seek playback"></canvas>
<div class="controls">
<span id="time">0.000 s</span>
<button id="begin" class="primary">Set selected onset [</button>
<button id="finish" class="primary">Set selected end ]</button>
<button id="play">Replay selected word</button>
<button id="next">Next word</button>
<button id="clear">Clear times</button>
</div>
<p class="small">Click the waveform to seek, use audio playback to hear each boundary.
The [ and ] keys mark onset/end when focus is not in an input.
Every word needs a separate explicit verification checkbox.</p>
</div>
<div class="panel"><h2>Frozen human reference</h2>
<div class="wordrow small"><strong>#</strong><strong class="wordname">Human word</strong><strong>Start ms</strong><strong>End ms</strong><strong>Human verified?</strong></div>
<div id="wordrows"></div><p class="small" id="count"></p></div>
<div class="panel"><div class="controls">
<button id="exportjson" class="primary">Download review JSON</button>
<button id="exportcsv">Download review CSV</button>
</div><p class="small">Store completed outputs as new files in this private folder.
An incomplete export is allowed but remains blocked by the validator.
Do not replace the original frozen word list or locked CSV.</p></div>
<script id="data" type="application/json">__DATA__</script>
<script>
"use strict";
const pack=JSON.parse(document.getElementById("data").textContent);
const source=pack.review,words=source.words.map(w=>Object.assign({},w));
const start=source.start_ms,end=source.end_ms,duration=end-start;
const audio=document.getElementById("audio"),canvas=document.getElementById("wave"),ctx=canvas.getContext("2d");
let selected=0;const nodes=[];
document.getElementById("meta").textContent="Original source clock: "+start+"–"+end+
 " ms. "+words.length+" human words. Audio SHA "+pack.clip_sha256.substring(0,16)+"…";
function clock(){return start+Math.max(0,Math.min(duration,Math.round(audio.currentTime*1000)))}
function progress(){document.getElementById("count").textContent=
 words.filter(w=>w.human_timing_verified===true).length+"/"+words.length+
 " words explicitly verified; remaining words are not aligned ground truth."}
function select(i){selected=i;nodes.forEach((n,k)=>n.classList.toggle("active",k===i));draw()}
function refresh(i){
 const w=words[i],n=nodes[i];
 n.querySelector(".onset").value=w.word_start_ms===null?"":w.word_start_ms;
 n.querySelector(".offset").value=w.word_end_ms===null?"":w.word_end_ms;
 n.querySelector(".verified").checked=w.human_timing_verified===true;
 progress();
}
function valid(v){return Number.isInteger(v)&&v>=start&&v<=end}
function draw(){
 const w=canvas.width,h=canvas.height,mid=h/2;
 ctx.clearRect(0,0,w,h);ctx.fillStyle="#f5faff";ctx.fillRect(0,0,w,h);
 ctx.strokeStyle="#486780";ctx.beginPath();
 pack.waveform.forEach((v,i)=>{
  const x=i/Math.max(1,pack.waveform.length-1)*w;
  ctx.moveTo(x,mid-v[1]*h*.43);ctx.lineTo(x,mid-v[0]*h*.43);
 });ctx.stroke();
 words.forEach((a,i)=>{
  if(a.word_start_ms!==null&&a.word_end_ms!==null){
   ctx.fillStyle=i===selected?"rgba(244,180,64,.25)":"rgba(60,120,174,.12)";
   ctx.fillRect((a.word_start_ms-start)/duration*w,0,
    (a.word_end_ms-a.word_start_ms)/duration*w,h)
  }
 });
 ctx.strokeStyle="#c34438";ctx.lineWidth=2;ctx.beginPath();
 const x=Math.min(w,Math.max(0,audio.currentTime*1000/duration*w));
 ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();ctx.lineWidth=1;
}
function settime(key,val){
 const w=words[selected];
 if(!valid(val)){alert("Use original source-clock integer milliseconds between "+start+" and "+end);return}
 w[key]=val;w.human_timing_verified=false;refresh(selected);draw()
}
function build(){
 const area=document.getElementById("wordrows");
 words.forEach((word,i)=>{
  const row=document.createElement("div");row.className="wordrow"+(word.heard_fragment?" partial":"");
  const no=document.createElement("span");no.textContent=String(i+1);
  const name=document.createElement("strong");name.className="wordname";
  name.textContent=word.transcribed_word+(word.heard_fragment?" [heard: "+word.heard_fragment+"]":"");
  const onset=document.createElement("input");onset.type="number";onset.step="1";onset.min=start;onset.max=end;onset.className="onset";
  const offset=document.createElement("input");offset.type="number";offset.step="1";offset.min=start;offset.max=end;offset.className="offset";
  const label=document.createElement("label");
  const tick=document.createElement("input");tick.type="checkbox";tick.className="verified";
  label.append(tick,document.createTextNode(" listened"));
  row.append(no,name,onset,offset,label);area.appendChild(row);nodes.push(row);
  no.addEventListener("click",()=>select(i));name.addEventListener("click",()=>select(i));
  onset.addEventListener("change",()=>{
   select(i);if(onset.value===""){word.word_start_ms=null;word.human_timing_verified=false;refresh(i);return}
   settime("word_start_ms",Number(onset.value));
  });
  offset.addEventListener("change",()=>{
   select(i);if(offset.value===""){word.word_end_ms=null;word.human_timing_verified=false;refresh(i);return}
   settime("word_end_ms",Number(offset.value));
  });
  tick.addEventListener("change",()=>{
   if(tick.checked){
    if(!valid(word.word_start_ms)||!valid(word.word_end_ms)||
        word.word_start_ms>=word.word_end_ms){
      alert("Set valid onset/end times first");tick.checked=false;
    }else{
      word.human_timing_verified=true
    }
   }else{word.human_timing_verified=false}
   progress();
  });
  refresh(i)
 });
 select(0)
}
canvas.addEventListener("pointerdown",ev=>{
 const r=canvas.getBoundingClientRect();
 const f=Math.max(0,Math.min(1,(ev.clientX-r.left)/r.width));
 audio.currentTime=f*duration/1000;draw();
});
document.getElementById("begin").onclick=()=>settime("word_start_ms",clock());
document.getElementById("finish").onclick=()=>settime("word_end_ms",clock());
document.getElementById("next").onclick=()=>select(Math.min(words.length-1,selected+1));
document.getElementById("clear").onclick=()=>{
 words[selected].word_start_ms=null;words[selected].word_end_ms=null;
 words[selected].human_timing_verified=false;refresh(selected);draw()
};
document.getElementById("play").onclick=()=>{
 const w=words[selected];
 if(!valid(w.word_start_ms)||!valid(w.word_end_ms)){alert("Set both times first");return}
 audio.currentTime=(w.word_start_ms-start)/1000;
 const stop=()=>{
  if(audio.currentTime>=(w.word_end_ms-start)/1000){audio.pause();audio.removeEventListener("timeupdate",stop)}
 };
 audio.addEventListener("timeupdate",stop);audio.play().catch(()=>{});
};
document.addEventListener("keydown",ev=>{
 if(["INPUT","TEXTAREA","BUTTON"].includes(document.activeElement.tagName))return;
 if(ev.key==="["){settime("word_start_ms",clock());ev.preventDefault()}
 if(ev.key==="]"){settime("word_end_ms",clock());ev.preventDefault()}
});
audio.addEventListener("timeupdate",()=>{
 document.getElementById("time").textContent=audio.currentTime.toFixed(3)+" s · "+clock()+" ms";
 draw()
});
function download(filename,content,mime){
 const blob=new Blob([content],{type:mime}),url=URL.createObjectURL(blob);
 const a=document.createElement("a");a.href=url;a.download=filename;
 document.body.appendChild(a);a.click();a.remove();
 setTimeout(()=>URL.revokeObjectURL(url),1000);
}
document.getElementById("exportjson").onclick=()=>{
 const result=Object.assign({},source,{
  words:words.map(w=>Object.assign({},w)),
  review_status:"HUMAN_TIMING_WORKSHEET_NOT_EDITORIAL_APPROVAL",
  original_inventory_sha256:pack.review_sha256,
  original_audio_clip_sha256:pack.clip_sha256,
  auto_release_authorized:false
 });
 download("G01_HUMAN_WORD_TIMING_REVIEW_v2.json",JSON.stringify(result,null,2)+"\n","application/json")
};
document.getElementById("exportcsv").onclick=()=>{
 const keys=["index","transcribed_word","audibility","heard_fragment","word_start_ms","word_end_ms","human_timing_verified"];
 const field=x=>'"'+String(x===null?"":x).replace(/"/g,'""')+'"';
 const rows=[keys.join(",")].concat(words.map(w=>keys.map(k=>field(w[k])).join(",")));
 download("G01_HUMAN_WORD_TIMING_REVIEW_v2.csv","\uFEFF"+rows.join("\r\n")+"\r\n","text/csv;charset=utf-8")
};
build();
</script></body></html>
"""