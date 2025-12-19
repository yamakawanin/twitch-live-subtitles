import subprocess
import numpy as np
import time
import threading
from datetime import datetime
import uuid
import html
import os
import base64
from collections import deque
import queue
from difflib import SequenceMatcher

import torch
import webrtcvad
from faster_whisper import WhisperModel

from transformers import MarianMTModel, MarianTokenizer
import gradio as gr

# ======================
# CONFIG & ASSETS
# ======================
BG_IMAGE_NAME = "farewall.jpg"


def get_b64_image(image_path):
    if not os.path.exists(image_path):
        print(f"Warning: Background image '{image_path}' not found.")
        return None
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{data}"


bg_data = get_b64_image(BG_IMAGE_NAME)
bg_css_rule = ""
if bg_data:
    bg_css_rule = f"""
    .gradio-container {{
        background-image: url('{bg_data}') !important;
        background-size: cover !important;
        background-position: center center !important;
        background-attachment: fixed !important;
    }}
    .gradio-container::before {{
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(255, 255, 255, 0.15); /* 极淡白遮罩 */
        z-index: 0;
        pointer-events: none;
    }}
    """

# ======================
# ULTIMATE WHITE GLASS CSS
# ======================
MODERN_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');

/* 1. 全局变量覆盖 */
:root, .dark, .gradio-container {{
    --body-background-fill: transparent !important;
    --background-fill-primary: transparent !important;
    --background-fill-secondary: transparent !important;
    --block-background-fill: transparent !important;
    --panel-background-fill: transparent !important;
    --input-background-fill: transparent !important;
    --border-color-primary: rgba(0, 0, 0, 0.1) !important;
    --text-body: #000000 !important;
    --font-sans: 'Inter', system-ui, -apple-system, sans-serif;
    --font-mono: 'JetBrains Mono', 'Menlo', monospace;
}}

body {{
    font-family: var(--font-sans);
    color: #000000;
}}

{bg_css_rule}

#wrap {{
    position: relative;
    z-index: 1;
    max-width: 1280px;
    margin: 0 auto;
    padding: 20px;
}}

/* 2. 玻璃容器样式 (白色磨砂) */
.glass-card, .topbar, .stage-wrapper, .gh-card {{
    background: rgba(255, 255, 255, 0.3) !important; 
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.5) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.05) !important;
}}

/* Topbar 布局 - 标题居中 */
.topbar {{
    padding: 16px 24px;
    display: flex; 
    align-items: center; 
    justify-content: center; /* 内容居中 */
    position: relative;      /* 为绝对定位做参照 */
    margin-bottom: 24px;
}}

.brand {{
    display: flex;
    flex-direction: column;
    align-items: center; 
    text-align: center;
}}

/* 状态 Badge 绝对定位到右侧 */
.status-badge {{
    position: absolute;
    right: 24px;
    background: rgba(255, 255, 255, 0.5);
    border: 1px solid rgba(0, 0, 0, 0.1);
    padding: 6px 12px; border-radius: 99px;
    font-size: 12px; color: #333;
    display: flex; align-items: center; gap: 8px;
    font-weight: 600;
}}
.status-dot {{ width: 8px; height: 8px; border-radius: 50%; background: #10b981; box-shadow: 0 0 5px #10b981; }}

.glass-card {{
    padding: 20px !important;
    margin-bottom: 16px !important;
}}

/* 3. 强制清除内部容器背景 */
.glass-card .block, 
.glass-card .form, 
.glass-card .group, 
.glass-card .wrap,
.glass-card fieldset {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}}

/* 4. 输入框 (黑字 + 黑光标) */
.glass-card input, 
.glass-card textarea, 
.glass-card select, 
.glass-card .dropdown-trigger, 
.glass-card .gr-box {{
    background-color: rgba(255, 255, 255, 0.4) !important;
    border: 1px solid rgba(0, 0, 0, 0.1) !important;
    color: #000000 !important;
    caret-color: #000000 !important;
    border-radius: 10px !important;
    backdrop-filter: blur(4px);
    transition: all 0.2s ease;
    font-weight: 600;
}}

.glass-card input:focus, .glass-card textarea:focus {{
    background-color: rgba(255, 255, 255, 0.6) !important;
    border-color: #000000 !important;
    box-shadow: 0 0 0 2px rgba(0, 0, 0, 0.1) !important;
}}

/* 下拉菜单 */
.secondary-wrap {{ background: transparent !important; border: none !important; }}
ul.options {{
    background-color: rgba(255, 255, 255, 0.95) !important;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(0,0,0,0.1) !important;
}}
li.item {{ color: #000 !important; }}
li.item:hover {{ background: rgba(0,0,0,0.05) !important; }}

/* 5. Radio 按钮去边框 */
#seg_mode {{ background: transparent !important; border: none !important; padding: 0 !important; }}
#seg_mode .wrap {{ border: none !important; gap: 8px; }}
#seg_mode label {{ background: transparent !important; border: none !important; box-shadow: none !important; }}
#seg_mode label span {{ 
    background: rgba(255, 255, 255, 0.4) !important;
    border: 1px solid rgba(0, 0, 0, 0.1) !important;
    color: #333 !important;
    border-radius: 8px !important;
    padding: 8px 16px;
    font-weight: 600;
}}
#seg_mode label.selected span {{ 
    background: rgba(0, 0, 0, 0.8) !important;
    border-color: #000 !important;
    color: #fff !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
}}

/* 6. 按钮 */
button.primary-btn {{
    background: linear-gradient(135deg, #1f2937 0%, #000000 100%) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    color: white !important; font-weight: 700 !important;
    text-transform: uppercase; letter-spacing: 0.05em;
}}
button.primary-btn:hover {{
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
}}
button.secondary-btn {{
    background: rgba(255,255,255,0.4) !important;
    border: 1px solid rgba(0,0,0,0.1) !important;
    color: #000 !important;
}}
button.secondary-btn:hover {{ background: rgba(255,255,255,0.6) !important; }}

/* 7. 右侧 Stage */
.stage-wrapper {{
    height: 620px;
    display: flex; flex-direction: column;
    overflow: hidden;
    position: relative;
    padding: 0 !important;
}}
.live-box {{
    padding: 40px 30px; text-align: center;
    background: transparent;
    border-bottom: 1px solid rgba(0,0,0,0.1);
    position: relative;
}}
.live-text-en {{ 
    color: #000; font-size: 26px; font-weight: 800; line-height: 1.3; 
    text-shadow: 0 0 20px rgba(255,255,255,0.8); 
}}
.live-text-zh {{ 
    color: #1f2937; font-size: 20px; font-weight: 700; margin-top: 10px; 
    text-shadow: 0 0 20px rgba(255,255,255,0.8); 
}}
.live-meta {{ margin-top: 16px; font-size: 11px; color: #555; font-family: var(--font-mono); font-weight: 600; }}

.feed-container {{ flex: 1; overflow-y: auto; padding: 10px 0; }}
.feed-item {{ padding: 10px 24px; border-bottom: 1px solid rgba(0,0,0,0.05); }}
.feed-item:hover {{ background: rgba(255,255,255,0.3); }}
.feed-ts {{ font-family: var(--font-mono); font-size: 10px; color: #666; }}
.feed-en {{ color: #000; font-size: 13px; margin-top: 2px; font-weight: 600; }}
.feed-zh {{ color: #333; font-size: 13px; margin-top: 2px; }}

/* 8. Footer (头像白色框) */
.footerbar {{ margin-top: 20px; display: flex; justify-content: center; }}
.gh-card {{
    display: flex; align-items: center; gap: 12px;
    padding: 8px 16px; border-radius: 99px;
    text-decoration: none !important;
    transition: all 0.2s ease;
    color: #000 !important;
}}
.gh-card:hover {{
    background: rgba(255,255,255,0.5) !important;
    transform: translateY(-2px);
    border-color: #000 !important;
}}
.gh-avatar {{ 
    width: 28px; height: 28px; border-radius: 50%; 
    border: 2px solid #ffffff !important; 
}}
.gh-name {{ font-size: 13px; font-weight: 700; color: #000; }}
.gh-sub {{ font-size: 10px; color: #444; }}

/* 其他 */
.brand .t {{ 
    font-size: 22px; font-weight: 900; color: #000; 
    text-shadow: 0 2px 10px rgba(255,255,255,0.5); 
}}
.brand .s {{ font-size: 12px; color: #444; margin-top: 4px; font-weight: 600; }}

.card-title {{ 
    font-weight: 800; font-size: 12px; color: #000; 
    text-transform: uppercase; letter-spacing: 0.05em; 
    padding-bottom: 8px; border-bottom: 1px solid rgba(0,0,0,0.1); margin-bottom: 16px; 
    opacity: 0.8;
}}

#log textarea {{ 
    background: transparent !important; border: none !important; 
    color: #000 !important; font-size: 11px !important; 
    caret-color: #000 !important; font-family: var(--font-mono) !important;
}}
"""

# ======================
# Utils / global state
# ======================
STOP_EVENTS = {}


def esc(s: str) -> str:
    return html.escape(s or "")


def now_ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def win_startupinfo_hide():
    if os.name != "nt":
        return None
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = subprocess.SW_HIDE
    return si


# ======================
# Twitch HLS
# ======================
def get_hls_url_by_cookie(username: str, auth_token: str, persistent: str):
    cookie_header = f"auth-token={auth_token}; persistent={persistent}"
    cmd = [
        "yt-dlp",
        f"https://www.twitch.tv/{username}",
        "--add-header", f"Cookie:{cookie_header}",
        "-g",
        "--no-warnings",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, startupinfo=win_startupinfo_hide())
    out = (r.stdout or "").strip()
    err = (r.stderr or "").strip()

    if r.returncode != 0 or not out:
        return None, (err[-900:] if err else f"yt-dlp failed, code={r.returncode}")

    for line in out.splitlines():
        line = line.strip()
        if line:
            return line, ""
    return None, "yt-dlp returned empty output"


# ======================
# Render
# ======================
def render_feed(feed_records):
    items = []
    for it in feed_records:
        items.append(f"""
        <div class="feed-item">
          <div class="feed-ts">{esc(it["ts"])}</div>
          <div class="feed-en">{esc(it["en"])}</div>
          <div class="feed-zh">{esc(it["zh"])}</div>
        </div>
        """)
    return "".join(items)


def render_stage(latest_ts, latest_en, latest_zh, feed_records):
    live_html = f"""
    <div class="live-box">
      <div class="live-badge" style="position: absolute; top: 15px; right: 15px; background: #ef4444; color: white; font-size: 10px; padding: 2px 8px; border-radius: 4px; font-weight: 800;">LIVE</div>
      <div class="live-text-en">{esc(latest_en)}</div>
      <div class="live-text-zh">{esc(latest_zh)}</div>
      <div class="live-meta">{esc(latest_ts)} · EN → ZH</div>
    </div>
    """
    feed_html = f"<div class='feed-container'>{render_feed(feed_records)}</div>"
    return f"<div class='stage-wrapper'>{live_html}{feed_html}</div>"


# ======================
# Engine
# ======================
class Engine:
    def __init__(self, fw_model_size: str, trans_model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if self.device == "cuda" else "int8"
        print("Using device:", self.device, "| compute_type:", compute_type)

        self.whisper = WhisperModel(
            fw_model_size,
            device=self.device,
            compute_type=compute_type,
        )

        self.tok = MarianTokenizer.from_pretrained(trans_model_name)
        self.mt = MarianMTModel.from_pretrained(trans_model_name)
        if self.device == "cuda":
            self.mt = self.mt.to("cuda")
        self.mt.eval()

    def fw_transcribe(self, audio_f32: np.ndarray, task: str, force_en: bool, beam_size: int) -> str:
        lang = "en" if (force_en and task == "transcribe") else None
        segments, info = self.whisper.transcribe(
            audio_f32,
            language=lang,
            task=task,
            beam_size=int(beam_size),
            vad_filter=False,
            condition_on_previous_text=False,
            temperature=0.0,
        )
        texts = []
        for seg in segments:
            t = (seg.text or "").strip()
            if t:
                texts.append(t)
        return " ".join(texts).strip()

    def en2zh(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        batch = self.tok([text], return_tensors="pt", truncation=True, padding=True)
        if self.device == "cuda":
            batch = {k: v.to("cuda") for k, v in batch.items()}
        with torch.no_grad():
            gen = self.mt.generate(**batch, max_new_tokens=80)
        out = self.tok.batch_decode(gen, skip_special_tokens=True)[0]
        return (out or "").strip()


ENGINE_CACHE = {}


def get_engine(fw_model_size: str) -> Engine:
    if fw_model_size not in ENGINE_CACHE:
        ENGINE_CACHE[fw_model_size] = Engine(fw_model_size, "Helsinki-NLP/opus-mt-en-zh")
    return ENGINE_CACHE[fw_model_size]


# ======================
# ffmpeg spawn
# ======================
def spawn_ffmpeg(target: str, mode: str, auth_token: str, persistent: str):
    headers = None
    if mode == "Cookie 模式":
        cookie = f"auth-token={auth_token}; persistent={persistent}"
        headers = f"Cookie: {cookie}\r\n"

    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "warning",
        "-fflags", "nobuffer", "-flags", "low_delay",
        "-probesize", "32k", "-analyzeduration", "0",
    ]
    if headers:
        cmd += ["-headers", headers, "-user_agent", "Mozilla/5.0"]

    cmd += [
        "-i", target, "-vn", "-ac", "1", "-ar", "16000",
        "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1",
    ]

    return subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        bufsize=0, startupinfo=win_startupinfo_hide(),
    )


# ======================
# Reader & VAD
# ======================
class PipeReader:
    def __init__(self, proc: subprocess.Popen, stop_ev: threading.Event):
        self.proc = proc
        self.stop_ev = stop_ev
        self.buf = bytearray()
        self.lock = threading.Lock()
        self.eof = False
        self.th = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.th.start()

    def _run(self):
        try:
            while not self.stop_ev.is_set():
                data = self.proc.stdout.read(4096)
                if not data:
                    self.eof = True
                    break
                with self.lock:
                    self.buf.extend(data)
        except Exception:
            self.eof = True

    def pop_all(self) -> bytes:
        with self.lock:
            if not self.buf: return b""
            data = bytes(self.buf)
            self.buf.clear()
            return data


class VadSegmenter:
    def __init__(self, aggressiveness=2, pre_ms=300, start_trigger=3, end_trigger=10, max_sec=12.0):
        self.vad = webrtcvad.Vad(int(aggressiveness))
        self.sr = 16000
        self.frame_ms = 30
        self.frame_bytes = int(self.sr * self.frame_ms / 1000) * 2
        self.pre_frames = max(1, pre_ms // self.frame_ms)
        self.prebuf = deque(maxlen=self.pre_frames)
        self.start_trigger = int(start_trigger)
        self.end_trigger = int(end_trigger)
        self.max_bytes = int(self.sr * float(max_sec)) * 2
        self._in_speech = False
        self._speech_run = 0
        self._sil_run = 0
        self._cur = bytearray()

    def push(self, chunk: bytes):
        out = []
        buf = memoryview(chunk)
        i = 0
        while i + self.frame_bytes <= len(buf):
            frame = bytes(buf[i:i + self.frame_bytes])
            i += self.frame_bytes
            is_speech = self.vad.is_speech(frame, self.sr)
            if not self._in_speech:
                self.prebuf.append(frame)
                if is_speech:
                    self._speech_run += 1
                else:
                    self._speech_run = 0
                if self._speech_run >= self.start_trigger:
                    self._in_speech = True
                    self._sil_run = 0
                    self._cur = bytearray()
                    for pf in self.prebuf: self._cur.extend(pf)
                    self._cur.extend(frame)
            else:
                self._cur.extend(frame)
                if is_speech:
                    self._sil_run = 0
                else:
                    self._sil_run += 1
                if len(self._cur) >= self.max_bytes:
                    out.append(bytes(self._cur))
                    self._reset_state()
                    continue
                if self._sil_run >= self.end_trigger:
                    out.append(bytes(self._cur))
                    self._reset_state()
        tail = bytes(buf[i:]) if i < len(buf) else b""
        return out, tail

    def flush(self):
        if self._in_speech and len(self._cur) > 0:
            seg = bytes(self._cur)
            self._reset_state()
            return seg
        return None

    def _reset_state(self):
        self._in_speech = False
        self._speech_run = 0
        self._sil_run = 0
        self._cur = bytearray()
        self.prebuf.clear()


# ======================
# UI Helpers
# ======================
def precheck(mode, twitch_user, auth_token, persistent, hls_url):
    if mode == "手动 HLS":
        return "OK: HLS URL ready" if (hls_url or "").strip() else "ERR: HLS URL empty"
    if not (twitch_user or "").strip():
        return "ERR: missing twitch username"
    if not (auth_token or "").strip():
        return "ERR: missing auth-token"
    if not (persistent or "").strip():
        return "ERR: missing persistent"
    return "OK"


def clear_all():
    return (render_stage("--:--:--", "Ready.", "准备就绪。", []), "ready")


def stop_stream(sid):
    if sid and sid in STOP_EVENTS:
        STOP_EVENTS[sid].set()
        return "stopping…"
    return "ready"


# ======================
# Stream logic
# ======================
def start_stream(
        mode, twitch_user, auth_token, persistent, hls_url,
        fw_model_size, whisper_task, force_en, beam_size,
        vad_aggr, vad_pre_ms, vad_start, vad_end, vad_max_sec,
        dedup_threshold, max_items, sid
):
    if not sid: sid = uuid.uuid4().hex
    ev = STOP_EVENTS.setdefault(sid, threading.Event())
    ev.clear()
    latest_ts, latest_en, latest_zh = "--:--:--", "Connecting…", "正在连接…"
    feed = []
    log_lines = [f"[{now_ts()}] boot"]
    yield (render_stage(latest_ts, latest_en, latest_zh, feed), sid, "\n".join(log_lines))

    if mode == "手动 HLS":
        target = (hls_url or "").strip()
        if not target:
            log_lines.append(f"[{now_ts()}] error: empty hls url")
            yield (render_stage(latest_ts, "HLS URL is empty.", "HLS 链接为空。", feed), sid, "\n".join(log_lines))
            return
    else:
        user = (twitch_user or "").strip()
        at = (auth_token or "").strip()
        ps = (persistent or "").strip()
        if not user or not at or not ps:
            log_lines.append(f"[{now_ts()}] error: missing cookie creds")
            yield (
            render_stage(latest_ts, "Missing credentials.", "Cookie 信息不完整。", feed), sid, "\n".join(log_lines))
            return
        log_lines.append(f"[{now_ts()}] yt-dlp: fetching hls for user={user}")
        yield (render_stage(latest_ts, "Fetching HLS…", "正在获取 HLS…", feed), sid, "\n".join(log_lines))
        target, yt_err = get_hls_url_by_cookie(user, at, ps)
        if not target:
            log_lines.append(f"[{now_ts()}] error: hls fetch failed")
            yield (render_stage(latest_ts, "HLS fetch failed.", "获取 HLS 失败（未开播/Cookie/地区限制）。", feed), sid,
                   "\n".join(log_lines))
            return

    engine = get_engine(fw_model_size)
    log_lines.append(f"[{now_ts()}] models ready")
    yield (render_stage(latest_ts, "Models ready.", "模型就绪。", feed), sid, "\n".join(log_lines))

    try:
        p = spawn_ffmpeg(target, mode, (auth_token or "").strip(), (persistent or "").strip())
    except Exception as e:
        log_lines.append(f"[{now_ts()}] ffmpeg spawn exception: {e}")
        yield (
        render_stage(latest_ts, f"FFmpeg start failed: {e}", "FFmpeg 启动失败。", feed), sid, "\n".join(log_lines))
        return

    reader = PipeReader(p, ev)
    reader.start()

    asr_in = queue.Queue()
    asr_out = queue.Queue()
    trans_in = queue.Queue()
    trans_out = queue.Queue()

    def asr_worker():
        while not ev.is_set():
            try:
                item_id, pcm_bytes, task, force_en_local, beam_local = asr_in.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                audio = np.frombuffer(pcm_bytes, np.int16).astype(np.float32) / 32768.0
                text = engine.fw_transcribe(audio, task=task, force_en=force_en_local, beam_size=beam_local)
            except Exception:
                text = ""
            asr_out.put((item_id, text))

    def trans_worker():
        while not ev.is_set():
            try:
                item_id, en_text = trans_in.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                zh = engine.en2zh(en_text)
            except Exception:
                zh = ""
            trans_out.put((item_id, zh))

    threading.Thread(target=asr_worker, daemon=True).start()
    threading.Thread(target=trans_worker, daemon=True).start()

    seg = VadSegmenter(int(vad_aggr), int(vad_pre_ms), int(vad_start), int(vad_end), float(vad_max_sec))
    dedup_threshold = float(dedup_threshold)
    max_items = int(max_items)
    latest_id = None
    last_en = ""
    tail = b""

    log_lines.append(f"[{now_ts()}] stream loop start")
    yield (render_stage(latest_ts, "Listening…", "正在监听…", feed), sid, "\n".join(log_lines))

    try:
        while not ev.is_set():
            updated = False
            got = reader.pop_all()
            if got:
                data = tail + got
                segments, tail = seg.push(data)
                for pcm in segments:
                    item_id = uuid.uuid4().hex
                    ts = now_ts()
                    feed.insert(0, {"id": item_id, "ts": ts, "en": "(recognizing…)", "zh": "…"})
                    if len(feed) > max_items: feed[:] = feed[:max_items]
                    latest_id = item_id
                    latest_ts, latest_en, latest_zh = ts, "(recognizing…)", "…"
                    updated = True
                    asr_in.put((item_id, pcm, str(whisper_task), bool(force_en), int(beam_size)))

            if reader.eof:
                time.sleep(0.1)
                yield (render_stage(latest_ts, "Stopped.", "已停止。", feed), sid, "\n".join(log_lines))
                return

            while True:
                try:
                    item_id, text = asr_out.get_nowait()
                except queue.Empty:
                    break
                idx = next((i for i, it in enumerate(feed) if it["id"] == item_id), None)
                if idx is None: continue
                text = (text or "").strip()
                if not text:
                    feed[idx]["en"] = "(no speech)"
                    feed[idx]["zh"] = ""
                    updated = True
                    continue
                if last_en and similar(text, last_en) >= dedup_threshold:
                    feed.pop(idx)
                    updated = True
                    continue
                feed[idx]["en"] = text
                feed[idx]["zh"] = "…"
                last_en = text
                if item_id == latest_id or idx == 0:
                    latest_ts = feed[idx]["ts"]
                    latest_en = text
                    latest_zh = "…"
                    latest_id = item_id
                trans_in.put((item_id, text))
                updated = True

            while True:
                try:
                    item_id, zh = trans_out.get_nowait()
                except queue.Empty:
                    break
                idx = next((i for i, it in enumerate(feed) if it["id"] == item_id), None)
                if idx is None: continue
                feed[idx]["zh"] = zh
                if item_id == latest_id: latest_zh = zh
                updated = True

            if updated:
                yield (render_stage(latest_ts, latest_en, latest_zh, feed), sid, "\n".join(log_lines))
            time.sleep(0.01)

    finally:
        try:
            p.terminate()
            p.wait(timeout=1)
        except Exception:
            pass
        log_lines.append(f"[{now_ts()}] stopped")
        yield (render_stage(latest_ts, latest_en, latest_zh, feed), sid, "\n".join(log_lines))


# ======================
# UI Layout
# ======================
demo = gr.Blocks(title="Twitch AI Subtitles")

with demo:
    gr.HTML(f"<style>{MODERN_CSS}</style>")
    sid = gr.State("")

    with gr.Column(elem_id="wrap"):
        gr.HTML("""
        <div class="topbar">
          <div class="brand">
            <div class="t">Twitch Live Intelligence</div>
            <div class="s">Real-time Translation & Transcription</div>
          </div>
          <div class="status-badge">
            <span class="status-dot"></span>
            <span>SYSTEM READY</span>
          </div>
        </div>
        """)

        with gr.Row(equal_height=False, variant="compact"):
            with gr.Column(scale=1):
                # --- Source Card ---
                with gr.Group(elem_classes="glass-card"):
                    gr.HTML('<div class="card-title"><span>📡 Signal Source</span></div>')
                    mode = gr.Radio(["Cookie 模式", "手动 HLS"], value="Cookie 模式", show_label=False,
                                    elem_id="seg_mode")
                    with gr.Column(elem_classes="padded") as grp_cookie:
                        twitch_user = gr.Textbox(placeholder="Twitch Username", show_label=False, container=False)
                        auth_token = gr.Textbox(placeholder="auth-token", type="password", show_label=False,
                                                container=False)
                        persistent = gr.Textbox(placeholder="persistent", type="password", show_label=False,
                                                container=False)
                    with gr.Column(visible=False, elem_classes="padded") as grp_hls:
                        hls_url = gr.Textbox(placeholder="Paste .m3u8 URL here...", lines=2, show_label=False,
                                             container=False)
                    with gr.Row():
                        btn_pre = gr.Button("Check Connection", variant="secondary", elem_classes="secondary-btn")
                        btn_clear = gr.Button("Clear UI", variant="secondary", elem_classes="secondary-btn")

                # --- Neural Card ---
                with gr.Group(elem_classes="glass-card"):
                    gr.HTML('<div class="card-title"><span>🎛️ Neural Engine</span></div>')
                    fw_model_size = gr.Dropdown(["base", "small", "medium", "large"], value="small", label="Model Size",
                                                container=False)
                    with gr.Row():
                        whisper_task = gr.Dropdown(["transcribe", "translate"], value="transcribe", show_label=False,
                                                   container=False)
                        beam_size = gr.Slider(1, 5, value=3, step=1, label="Beam", container=False)
                    force_en = gr.Checkbox(value=True, label="Force English Input", container=False)
                    with gr.Accordion("VAD Settings", open=False):
                        vad_aggr = gr.Slider(0, 3, value=2, step=1, label="Aggressiveness")
                        vad_pre_ms = gr.Slider(0, 800, value=300, step=50, label="Pre-roll (ms)")
                        with gr.Row():
                            vad_start = gr.Slider(1, 8, value=3, step=1, label="Start Trig")
                            vad_end = gr.Slider(4, 30, value=10, step=1, label="End Trig")
                        vad_max_sec = gr.Slider(3.0, 20.0, value=12.0, step=0.5, label="Max Sec")
                    with gr.Accordion("Advanced", open=False):
                        dedup_threshold = gr.Slider(0.7, 0.99, value=0.85, step=0.01, label="Dedup")
                        max_items = gr.Slider(10, 120, value=50, step=5, label="History Size")
                    with gr.Row():
                        btn_start = gr.Button("INITIALIZE STREAM", variant="primary", elem_classes="primary-btn")
                        btn_stop = gr.Button("TERMINATE", variant="secondary", elem_classes="secondary-btn")

                # --- Log Card ---
                with gr.Group(elem_classes="glass-card"):
                    gr.HTML('<div class="card-title"><span>_SYSTEM_LOGS</span></div>')
                    log = gr.Textbox(value="System initialized.", lines=8, show_label=False, elem_id="log",
                                     container=False)

            with gr.Column(scale=2):
                stage = gr.HTML(value=render_stage("--:--:--", "Ready.", "准备就绪。", []))
                gr.HTML("""
                <div class="footerbar">
                    <a href="https://github.com/yamakawanin" target="_blank" class="gh-card">
                        <img src="https://github.com/yamakawanin.png" class="gh-avatar" alt="avatar">
                        <div class="gh-info">
                            <span class="gh-name">yamakawanin</span>
                            <span class="gh-sub">@github</span>
                        </div>
                    </a>
                </div>
                """)


    def on_mode_change(v):
        return gr.update(visible=(v == "Cookie 模式")), gr.update(visible=(v == "手动 HLS"))


    mode.change(on_mode_change, mode, [grp_cookie, grp_hls])

    btn_pre.click(precheck, [mode, twitch_user, auth_token, persistent, hls_url], log)
    btn_stop.click(stop_stream, sid, log)
    btn_clear.click(lambda: clear_all(), None, [stage, log])
    btn_start.click(
        start_stream,
        inputs=[
            mode, twitch_user, auth_token, persistent, hls_url,
            fw_model_size, whisper_task, force_en, beam_size,
            vad_aggr, vad_pre_ms, vad_start, vad_end, vad_max_sec,
            dedup_threshold, max_items, sid
        ],
        outputs=[stage, sid, log]
    )

    # 强制执行 Light Mode JS (在页面加载时)
    demo.load(None, None, None,
              js="() => { document.body.classList.remove('dark'); document.body.classList.add('light'); }")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
