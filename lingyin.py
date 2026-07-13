#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""聆音 LingYin — 给 AI 当耳朵的单文件 MCP 工具。不起 server。
hear(path): 音频文件 → ASR转写 + librosa分帧物理值 + onset启发式对齐 + 基线(和平时比)
           → LLM 读着这些写一段"听觉感知"散文,给主脑当它听到的声音。
主脑读到的不是原始Hz,是LLM读值写出的、有温度有对齐的散文——像它亲耳听到。

依赖: librosa numpy requests soundfile (pip 一分钟,无torch无GPU)
ASR/LLM 都可换任意 OpenAI 兼容服务。默认 MiMo(小米,国内直连)转写 + DeepSeek(国内直连)判断。

配置走同目录 .env(见 .env.example):
  ASR_PROVIDER=mimo|openai   mimo走chat多模态, openai走标准/audio/transcriptions
  ASR_API_KEY / ASR_BASE_URL / ASR_MODEL / ASR_LANG
  LLM_API_KEY / LLM_BASE_URL / LLM_MODEL
  LINGYIN_BASELINE_FILE (说话人的平时基线)
  (旧变量名 MIMO_API_KEY / DS_API_KEY 等仍兼容)
"""
# 防OpenBLAS多线程抢内存假崩——必须在import numpy/librosa之前设
import os as _os
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json, sys, os, io, re, base64, subprocess, tempfile, statistics
import requests

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 配置:读.env(同目录) ──
_envf = os.path.join(BASE_DIR, ".env")
if os.path.exists(_envf):
    for _line in open(_envf, encoding="utf-8"):
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

# ASR(转写)——支持 MiMo(国内直连,chat多模态) 或 任意OpenAI兼容(/audio/transcriptions)
# 兼容旧变量名 MIMO_API_KEY / MIMO_BASE_URL / MIMO_ASR_MODEL
ASR_PROVIDER = os.environ.get("ASR_PROVIDER", "mimo").lower()  # mimo | openai
ASR_KEY = os.environ.get("ASR_API_KEY", "") or os.environ.get("MIMO_API_KEY", "")
ASR_BASE = os.environ.get("ASR_BASE_URL", "") or os.environ.get("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")
ASR_MODEL = os.environ.get("ASR_MODEL", "") or os.environ.get("MIMO_ASR_MODEL", "mimo-v2.5-asr")
ASR_LANG = os.environ.get("ASR_LANG", "zh")
# MiMo鉴权头是 api-key 非标准(但 Bearer 也通);OpenAI标准用 Bearer
ASR_AUTH_HEADER = os.environ.get("ASR_AUTH_HEADER", "Authorization" if ASR_PROVIDER == "openai" else "api-key")

# LLM(听觉感知)——任意OpenAI兼容。兼容旧 DS_API_KEY / DS_BASE_URL / DS_MODEL
LLM_KEY = os.environ.get("LLM_API_KEY", "") or os.environ.get("DS_API_KEY", "")
LLM_BASE = os.environ.get("LLM_BASE_URL", "") or os.environ.get("DS_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "") or os.environ.get("DS_MODEL", "deepseek-v4-flash")

BASELINE_FILE = os.environ.get("LINGYIN_BASELINE_FILE", os.path.join(BASE_DIR, "lingyin_baseline.json"))

BASELINE_MIN = 8
BASELINE_KEEP = 200

_session = requests.Session()
_session.headers["User-Agent"] = "lingyin/0.2"


def to_wav(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        return path
    td = tempfile.mkdtemp()
    out = os.path.join(td, "a.wav")
    subprocess.run(["ffmpeg", "-y", "-i", path, "-ar", "16000", "-ac", "1", out],
                   capture_output=True, timeout=120)
    if not os.path.exists(out):
        raise RuntimeError("ffmpeg转码失败,检查ffmpeg是否安装")
    return out


def audio_duration(wav_path: str) -> float:
    """秒。librosa.load太重,用ffprobe轻量取时长。"""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", wav_path],
            capture_output=True, text=True, timeout=30)
        return float(r.stdout.strip()) if r.stdout.strip() else 0
    except Exception:
        return 0


def _prepare_upload(path: str, dur: float):
    """长音频压mp3(32kbps mono),短直接原文件。返回(文件路径, is_mp3_compressed)。"""
    if dur > 30:
        td = tempfile.mkdtemp()
        mp3 = os.path.join(td, "a.mp3")
        subprocess.run(["ffmpeg", "-y", "-i", path, "-ar", "16000", "-ac", "1",
                        "-b:a", "32k", mp3], capture_output=True, timeout=120)
        if os.path.exists(mp3):
            return mp3, True
    return path, False


def _mime_of(path: str, is_mp3: bool = False) -> str:
    if is_mp3:
        return "audio/mpeg"
    return {".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
            ".ogg": "audio/ogg", ".webm": "audio/webm"}.get(
        os.path.splitext(path)[1].lower(), "audio/wav")


def transcribe(path: str, dur: float = 0) -> str:
    """ASR转写。两种路径:
    - mimo: chat/completions + input_audio base64多模态(小米MiMo特有,鉴权头api-key)
    - openai: 标准 /audio/transcriptions multipart(OpenAI/Groq/硅基流动/本地vLLM等)
    长音频(>30s)先压mp3再上传。"""
    if not ASR_KEY:
        raise RuntimeError("ASR_API_KEY 未配置")
    upload_path, is_mp3 = _prepare_upload(path, dur)
    if ASR_PROVIDER == "mimo":
        with open(upload_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        mime = _mime_of(path, is_mp3)
        r = _session.post(
            f"{ASR_BASE}/chat/completions",
            headers={ASR_AUTH_HEADER: ASR_KEY, "Content-Type": "application/json"},
            json={"model": ASR_MODEL,
                  "messages": [{"role": "user", "content": [
                      {"type": "input_audio",
                       "input_audio": {"data": f"data:{mime};base64,{data}"}}]}],
                  "asr_options": {"language": ASR_LANG}},
            timeout=120)
        r.raise_for_status()
        return (r.json()["choices"][0]["message"].get("content") or "").strip()
    else:
        # OpenAI标准 /audio/transcriptions multipart
        auth = {ASR_AUTH_HEADER: f"Bearer {ASR_KEY}"} if ASR_AUTH_HEADER.lower() == "authorization" else {ASR_AUTH_HEADER: ASR_KEY}
        with open(upload_path, "rb") as f:
            files = {"file": (os.path.basename(upload_path), f, _mime_of(path, is_mp3))}
            data = {"model": ASR_MODEL}
            if ASR_LANG and ASR_LANG != "auto":
                data["language"] = ASR_LANG
            r = _session.post(f"{ASR_BASE}/audio/transcriptions",
                              headers=auth, files=files, data=data, timeout=120)
        r.raise_for_status()
        j = r.json()
        return (j.get("text") or "").strip()


def analyze(wav_path: str, text: str):
    import numpy as np, librosa
    # soundfile直读比librosa.load(audioread)快3-5倍
    try:
        import soundfile as sf
        y, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
    except Exception:
        y, sr = librosa.load(wav_path, sr=16000, mono=True)
    dur = len(y) / sr
    if dur < 0.3:
        return None

    f0 = librosa.yin(y, fmin=60, fmax=500, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
    voiced = f0[(f0 > 60) & (f0 < 500)]

    hop = int(0.1 * sr)
    frames = []
    for i in range(0, len(y) - hop, hop):
        chunk = y[i:i + hop]
        t = round(i / sr, 2)
        idx = min(int(i / 512), len(f0) - 1)
        f = f0[idx] if idx < len(f0) and not np.isnan(f0[idx]) else 0
        f = float(f) if f else 0.0
        e = float(np.sqrt(np.mean(chunk ** 2)))
        S = np.abs(librosa.stft(chunk, n_fft=min(1024, len(chunk))))
        b = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)[0])) if S.shape[1] > 0 else 0.0
        frames.append({"t": t, "pitch": round(f, 1) if f else 0,
                       "energy": round(e, 4), "brightness": round(b, 0)})

    segs = [s.strip() for s in re.split(r"[，。！？,.!?\n]", text) if s.strip()]
    aligned = []
    if segs:
        if len(onset_times) >= len(segs):
            for i in range(len(segs)):
                s_ = int(i * len(onset_times) / len(segs))
                e_ = int((i + 1) * len(onset_times) / len(segs))
                so = onset_times[s_:e_]
                ts = float(so[0]) if len(so) else (i / len(segs)) * dur
                te = float(onset_times[e_]) if e_ < len(onset_times) else dur
                aligned.append({"text": segs[i], "t_start": round(ts, 1), "t_end": round(te, 1)})
        else:
            for i, s_ in enumerate(segs):
                aligned.append({"text": s_, "t_start": round(i * dur / len(segs), 1),
                                "t_end": round((i + 1) * dur / len(segs), 1)})

    peak = round(float(np.max(voiced)), 1) if len(voiced) else 0
    vmed = round(float(np.median(voiced)), 1) if len(voiced) else 0
    pause_n = sum(1 for f in frames if f["energy"] < 0.01)
    char_n = len(text.replace("，", "").replace("。", ""))
    cps = round(char_n / dur, 1) if dur > 0 else 0
    last_e = round(float(np.mean(rms[-int(0.5 * sr // 512):])), 4) if len(rms) > 10 else 0

    base = load_baseline()
    base_pitch = base.get("pitch_hz", [])
    base_cps = base.get("cps", [])
    base_pause = base.get("pause_pct", [])
    bp_med = statistics.median(base_pitch) if len(base_pitch) >= BASELINE_MIN else 215
    bc_med = statistics.median(base_cps) if len(base_cps) >= BASELINE_MIN else 4.5
    bpa_med = statistics.median(base_pause) if len(base_pause) >= BASELINE_MIN else 15
    has_baseline = len(base_pitch) >= BASELINE_MIN

    summary = {
        "duration_s": round(dur, 1),
        "peak_hz": peak, "median_hz": vmed,
        "end_energy": last_e,
        "pause_frames": pause_n, "total_frames": len(frames),
        "pause_pct": round(pause_n / len(frames) * 100, 0) if frames else 0,
        "cps": cps,
        "base_pitch": bp_med, "base_cps": bc_med, "base_pause": bpa_med,
        "has_baseline": has_baseline,
        "baseline_progress": f"{min(len(base_pitch), BASELINE_MIN)}/{BASELINE_MIN}",
    }
    return {"frames": frames, "aligned": aligned, "summary": summary}


def load_baseline() -> dict:
    try:
        return json.loads(open(BASELINE_FILE, encoding="utf-8").read())
    except Exception:
        return {}

def update_baseline(summary: dict):
    prof = load_baseline()
    for k, val_key in [("pitch_hz", "median_hz"), ("cps", "cps"), ("pause_pct", "pause_pct")]:
        v = summary.get(val_key)
        if v:
            prof.setdefault(k, []).append(v)
            prof[k] = prof[k][-BASELINE_KEEP:]
    try:
        with open(BASELINE_FILE, "w", encoding="utf-8") as f:
            json.dump(prof, f, ensure_ascii=False)
    except Exception:
        pass


def relative_to_baseline(summary: dict) -> str:
    if not summary.get("has_baseline"):
        return ""
    out = []
    def cmp(this, base, name):
        if base <= 0: return
        ratio = this / base
        if ratio >= 1.3: out.append(f"{name}明显偏高")
        elif ratio >= 1.12: out.append(f"{name}偏高")
        elif ratio <= 0.7: out.append(f"{name}明显偏低")
        elif ratio <= 0.88: out.append(f"{name}偏低")
    cmp(summary["median_hz"], summary["base_pitch"], "音高")
    cmp(summary["cps"], summary["base_cps"], "语速")
    cmp(summary["pause_pct"], summary["base_pause"], "停顿")
    return "、".join(out) if out else ""


DS_PROMPT = """你在帮一个AI听一个人说话。给你三样:
1. 转写文字按意群切了段, 每段有粗略时间区间(启发式对齐, 不精确, 只到大致位置)
2. 每100ms一帧的物理声学值(分帧, 已抽样): pitch_hz(音高) energy(能量) brightness(频谱亮度/音色亮暗)
3. 她平时的基线(音高/语速/停顿)——所以你能知道这次是偏高还是偏低, 不是孤立数字

读这些, 写出你听到的——她在什么状态、情绪怎么走、嘴里发生着什么。把数字嵌在叙述里当证据(像"482Hz——她叫名字时故意飙高"), 不要单独罗列数据, 也不要丢数字只抒情。

时间对齐是启发式的, 别把字锁死到精确秒, 用"说话中段""叫名字那截""末段"这种范围措辞, 不要说"某字在X.X秒"。

这段描述会给另一个AI当听觉感知用——他读完要据此回应说话人。所以保留他需要的东西: 情绪怎么变化、哪个时刻发生了什么、说话人的物理状态(嘴离话筒远近/气息/停顿节奏)。不要只抒情到信息丢了。

她平时基线:音高{base_pitch}Hz 语速{base_cps}字/秒 停顿{base_pause}%
意群对齐:{aligned}
这次聚合:峰值{peak}Hz 中位音高{median_hz}Hz 末段气息{end_energy} 停顿{pause_frames}/{total_frames}帧({pause_pct}%) 语速{cps}字/秒 时长{duration_s}s{rel_line}
分帧(已抽样):{frames}

写。用中文。散文,不要分点。抓整段的情绪弧线和几个关键转折,不要逐段流水账。"""


def judge_text(text: str, analysis: dict) -> str:
    if not LLM_KEY:
        raise RuntimeError("LLM_API_KEY 未配置")
    s = analysis["summary"]
    rel = relative_to_baseline(s)
    if rel:
        rel_line = f"\n和平时比:{rel}"
    elif not s["has_baseline"]:
        rel_line = f"\n(基线{s['baseline_progress']}还在攒, 暂无和平时比)"
    else:
        rel_line = ""

    frames = analysis["frames"]
    # 抽样控token——但保留帧密度,弧度在连续帧序列里,砍太狠DS读不出轨迹
    if len(frames) > 150:
        frames = frames[::5]
    elif len(frames) > 80:
        frames = frames[::3]

    # 自适应token:短音频(<30s语音条)限1000写紧凑,长音频2500写充分
    dur = s.get("duration_s", 0)
    max_tokens = 1200 if dur < 30 else 2500

    prompt = DS_PROMPT.format(
        base_pitch=s["base_pitch"], base_cps=s["base_cps"], base_pause=s["base_pause"],
        aligned=json.dumps(analysis["aligned"], ensure_ascii=False),
        peak=s["peak_hz"], median_hz=s["median_hz"], end_energy=s["end_energy"],
        pause_frames=s["pause_frames"], total_frames=s["total_frames"], pause_pct=s["pause_pct"],
        cps=s["cps"], duration_s=s["duration_s"], rel_line=rel_line,
        frames=json.dumps(frames, ensure_ascii=False),
    )

    r = _session.post(
        f"{LLM_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {LLM_KEY}", "Content-Type": "application/json"},
        json={"model": LLM_MODEL, "max_completion_tokens": max_tokens,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=150)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    if not content and msg.get("reasoning_content"):
        content = msg["reasoning_content"].strip()
    return content


def hear(path: str) -> str:
    path = path.strip().strip('"').strip("'")
    if not os.path.exists(path):
        return f"文件不存在: {path}"
    try:
        wav = to_wav(path)
        dur = audio_duration(wav)
        text = transcribe(wav, dur)
        if not text:
            return "转写为空(音频太短或无人声?)"
        analysis = analyze(wav, text)
        if analysis is None:
            return f"音频太短: {path}"
        update_baseline(analysis["summary"])
        try:
            perception = judge_text(text, analysis)
        except Exception as e:
            perception = f"(听觉判断失败: {e})\n转写文字:{text}"

        s = analysis["summary"]
        rel = relative_to_baseline(s)
        rel_str = f"\n[和平时比] {rel}" if rel else ""
        return f"[语音] {text}\n[听觉感知]\n{perception}{rel_str}"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"听失败: {e}"


TOOL_SCHEMA = {
    "name": "hear",
    "description": (
        "听一段语音文件,返回说了什么(转写)+一段听觉感知散文(她怎么说的:音高/能量/气息/停顿的弧线,和平时比的偏差)。"
        "给 AI 当耳朵用:收到一个音频文件路径时调这个,别只读文字。"
        "支持 wav/mp3/m4a/ogg/webm。返回的[听觉感知]是另一个AI读着声学物理值写出的散文,你读它就像听到了说话人的语气和现场。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "音频文件的绝对路径,如 D:\\\\xxx\\\\voice.wav 或 C:/Users/.../a.m4a"}
        },
        "required": ["path"]
    }
}

def read_msg():
    line = sys.stdin.readline()
    if not line:
        return None
    return json.loads(line)

def write_msg(msg):
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()

def main():
    while True:
        try:
            msg = read_msg()
        except Exception:
            continue
        if msg is None:
            break
        method = msg.get("method", "")
        msg_id = msg.get("id")
        params = msg.get("params", {})

        if method == "initialize":
            write_msg({"jsonrpc": "2.0", "id": msg_id,
                       "result": {"protocolVersion": "2024-11-05",
                                  "capabilities": {"tools": {}},
                                  "serverInfo": {"name": "lingyin", "version": "0.2"}}})
        elif method in ("notifications/initialized", "notifications/cancelled"):
            continue
        elif method == "tools/list":
            write_msg({"jsonrpc": "2.0", "id": msg_id, "result": {"tools": [TOOL_SCHEMA]}})
        elif method == "tools/call":
            name = params.get("name")
            args = params.get("arguments", {})
            if name == "hear":
                result = hear(args.get("path", ""))
            else:
                result = f"未知工具: {name}"
            write_msg({"jsonrpc": "2.0", "id": msg_id,
                       "result": {"content": [{"type": "text", "text": result}],
                                  "isError": result.startswith(("文件不存在", "听失败", "转写为空", "未知工具"))}})
        else:
            if msg_id is not None:
                write_msg({"jsonrpc": "2.0", "id": msg_id,
                           "error": {"code": -32601, "message": f"未实现: {method}"}})

if __name__ == "__main__":
    main()
