
import os, re, math, json, time, shutil, subprocess, tempfile, urllib.request, hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# ============================================================
# LYRIC-AI STUDIO — editorial kinetic typography engine
# Python 3.12 / Streamlit
# ============================================================

APP_VERSION = "4.0-EDITORIAL"
W, H = 720, 1280          # reliable default; can be switched to 1080x1920
FPS = 30
CACHE_DIR = Path(".lyric_cache")
FONT_DIR = CACHE_DIR / "fonts"
FONT_DIR.mkdir(parents=True, exist_ok=True)

# Open-license fonts from Google Fonts. The app downloads them on first use.
FONT_SOURCES = {
    "Anton": "https://raw.githubusercontent.com/google/fonts/main/ofl/anton/Anton-Regular.ttf",
    "Bebas Neue": "https://raw.githubusercontent.com/google/fonts/main/ofl/bebasneue/BebasNeue-Regular.ttf",
    "Montserrat": "https://raw.githubusercontent.com/google/fonts/main/ofl/montserrat/Montserrat%5Bwght%5D.ttf",
    "Oswald": "https://raw.githubusercontent.com/google/fonts/main/ofl/oswald/Oswald%5Bwght%5D.ttf",
    "Playfair Display": "https://raw.githubusercontent.com/google/fonts/main/ofl/playfairdisplay/PlayfairDisplay%5Bwght%5D.ttf",
    "Cormorant Garamond": "https://raw.githubusercontent.com/google/fonts/main/ofl/cormorantgaramond/CormorantGaramond%5Bwght%5D.ttf",
    "DM Serif Display": "https://raw.githubusercontent.com/google/fonts/main/ofl/dmserifdisplay/DMSerifDisplay-Regular.ttf",
    "Archivo Black": "https://raw.githubusercontent.com/google/fonts/main/ofl/archivoblack/ArchivoBlack-Regular.ttf",
    "Libre Baskerville": "https://raw.githubusercontent.com/google/fonts/main/ofl/librebaskerville/LibreBaskerville-Regular.ttf",
    "Space Mono": "https://raw.githubusercontent.com/google/fonts/main/ofl/spacemono/SpaceMono-Regular.ttf",
}

# Fallback paths available in Linux/Streamlit containers.
SYSTEM_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
]

# ---------- basic utilities ----------

def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:100]

def clamp(v, a, b):
    return max(a, min(b, v))

def lerp(a, b, t):
    return a + (b-a)*t

def ease_out(t):
    t = clamp(t, 0.0, 1.0)
    return 1 - (1-t)**3

def ease_in_out(t):
    t = clamp(t, 0.0, 1.0)
    return t*t*(3-2*t)

def wrap_words(text: str, max_chars: int) -> List[str]:
    words = text.split()
    lines, cur = [], []
    count = 0
    for word in words:
        extra = len(word) + (1 if cur else 0)
        if cur and count + extra > max_chars:
            lines.append(" ".join(cur))
            cur, count = [word], len(word)
        else:
            cur.append(word)
            count += extra
    if cur:
        lines.append(" ".join(cur))
    return lines

def normalize_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s

def words_from_manual_lyrics(text: str, duration: float) -> List[dict]:
    """Fallback when user provides lyrics. Timings are estimated proportionally."""
    raw = re.findall(r"\S+", normalize_text(text))
    if not raw:
        return []
    # Slightly longer duration for longer words; normalized.
    weights = np.array([max(1.0, len(re.sub(r"[^\wÀ-ÿ]", "", x)))**0.75 for x in raw], dtype=float)
    weights /= weights.sum()
    cur = 0.0
    out = []
    for i, (tok, wt) in enumerate(zip(raw, weights)):
        start = cur
        end = duration * float(cur + wt)
        out.append({"word": tok, "start": start, "end": max(start+0.08, end), "prob": 1.0})
        cur = end
    return out

def get_ffmpeg() -> str:
    # Prefer imageio-ffmpeg bundled executable.
    try:
        import imageio_ffmpeg
        p = imageio_ffmpeg.get_ffmpeg_exe()
        if p and os.path.exists(p):
            return p
    except Exception:
        pass
    p = shutil.which("ffmpeg")
    if p:
        return p
    raise RuntimeError("FFmpeg não encontrado. O requirements.txt inclui imageio-ffmpeg; aguarde a instalação.")

def run_cmd(cmd, timeout=None):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    if p.returncode != 0:
        tail = p.stderr[-5000:]
        raise RuntimeError(tail)
    return p.stdout

def media_duration(path: str) -> float:
    ff = get_ffmpeg()
    try:
        p = subprocess.run(
            [ff, "-hide_banner", "-i", path, "-f", "null", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            timeout=45
        )
        text = (p.stderr or "") + "\n" + (p.stdout or "")
        m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", text)
        if m:
            return int(m.group(1))*3600 + int(m.group(2))*60 + float(m.group(3))
    except Exception:
        pass
    return 0.0

# ---------- fonts ----------

def download_font(name: str) -> Optional[str]:
    target = FONT_DIR / (safe_name(name) + ".ttf")
    if target.exists() and target.stat().st_size > 10000:
        return str(target)
    url = FONT_SOURCES.get(name)
    if not url:
        return None
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LyricAI/4.0"})
        with urllib.request.urlopen(req, timeout=20) as r, open(target, "wb") as f:
            f.write(r.read())
        if target.stat().st_size > 10000:
            return str(target)
    except Exception:
        try:
            target.unlink(missing_ok=True)
        except Exception:
            pass
    return None

@st.cache_resource(show_spinner=False)
def load_font_registry():
    registry = {}
    for name in FONT_SOURCES:
        p = download_font(name)
        if p:
            registry[name] = p
    # Always ensure at least one font.
    if not registry:
        for p in SYSTEM_FONT_CANDIDATES:
            if os.path.exists(p):
                registry["System fallback"] = p
                break
    return registry

def font_path(name: str, registry: dict) -> str:
    if name in registry:
        return registry[name]
    for p in SYSTEM_FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    raise RuntimeError("Nenhuma fonte compatível foi encontrada no ambiente.")

def fit_font(text: str, max_width: int, start_size: int, path: str, max_size=None):
    size = int(max_size or start_size)
    while size >= 20:
        f = ImageFont.truetype(path, size=size)
        box = f.getbbox(text)
        if box[2]-box[0] <= max_width:
            return f
        size -= 2
    return ImageFont.truetype(path, size=20)

# ---------- transcription ----------

@st.cache_resource(show_spinner=False)
def get_whisper(model_name: str):
    from faster_whisper import WhisperModel
    # CPU int8 is the safest Streamlit configuration.
    return WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
        cpu_threads=max(2, min(6, os.cpu_count() or 4)),
        num_workers=1,
    )

def transcribe_audio(path: str, model_name: str, status=None) -> Tuple[List[dict], str, float]:
    model = get_whisper(model_name)
    if status:
        status.write(f"Transcrevendo com **{model_name}**…")
    segments, info = model.transcribe(
        path,
        language="pt",
        task="transcribe",
        beam_size=5,
        best_of=5,
        patience=1.0,
        temperature=[0.0, 0.2, 0.4],
        condition_on_previous_text=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=350, speech_pad_ms=180),
        word_timestamps=True,
        initial_prompt=(
            "Transcrição de música sertaneja brasileira em português. "
            "Preserve palavras, repetições, gírias e contrações. "
            "Não traduza. Não invente palavras."
        ),
    )
    words = []
    for seg in segments:
        if not seg.words:
            continue
        for w in seg.words:
            txt = normalize_text(w.word)
            if not txt:
                continue
            words.append({
                "word": txt,
                "start": float(w.start),
                "end": float(w.end),
                "prob": float(getattr(w, "probability", 0.0) or 0.0),
            })
    return words, getattr(info, "language", "pt"), float(getattr(info, "duration", 0.0) or 0.0)

# ---------- audio / structure analysis ----------

def audio_features(path: str, duration: float) -> dict:
    """Lightweight analysis through ffmpeg -> mono PCM. Used only for visual energy."""
    ff = get_ffmpeg()
    try:
        p = subprocess.run([
            ff, "-v", "error", "-i", path,
            "-ac", "1", "-ar", "8000", "-f", "s16le", "-"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=max(60, int(duration*3)))
        x = np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32)
        if x.size == 0:
            return {"rms": [], "duration": duration}
        hop = 8000 // 4
        rms = []
        for i in range(0, len(x), hop):
            chunk = x[i:i+hop]
            rms.append(float(np.sqrt(np.mean(chunk*chunk)+1e-8)))
        arr = np.array(rms, dtype=float)
        if arr.size:
            lo, hi = np.percentile(arr, [10, 90])
            norm = np.clip((arr-lo)/(hi-lo+1e-6), 0, 1)
        else:
            norm = arr
        return {"rms": norm.tolist(), "duration": duration}
    except Exception:
        return {"rms": [], "duration": duration}

def energy_at(features, t):
    a = features.get("rms", [])
    if not a:
        return 0.5
    idx = int(clamp(t / max(features.get("duration", 1), 1) * len(a), 0, len(a)-1))
    return float(a[idx])

# ---------- lyric cleanup / segmentation ----------

FILLER = {"é", "ah", "oh", "ei", "hum", "hã", "uh", "yeah", "uau"}

def clean_transcription(words: List[dict]) -> List[dict]:
    out = []
    for w in words:
        txt = normalize_text(w["word"])
        txt = txt.replace("  ", " ")
        if not txt:
            continue
        # Remove only obvious repeated hallucination tokens, not normal repetitions.
        if len(txt) > 35:
            continue
        d = dict(w)
        d["word"] = txt
        out.append(d)
    # repair monotonic timestamps
    prev = 0.0
    for w in out:
        w["start"] = max(float(w["start"]), prev)
        w["end"] = max(float(w["end"]), w["start"] + 0.06)
        prev = w["end"]
    return out

def segment_lyrics(words: List[dict], max_words=7, max_seconds=4.2) -> List[dict]:
    if not words:
        return []
    scenes = []
    cur = []
    for w in words:
        if not cur:
            cur = [w]
            continue
        gap = float(w["start"]) - float(cur[-1]["end"])
        proposed = cur + [w]
        duration = proposed[-1]["end"] - proposed[0]["start"]
        # Hard boundaries: long pause, punctuation, or excessive length.
        punctuation_break = bool(re.search(r"[.!?,;:]$", cur[-1]["word"]))
        if gap > 0.75 or len(proposed) > max_words or duration > max_seconds or punctuation_break:
            scenes.append(cur)
            cur = [w]
        else:
            cur = proposed
    if cur:
        scenes.append(cur)

    result = []
    for i, ws in enumerate(scenes):
        start = ws[0]["start"]
        end = ws[-1]["end"]
        # Expand scene slightly to prevent flicker.
        pre = 0.06 if i == 0 else min(0.08, max(0, start - scenes[i-1][-1]["end"]))
        post = 0.08
        result.append({
            "words": ws,
            "start": max(0, start-pre),
            "end": end+post,
        })
    return result

def ensure_coverage(scenes: List[dict], duration: float) -> List[dict]:
    """Never leaves large holes when transcription has valid words."""
    if not scenes:
        return []
    out = []
    for i, s in enumerate(scenes):
        s = dict(s)
        if i == 0:
            s["start"] = max(0.0, min(s["start"], 0.05))
        if i > 0 and s["start"] - out[-1]["end"] > 1.15:
            # Don't invent lyrics; mark an instrumental gap so the renderer can
            # show a designed breathing scene instead of an empty black screen.
            out.append({
                "words": [],
                "start": out[-1]["end"],
                "end": s["start"],
                "instrumental": True,
            })
        s["instrumental"] = False
        out.append(s)
    if out[-1]["end"] < duration-0.15:
        out.append({"words": [], "start": out[-1]["end"], "end": duration, "instrumental": True})
    return out

# ---------- autonomous art direction ----------

@dataclass
class Style:
    bg: Tuple[int,int,int]
    fg: Tuple[int,int,int]
    accent: Tuple[int,int,int]
    muted: Tuple[int,int,int]
    font: str
    display_font: str
    align: str
    layout: str
    outline: bool
    shadow: bool
    grain: float

PALETTES = [
    ((7,7,8),(245,242,234),(188,168,137),(155,153,148)),
    ((18,18,18),(250,250,248),(213,196,161),(145,145,145)),
    ((235,231,222),(18,18,18),(119,91,58),(98,98,98)),
    ((38,31,27),(247,241,229),(209,184,150),(170,164,154)),
]

SERIF = ["Playfair Display", "Cormorant Garamond", "DM Serif Display", "Libre Baskerville"]
SANS = ["Montserrat", "Oswald", "Anton", "Archivo Black"]
MONO = ["Space Mono"]

def choose_style(scene, features, registry, seed=0, global_theme="Auto"):
    text = " ".join(w["word"] for w in scene.get("words", []))
    e = energy_at(features, (scene["start"]+scene["end"])/2)
    n = len(scene.get("words", []))
    # deterministic variation based on scene index and text.
    h = abs(hash(text)) + seed
    p = PALETTES[h % len(PALETTES)]
    # References strongly favor editorial serif + bold sans.
    if e > 0.70 or n <= 2:
        fchoices = [x for x in SANS if x in registry] or list(registry)
        dchoices = [x for x in SERIF if x in registry] or list(registry)
        layout = ["hero", "stack", "split"][h % 3]
    else:
        fchoices = [x for x in SERIF if x in registry] or list(registry)
        dchoices = [x for x in SANS if x in registry] or list(registry)
        layout = ["stack", "center", "editorial"][h % 3]
    if global_theme == "Black & White":
        p = ((8,8,8),(248,248,246),(195,188,176),(145,145,145))
    elif global_theme == "Beige Editorial":
        p = ((229,219,205),(26,24,22),(116,87,56),(102,100,95))
    elif global_theme == "Dark Editorial":
        p = ((10,10,11),(244,242,236),(184,160,126),(142,140,136))
    return Style(p[0],p[1],p[2],p[3],
                 fchoices[h % len(fchoices)], dchoices[(h//3) % len(dchoices)],
                 "center", layout, outline=False, shadow=True, grain=0.018)

# ---------- background generation ----------

def make_background(size, style: Style, t: float, motion=True, background_frame=None):
    w,h = size
    if background_frame is not None:
        img = Image.fromarray(background_frame).convert("RGB").resize((w,h), Image.Resampling.LANCZOS)
        # Editorial treatment: desaturate / contrast depending on style.
        img = ImageEnhance.Color(img).enhance(0.10)
        img = ImageEnhance.Contrast(img).enhance(1.12)
        img = ImageEnhance.Brightness(img).enhance(0.72)
        # Soft vignette.
        arr = np.asarray(img).astype(np.float32)
        yy,xx = np.mgrid[0:h,0:w]
        cx,cy=w/2,h/2
        d=((xx-cx)/(w*.72))**2+((yy-cy)/(h*.72))**2
        vig=np.clip(1.0-0.55*np.maximum(0,d-0.15),0.45,1.0)
        arr*=vig[...,None]
        return Image.fromarray(np.uint8(np.clip(arr,0,255)))
    # Own generated background: layered editorial paper/gradient.
    base = np.zeros((h,w,3),dtype=np.float32)
    c=np.array(style.bg,dtype=np.float32)
    base[:]=c
    yy,xx=np.mgrid[0:h,0:w]
    # two soft radial fields; avoids fragile FFmpeg geq/filter_complex.
    for k, (cx,cy,amp,scale) in enumerate([
        (w*(0.25+0.05*math.sin(t/5)), h*(0.22+0.04*math.cos(t/4)), 0.16, .60),
        (w*(0.78+0.05*math.cos(t/6)), h*(0.70+0.04*math.sin(t/3)), 0.10, .55),
    ]):
        d=((xx-cx)/(w*scale))**2+((yy-cy)/(h*scale))**2
        field=np.exp(-2.4*d)[...,None]
        base=base*(1-field*amp)+np.array(style.accent,dtype=np.float32)*field*amp
    # subtle paper grain
    rng=np.random.default_rng(int(t*1000)%1000003)
    noise=rng.normal(0,255*style.grain,(h,w,1))
    base=np.clip(base+noise,0,255)
    return Image.fromarray(np.uint8(base))

# ---------- typography rendering ----------

def text_bbox(draw, text, font):
    b=draw.textbbox((0,0),text,font=font,stroke_width=0)
    return b[2]-b[0],b[3]-b[1]

def draw_centered(draw, text, y, font, fill, W, stroke=0, stroke_fill=None, shadow=True, alpha=255, x_offset=0):
    tw,th=text_bbox(draw,text,font)
    x=(W-tw)/2+x_offset
    if shadow:
        sh=(0,0,0,80 if alpha>220 else 55)
        draw.text((x+3,y+4),text,font=font,fill=sh,stroke_width=0)
    fill2=tuple(list(fill)[:3])+ (int(alpha),)
    sf=stroke_fill if stroke_fill else fill
    draw.text((x,y),text,font=font,fill=fill2,stroke_width=stroke,stroke_fill=sf)
    return x,y,tw,th

def highlight_words(draw, words, font, max_width, fg, accent, muted, style_kind="normal"):
    # Returns a list of line strings + individual word colors. Simple but robust.
    txt = " ".join(w["word"] for w in words)
    lines = wrap_words(txt, max(10, int(max_width/(max(16,font.size)*0.62))))
    flat = [x for line in lines for x in line.split()]
    important = set()
    # Autonomous keyword selection: content words, longer words, emotional lexicon.
    emotional = {"amor","saudade","beijo","coração","você","eu","mim","nunca","sempre",
                 "volta","voltar","embora","ciúmes","perfume","vida","desejo","paixão",
                 "sofrer","chora","chorar","quero","te","meu","minha","tudo","nada"}
    scored=[]
    for idx,w in enumerate(flat):
        clean=re.sub(r"[^\wÀ-ÿ]","",w.lower())
        score=len(clean)*0.55 + (2.0 if clean in emotional else 0) + (1.5 if len(clean)>=8 else 0)
        scored.append((score,idx))
    for _,idx in sorted(scored,reverse=True)[:max(1,min(2,len(flat)))]:
        important.add(idx)
    return lines, important

def render_scene_frame(scene, style, registry, W, H, t_local, duration, bg_img):
    # scene-relative 0..duration
    img = bg_img.convert("RGBA")
    overlay = Image.new("RGBA",(W,H),(0,0,0,0))
    d=ImageDraw.Draw(overlay)
    words=scene.get("words",[])
    text=" ".join(w["word"] for w in words)
    if not text:
        # instrumental: elegant micro motion / line.
        prog=clamp(t_local/max(duration,0.01),0,1)
        y=int(H*0.49+math.sin(t_local*1.1)*8)
        d.line((W*.22,y,W*.78,y),fill=style.accent+(150,),width=2)
        return Image.alpha_composite(img,overlay).convert("RGB")

    # Entrance / exit
    fade=0.18
    a=255
    if t_local<fade: a=int(255*ease_out(t_local/fade))
    if duration-t_local<fade: a=int(255*ease_out((duration-t_local)/fade))
    pulse=1.0+0.035*math.sin(math.pi*clamp(t_local/duration,0,1))
    energy=0.55

    # Layout selected from reference language.
    if style.layout=="hero" or len(words)<=2:
        fpath=font_path(style.display_font,registry)
        maxw=int(W*.90)
        fs=int(H*.105 if len(text)<=8 else H*.075)
        font=fit_font(text.upper(),maxw,fs,fpath)
        tw,th=text_bbox(d,text.upper(),font)
        scale=1.0
        if t_local<0.28:
            scale=lerp(1.10,1.0,ease_out(t_local/.28))
        # Render on temp for scale/alpha.
        tmp=Image.new("RGBA",(max(10,tw+80),max(10,th+80)),(0,0,0,0))
        td=ImageDraw.Draw(tmp)
        td.text((40,20),text.upper(),font=font,fill=style.fg+(a,),anchor=None)
        if scale!=1:
            tmp=tmp.resize((int(tmp.width*scale),int(tmp.height*scale)),Image.Resampling.LANCZOS)
        x=(W-tmp.width)//2
        y=int(H*.42-tmp.height/2)
        overlay.alpha_composite(tmp,(x,y))
        # small accent rule
        d=ImageDraw.Draw(overlay)
        linew=int(W*.18)
        d.line(((W-linew)/2,H*.64,(W+linew)/2,H*.64),fill=style.accent+(min(a,190),),width=2)
    else:
        fpath=font_path(style.font,registry)
        fs=int(H*.062 if len(words)<=5 else H*.050)
        font=fit_font(text.upper(),int(W*.86),fs,fpath)
        lines, important=highlight_words(d,words,font,int(W*.86),style.fg,style.accent,style.muted)
        line_gap=int(font.size*.82)
        total_h=line_gap*len(lines)
        y=int(H*.5-total_h/2)
        idx=0
        for li,line in enumerate(lines):
            tw,th=text_bbox(d,line.upper(),font)
            x=(W-tw)/2
            # Slight editorial vertical movement, not a cheesy slide.
            yy=y+li*line_gap + int(math.sin(t_local*1.2+li)*3)
            cursor=x
            for token in line.split():
                token_u=token.upper()
                clean=re.sub(r"[^\wÀ-ÿ]","",token.lower())
                # Match approximate word index.
                color=style.fg
                if idx in important:
                    color=style.accent
                    # impact word gets slightly larger by drawing a subtle halo.
                # shadow only when background is video.
                if style.shadow:
                    d.text((cursor+2,yy+3),token_u,font=font,fill=(0,0,0,min(80,a)))
                d.text((cursor,yy),token_u,font=font,fill=color+(a,))
                ww=text_bbox(d,token_u,font)[0]
                space=text_bbox(d," ",font)[0]
                cursor += ww+space
                idx+=1
    return Image.alpha_composite(img,overlay).convert("RGB")

# ---------- background video reader ----------

def video_info(path):
    import cv2
    cap=cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps=cap.get(cv2.CAP_PROP_FPS) or 30
    frames=cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    dur=frames/fps if fps else 0
    cap.release()
    return {"fps":fps,"frames":frames,"w":w,"h":h,"duration":dur}

def fit_crop_frame(frame, W,H):
    import cv2
    if frame is None:
        return None
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    h,w=frame.shape[:2]
    target=W/H
    ratio=w/h
    if ratio>target:
        nw=int(h*target)
        x=(w-nw)//2
        frame=frame[:,x:x+nw]
    else:
        nh=int(w/target)
        y=(h-nh)//2
        frame=frame[y:y+nh,:]
    frame=cv2.resize(frame,(W,H),interpolation=cv2.INTER_AREA)
    return frame

# ---------- render ----------

def render_video(audio_path, background_path, scenes, registry, style_theme,
                 out_path, resolution=(720,1280), fps=30, quality="Equilibrado",
                 progress=None):
    import cv2
    W,H=resolution
    duration=media_duration(audio_path)
    # If ffmpeg duration parsing is unavailable, use last lyric timestamp.
    if duration<=0 and scenes:
        duration=max(s["end"] for s in scenes)
    bgcap=None
    bginfo=None
    bg_static=None
    if background_path:
        bginfo=video_info(background_path)
        if bginfo:
            bgcap=cv2.VideoCapture(background_path)
        else:
            try:
                bg_static=np.asarray(Image.open(background_path).convert("RGB"))
            except Exception:
                bg_static=None

    # Codec selection. mp4v is broadly available in OpenCV containers.
    temp_video=Path(out_path).with_suffix(".silent.mp4")
    fourcc=cv2.VideoWriter_fourcc(*"mp4v")
    writer=cv2.VideoWriter(str(temp_video),fourcc,fps,(W,H))
    if not writer.isOpened():
        raise RuntimeError("Não foi possível abrir o renderizador de vídeo no ambiente.")

    scene_i=0
    total=max(1,int(math.ceil(duration*fps)))
    # Quality can change the final size without changing design.
    for frame_i in range(total):
        t=frame_i/fps
        while scene_i+1<len(scenes) and t>scenes[scene_i]["end"]:
            scene_i+=1
        scene=scenes[min(scene_i,len(scenes)-1)] if scenes else {"start":0,"end":duration,"words":[],"instrumental":True}
        # Background frame
        bgframe=None
        if bgcap is not None:
            bgcap.set(cv2.CAP_PROP_POS_MSEC,t*1000)
            ok,fr=bgcap.read()
            if ok:
                bgframe=fit_crop_frame(fr,W,H)
        elif bg_static is not None:
            bgframe=fit_crop_frame(bg_static,W,H)
        style=choose_style(scene,FEATURES_GLOBAL,registry,seed=scene_i,global_theme=style_theme)
        bg=make_background((W,H),style,t,background_frame=bgframe)
        local=clamp(t-scene["start"],0,max(0.001,scene["end"]-scene["start"]))
        final=render_scene_frame(scene,style,registry,W,H,local,scene["end"]-scene["start"],bg)
        writer.write(cv2.cvtColor(np.asarray(final),cv2.COLOR_RGB2BGR))
        if progress and frame_i%max(1,fps)==0:
            progress.progress(min(0.92,frame_i/total*0.92), text=f"Renderizando {int(frame_i/total*100)}%")
    writer.release()
    if bgcap: bgcap.release()

    # Mux original audio. Re-encode video to H.264 for mobile compatibility.
    ff=get_ffmpeg()
    cmd=[
        ff,"-y","-i",str(temp_video),"-i",audio_path,
        "-map","0:v:0","-map","1:a:0",
        "-c:v","libx264","-preset","veryfast","-crf","19",
        "-pix_fmt","yuv420p","-movflags","+faststart",
        "-c:a","aac","-b:a","192k","-shortest",str(out_path)
    ]
    try:
        run_cmd(cmd,timeout=max(180,int(duration*8)))
    except Exception:
        # Some builds do not expose libx264; fallback to MPEG-4.
        cmd2=[
            ff,"-y","-i",str(temp_video),"-i",audio_path,
            "-map","0:v:0","-map","1:a:0",
            "-c:v","mpeg4","-q:v","4","-c:a","aac","-b:a","192k",
            "-shortest",str(out_path)
        ]
        run_cmd(cmd2,timeout=max(180,int(duration*8)))
    try: temp_video.unlink(missing_ok=True)
    except Exception: pass
    if progress: progress.progress(1.0,text="Concluído.")

# ---------- Streamlit UI ----------

st.set_page_config(page_title="Lyric AI Studio",page_icon="🎵",layout="centered")
st.markdown("""
<style>
.block-container{max-width:1050px;padding-top:1.2rem}
h1{letter-spacing:-.04em}
.small{opacity:.72;font-size:.88rem}
div[data-testid="stFileUploader"]{border-radius:14px}
</style>
""",unsafe_allow_html=True)

st.title("🎵 Lyric AI Studio")
st.caption(f"Editorial Kinetic Engine · {APP_VERSION}")

with st.expander("O que esta versão faz",expanded=False):
    st.write(
        "Transcreve a música com timestamps de palavras, cria direção visual automática, "
        "alterna tipografia editorial, palavras de impacto, escala, composição e fundos próprios, "
        "e entrega MP4 vertical. O motor não reutiliza os vídeos de referência."
    )

audio_file=st.file_uploader("1. Envie a música ou vídeo com a música",type=["mp3","wav","m4a","mp4","mov","webm"])
bg_file=st.file_uploader("2. Fundo opcional (vídeo/imagem). Deixe vazio para a IA criar o fundo.",type=["mp4","mov","webm","jpg","jpeg","png"])
lyrics=st.text_area("3. Letra (opcional, mas RECOMENDADA para precisão máxima)",height=120,
                    placeholder="Cole a letra aqui. Se deixar vazio, a IA tentará transcrever automaticamente.")

col1,col2=st.columns(2)
with col1:
    model=st.selectbox("Qualidade da transcrição",["small","medium","large-v3-turbo"],index=1,
                       help="medium é o equilíbrio. large-v3-turbo pode ser pesado no Streamlit.")
with col2:
    theme=st.selectbox("Direção visual",["Auto","Black & White","Beige Editorial","Dark Editorial"])

col3,col4=st.columns(2)
with col3:
    res_label=st.selectbox("Resolução",["720×1280 — recomendado","1080×1920 — alta"],index=0)
with col4:
    quality=st.selectbox("Render",["Equilibrado","Alta qualidade"],index=0)

st.info("Dica: para máxima fidelidade da letra, cole a letra oficial. O áudio continua sendo usado para sincronizar o vídeo.")

registry=load_font_registry()
st.caption(f"Fontes disponíveis: {len(registry)}/10. O sistema usa fallback automaticamente se o download das fontes não estiver disponível.")

if registry:
    with st.expander("Fontes detectadas"):
        st.write(", ".join(registry.keys()))

if st.button("🚀 CRIAR LYRIC VIDEO",type="primary",use_container_width=True):
    if not audio_file:
        st.error("Envie a música/áudio primeiro.")
        st.stop()

    tmpdir=Path(tempfile.mkdtemp(prefix="lyricai_"))
    try:
        audio_path=tmpdir/safe_name(audio_file.name)
        audio_path.write_bytes(audio_file.getbuffer())
        bg_path=None
        if bg_file:
            bg_path=tmpdir/safe_name(bg_file.name)
            bg_path.write_bytes(bg_file.getbuffer())

        status=st.empty()
        bar=st.progress(0.0)

        status.write("Preparando áudio…")
        duration=media_duration(str(audio_path))
        if duration<=0: duration=60.0

        if lyrics.strip():
            words=words_from_manual_lyrics(lyrics,duration)
            lang="pt"
            status.write("Usando a letra fornecida e distribuindo a sincronização automaticamente.")
        else:
            try:
                words,lang,detdur=transcribe_audio(str(audio_path),model,status)
            except Exception as e:
                # automatic fallback to small if medium/large cannot initialize
                if model!="small":
                    status.warning("O modelo escolhido não iniciou no ambiente. Tentando automaticamente o modelo small.")
                    words,lang,detdur=transcribe_audio(str(audio_path),"small",status)
                else:
                    raise

        words=clean_transcription(words)
        if not words:
            raise RuntimeError("Nenhuma palavra foi reconhecida. Cole a letra no campo opcional e tente novamente.")

        bar.progress(.20,text="Organizando letra…")
        scenes=segment_lyrics(words)
        scenes=ensure_coverage(scenes,duration)

        global FEATURES_GLOBAL
        FEATURES_GLOBAL=audio_features(str(audio_path),duration)
        bar.progress(.30,text="Analisando ritmo e intensidade…")

        resolution=(1080,1920) if res_label.startswith("1080") else (720,1280)
        out=tmpdir/"lyric_video_final.mp4"
        bar.progress(.32,text="Criando direção visual…")

        render_video(
            str(audio_path),str(bg_path) if bg_path else None,
            scenes,registry,theme,str(out),resolution=resolution,fps=FPS,
            quality=quality,progress=bar
        )

        status.success("Vídeo criado.")
        st.video(str(out))
        st.download_button(
            "⬇️ Baixar MP4",
            data=out.read_bytes(),
            file_name="lyric_ai_final.mp4",
            mime="video/mp4",
            use_container_width=True,
        )

        # Diagnostic report — useful when a transcription is wrong.
        with st.expander("Diagnóstico da IA"):
            avg_prob=float(np.mean([w["prob"] for w in words])) if words else 0
            st.write(f"Palavras detectadas: **{len(words)}**")
            st.write(f"Confiança média reportada pelo modelo: **{avg_prob:.2f}**")
            st.write(f"Idioma detectado: **{lang}**")
            st.write(f"Cenas criadas: **{len([s for s in scenes if not s.get('instrumental')])}**")
            st.write("Transcrição:")
            st.code(" ".join(w["word"] for w in words))

    except Exception as e:
        st.error("A geração falhou.")
        st.code(str(e))
        st.info(
            "Se o erro mencionar memória/modelo, use 'small' ou cole a letra manualmente. "
            "Se mencionar FFmpeg, o código tenta automaticamente um codec alternativo."
        )
    finally:
        # Do not delete immediately: Streamlit may still need the generated file for playback/download.
        # OS cleanup is left to the temporary directory lifecycle.
        pass
