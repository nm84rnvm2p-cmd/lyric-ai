import os, re, math, json, time, shutil, subprocess, tempfile, urllib.request, hashlib, difflib, unicodedata
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

APP_VERSION = "6.0-PHRASE-WORD-SYNC"
W, H = 1080, 1920          # high-quality vertical default
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
    """Last-resort fallback only. Without ASR timestamps there is no way to know
    the exact singing moment, so this keeps the text usable rather than failing."""
    lines = [normalize_text(x) for x in (text or '').splitlines() if normalize_text(x)]
    raw = [tok for line in lines for tok in re.findall(r"\S+", line)]
    if not raw:
        return []
    weights = np.array([max(1.0, len(re.sub(r"[^\wÀ-ÿ]", "", x)))**0.75 for x in raw], dtype=float)
    weights /= max(weights.sum(), 1.0)
    out=[]; cur=0.0
    for i,(tok,wt) in enumerate(zip(raw,weights)):
        st=cur; en=duration*(cur+float(wt))
        out.append({"word":tok,"start":st,"end":max(st+0.07,en),"prob":0.4,"phrase_id":i})
        cur=en
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
        status.write(f"Transcrevendo com **{model_name}** com timestamps por palavra…")
    segments, info = model.transcribe(
        path,
        language="pt",
        task="transcribe",
        beam_size=5,
        best_of=5,
        patience=1.0,
        temperature=0.0,
        condition_on_previous_text=True,
        # Música cantada pode ser confundida com silêncio pelo VAD.
        # Desativá-lo evita que versos inteiros desapareçam.
        vad_filter=False,
        word_timestamps=True,
        initial_prompt=(
            "Letra de música brasileira em português. Preserve exatamente palavras, "
            "repetições, gírias, contrações e nomes próprios. Não traduza e não resuma."
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


def _norm_token(x: str) -> str:
    x = unicodedata.normalize("NFKD", x or "")
    x = "".join(c for c in x if not unicodedata.combining(c))
    return re.sub(r"[^a-zA-Z0-9]", "", x).lower()

def _token_similarity(a: str, b: str) -> float:
    a=_norm_token(a); b=_norm_token(b)
    if not a or not b: return 0.0
    if a==b: return 1.0
    if a in b or b in a: return 0.88
    return difflib.SequenceMatcher(None,a,b,autojunk=False).ratio()

def align_manual_lyrics(manual_text: str, asr_words: List[dict], duration: float) -> List[dict]:
    """Align the user's official lyric to REAL Whisper word timestamps.

    The previous version used a global SequenceMatcher and then interpolated large
    unmatched runs. That could create absurdly sparse timings. This version performs
    monotonic fuzzy matching token-by-token and keeps each lyric line as a phrase.
    Thus the displayed text is authoritative while the singer's actual timing remains
    authoritative for animation.
    """
    raw_lines=[normalize_text(x) for x in (manual_text or '').splitlines() if normalize_text(x)]
    if not raw_lines:
        return []
    if not asr_words:
        return words_from_manual_lyrics(manual_text,duration)

    out=[]; cursor=0; n=len(asr_words)
    for phrase_id,line in enumerate(raw_lines):
        toks=re.findall(r"\S+",line)
        if not toks: continue
        mapped=[]
        search_start=cursor
        for tok in toks:
            best_idx=None; best_score=0.0
            # Search a generous but monotonic window. This tolerates Whisper spelling
            # differences without ever jumping backwards in the song.
            upper=min(n,search_start+24)
            for j in range(search_start,upper):
                score=_token_similarity(tok,asr_words[j]["word"])
                if score>best_score:
                    best_score=score; best_idx=j
                if score>=0.995:
                    break
            if best_idx is not None and best_score>=0.58:
                mapped.append((tok,best_idx,best_score))
                search_start=best_idx+1
        if mapped:
            first_idx=mapped[0][1]; last_idx=mapped[-1][1]
            phrase_start=float(asr_words[first_idx]["start"])
            phrase_end=float(asr_words[last_idx]["end"])
            # Use the full ASR span for this phrase when possible.
            cursor=last_idx+1
        else:
            # No reliable match: place this phrase between surrounding ASR material.
            phrase_start=float(asr_words[min(cursor,n-1)]["start"]) if cursor<n else duration
            phrase_end=min(duration,phrase_start+max(0.4,0.18*len(toks)))

        # Build individual word timestamps. Known tokens inherit exact ASR times.
        known={m[0]:[] for m in mapped}
        # Duplicate words need ordered assignment, so map by token position.
        pos_map={}
        for tok,idx,score in mapped:
            pos_map.setdefault(tok,[]).append((idx,score))
        mapped_positions=[]
        used=0
        for tok in toks:
            candidates=pos_map.get(tok,[])
            if candidates:
                idx,score=candidates.pop(0); mapped_positions.append((idx,score))
            else:
                mapped_positions.append(None)

        known_pairs=[(i,m[0],m[1]) for i,m in enumerate(mapped_positions) if m is not None]
        # Phrase boundaries are expanded slightly, but word timestamps remain exact.
        for i,tok in enumerate(toks):
            if mapped_positions[i] is not None:
                idx,score=mapped_positions[i]
                st=float(asr_words[idx]["start"]); en=float(asr_words[idx]["end"])
                prob=max(float(asr_words[idx].get("prob",0.0)),float(score)*0.75)
            else:
                # Interpolate only inside this lyric line, never across the entire song.
                prev=[x for x in known_pairs if x[0]<i]
                nxt=[x for x in known_pairs if x[0]>i]
                if prev:
                    pi,pm,_=prev[-1]; base=float(asr_words[pm]["end"])
                else:
                    pi=-1; base=phrase_start
                if nxt:
                    ni,nm,_=nxt[0]; target=float(asr_words[nm]["start"])
                else:
                    ni=len(toks); target=phrase_end
                gap=max(0.12,target-base)
                frac=(i-pi)/max(1,ni-pi)
                st=base+gap*max(0.0,frac-1/max(1,ni-pi))
                en=base+gap*frac
                st=max(phrase_start,st); en=max(st+0.055,min(phrase_end,en))
                prob=0.5
            out.append({"word":tok,"start":max(0.0,st),"end":max(st+0.055,en),"prob":prob,"phrase_id":phrase_id,"phrase_text":line})

    # Enforce strict monotonicity while preserving real timestamps as much as possible.
    out.sort(key=lambda x:(x.get("phrase_id",0),x["start"]))
    prev=0.0
    for w in out:
        w["start"]=max(prev,float(w["start"]))
        w["end"]=max(w["start"]+0.055,float(w["end"]))
        prev=w["start"]
    return out

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

def segment_lyrics(words: List[dict], max_words=18, max_seconds=8.5) -> List[dict]:
    """Create visual phrases. Manual lyric lines stay together; automatic mode
    changes phrase only after a real singing pause/punctuation or a safety limit.
    A phrase is intentionally NOT limited to 7 words: the words accumulate on the
    same frame until the singer finishes the phrase."""
    if not words: return []
    scenes=[]; cur=[]
    manual_mode=any("phrase_id" in w for w in words)
    current_pid=None
    for w in words:
        if not cur:
            cur=[w]; current_pid=w.get("phrase_id") if manual_mode else None; continue
        if manual_mode and w.get("phrase_id")!=current_pid:
            scenes.append(cur); cur=[w]; current_pid=w.get("phrase_id"); continue
        gap=float(w["start"])-float(cur[-1]["end"])
        punctuation=bool(re.search(r"[.!?;:]$",cur[-1]["word"]))
        proposed=cur+[w]
        too_long=len(proposed)>max_words or (proposed[-1]["end"]-proposed[0]["start"]>max_seconds)
        # A gap is the primary automatic phrase boundary. A small gap still means
        # the current phrase is being sung and should keep accumulating words.
        if gap>0.72 or punctuation or too_long:
            scenes.append(cur); cur=[w]
        else:
            cur=proposed
    if cur: scenes.append(cur)

    result=[]
    for i,ws in enumerate(scenes):
        st=float(ws[0]["start"]); en=float(ws[-1]["end"])
        # For manual lines, never let the next line appear before the current one
        # has had a natural release. For automatic phrases use a tiny breathing room.
        pre=0.04 if i==0 else 0.02
        post=0.16
        result.append({"words":ws,"start":max(0.0,st-pre),"end":en+post,
                       "instrumental":False,"phrase_text":" ".join(w["word"] for w in ws)})
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

def _word_style_index(words, idx):
    token = re.sub(r"[^\wÀ-ÿ]", "", words[idx]["word"].lower())
    emotional = {"amor","saudade","beijo","coração","voce","você","eu","mim","nunca","sempre","volta","voltar","embora","ciúmes","perfume","vida","desejo","paixão","sofrer","chora","chorar","quero","meu","minha","tudo","nada"}
    if token in emotional or len(token) >= 8:
        return "accent"
    return "normal"

def _draw_word_animated(base, token, font, x, y, color, accent, progress, W, H, important=False):
    """Kinetic word entrance: opacity + scale + upward motion + soft glow, without FFmpeg filters."""
    p=clamp(progress,0,1)
    e=ease_out(p)
    scale=0.72+0.28*e
    alpha=int(255*e)
    box=font.getbbox(token)
    tw=max(1,box[2]-box[0]); th=max(1,box[3]-box[1])
    pad=28 if important else 18
    layer=Image.new("RGBA",(tw+pad*2,th+pad*2),(0,0,0,0))
    ld=ImageDraw.Draw(layer)
    fill=accent if important else color
    # glow only on impact words; subtle enough to remain editorial.
    if important:
        for sw,a in [(12,24),(7,34),(3,55)]:
            ld.text((pad,pad),token,font=font,fill=fill+(a,),stroke_width=sw,stroke_fill=fill+(a,))
    ld.text((pad,pad),token,font=font,fill=fill+(alpha,),stroke_width=1,stroke_fill=(0,0,0,min(100,alpha)))
    if scale != 1:
        layer=layer.resize((max(1,int(layer.width*scale)),max(1,int(layer.height*scale))),Image.Resampling.LANCZOS)
    yy=int(y + (1-e)*24)
    xx=int(x - (layer.width-tw-pad*2)/2)
    base.alpha_composite(layer,(xx,yy))
    return layer.width

def render_scene_frame(scene, style, registry, W, H, t_local, duration, bg_img):
    """Render one phrase as accumulated kinetic typography.

    Design rule: once a word is sung it remains visible. New words are added at their
    exact timestamps. The entire phrase stays in one composition and disappears only
    when the phrase ends. This is the intended lyric-video behavior.
    """
    img=bg_img.convert("RGBA")
    overlay=Image.new("RGBA",(W,H),(0,0,0,0)); d=ImageDraw.Draw(overlay)
    words=scene.get("words",[])
    if not words:
        p=clamp(t_local/max(duration,0.01),0,1)
        cx=W/2+math.sin(t_local*.65)*W*.10; cy=H*.50+math.cos(t_local*.85)*H*.05
        r=int(W*(.10+.045*math.sin(t_local*1.4)**2))
        for k,(rr,aa) in enumerate([(r*2.1,20),(r*1.55,35),(r,80)]):
            d.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),outline=style.accent+(aa,),width=max(2,int(W*.0025)))
        d.line((W*.12,H*.76,W*.88,H*.76),fill=style.accent+(100,),width=max(2,int(W*.002)))
        return Image.alpha_composite(img,overlay).convert("RGB")

    # Only words whose actual singing start has arrived are visible. All prior words
    # remain, so A -> A B -> A B C -> A B C D.
    spoken=[]
    for idx,w in enumerate(words):
        rs=float(w["start"])-float(scene["start"])
        if t_local>=rs-0.012:
            spoken.append((idx,w,rs))
    if not spoken:
        return Image.alpha_composite(img,overlay).convert("RGB")

    # Scene atmosphere: layered glass/card, accent ring, moving light streaks.
    pulse=0.5+0.5*math.sin(t_local*2.2)
    d.rounded_rectangle((W*.055,H*.12,W*.945,H*.88),radius=int(W*.055),
                        fill=(0,0,0,34),outline=style.accent+(42,),width=max(2,int(W*.002)))
    d.line((W*.09,H*.20,W*(.25+.05*pulse),H*.20),fill=style.accent+(120,),width=max(3,int(W*.004)))
    d.line((W*(.75-.05*pulse),H*.80,W*.91,H*.80),fill=style.accent+(90,),width=max(3,int(W*.004)))

    # Decorative particles/geometry tied to the phrase, deliberately restrained.
    seed=sum((i+1)*ord(c) for i,c in enumerate(scene.get("phrase_text","")))%997
    for k in range(5):
        ang=t_local*(.35+.07*k)+seed*.01+k
        x=W*(.12+.76*((math.sin(ang)+1)/2)); y=H*(.18+.64*((math.cos(ang*1.13)+1)/2))
        rr=int(W*(.004+.002*k)); d.ellipse((x-rr,y-rr,x+rr,y+rr),fill=style.accent+(35+k*8,))

    fpath=font_path(style.font,registry)
    display_path=font_path(style.display_font,registry)
    # Fit the COMPLETE spoken phrase, not only the last seven words.
    spoken_words=[x[1] for x in spoken]
    text=" ".join(w["word"].upper() for w in spoken_words)
    maxw=int(W*.80)
    base_size=int(H*.068 if len(spoken_words)>7 else H*.078)
    if style.layout=="hero" and len(spoken_words)<=4: base_size=int(H*.088)
    font=fit_font(text,maxw,base_size,fpath)

    # Wrap by real pixel width, preserving every spoken word.
    rows=[]; row=[]; roww=0; space_w=text_bbox(d," ",font)[0]
    for item in spoken:
        token=item[1]["word"].upper(); ww=text_bbox(d,token,font)[0]
        if row and roww+space_w+ww>maxw:
            rows.append((row,roww)); row=[item]; roww=ww
        else:
            row.append(item); roww += ww if not roww else space_w+ww
    if row: rows.append((row,roww))
    line_gap=int(font.size*1.08); total_h=line_gap*len(rows)
    y0=H*.50-total_h/2

    newest_idx=spoken[-1][0]
    for ri,(row,roww) in enumerate(rows):
        cursor=(W-roww)/2
        for item in row:
            idx,w,rs=item; token=w["word"].upper()
            ww=text_bbox(d,token,font)[0]
            p=clamp((t_local-rs)/.20,0,1); e=ease_out(p)
            important=_word_style_index(words,idx)=="accent"
            # Every newly spoken word gets a brief accent pulse; important words get glow.
            age=max(0,t_local-rs)
            if idx==newest_idx and age<0.55:
                pulse_amt=1.0+0.055*math.sin(age*18)*(1-age/0.55)
            else: pulse_amt=1.0
            col=style.accent if important or idx==newest_idx else style.fg
            alpha=int(255*e)
            # shadow + soft glow
            if important or idx==newest_idx:
                for sw,a in [(12,20),(7,30),(3,42)]:
                    d.text((cursor+sw*.12,y0+ri*line_gap+sw*.18),token,font=font,
                           fill=col+(a,),stroke_width=sw,stroke_fill=col+(a,))
            # slight rise on entrance, then settle.
            yy=y0+ri*line_gap+(1-e)*28
            d.text((cursor,yy),token,font=font,fill=col+(alpha,),
                   stroke_width=max(1,int(font.size*.012)),stroke_fill=(0,0,0,min(120,alpha)))
            # Micro underline for the newest word.
            if idx==newest_idx:
                uw=max(8,int(ww*.65)); ux=cursor+(ww-uw)/2
                ua=int(140*max(0,1-age/.65))
                d.rounded_rectangle((ux,y0+ri*line_gap+font.size*1.02,ux+uw,
                                     y0+ri*line_gap+font.size*1.02+max(3,int(font.size*.025))),
                                    radius=4,fill=style.accent+(ua,))
            cursor+=ww+space_w

    # Phrase progress bar subtly communicates the current vocal passage.
    frac=clamp(t_local/max(duration,.001),0,1)
    barw=W*.68; d.rounded_rectangle(((W-barw)/2,H*.78,(W+barw)/2,H*.78+7),radius=4,fill=style.muted+(45,))
    d.rounded_rectangle(((W-barw)/2,H*.78,(W-barw)/2+barw*frac,H*.78+7),radius=4,fill=style.accent+(150,))
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
    """High-quality renderer. Frames are piped directly to FFmpeg as raw RGB,
    avoiding the lossy OpenCV mp4v intermediate that caused visible softness."""
    import cv2
    W,H=resolution
    duration=media_duration(audio_path)
    if duration<=0 and scenes:
        duration=max(s["end"] for s in scenes)
    bgcap=None; bg_static=None
    if background_path:
        info=video_info(background_path)
        if info: bgcap=cv2.VideoCapture(background_path)
        else:
            try: bg_static=np.asarray(Image.open(background_path).convert("RGB"))
            except Exception: bg_static=None

    ff=get_ffmpeg()
    silent=Path(out_path).with_suffix('.silent.mp4')
    crf='14' if quality=="Alta qualidade" else '17'
    preset='slow' if quality=="Alta qualidade" else 'medium'
    enc=[ff,'-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24',
         '-s',f'{W}x{H}','-r',str(fps),'-i','-',
         '-an','-c:v','libx264','-preset',preset,'-crf',crf,
         '-pix_fmt','yuv420p','-movflags','+faststart',str(silent)]
    proc=subprocess.Popen(enc,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    scene_i=0; total=max(1,int(math.ceil(duration*fps)))
    try:
        for frame_i in range(total):
            t=frame_i/fps
            while scene_i+1<len(scenes) and t>scenes[scene_i]["end"]: scene_i+=1
            scene=scenes[min(scene_i,len(scenes)-1)] if scenes else {"start":0,"end":duration,"words":[],"instrumental":True}
            bgframe=None
            if bgcap is not None:
                # Sequential-ish seeking is still used for arbitrary input length;
                # OpenCV handles the decode while our typography stays deterministic.
                bgcap.set(cv2.CAP_PROP_POS_MSEC,t*1000); ok,fr=bgcap.read()
                if ok: bgframe=fit_crop_frame(fr,W,H)
            elif bg_static is not None:
                bgframe=fit_crop_frame(bg_static,W,H)
            style=choose_style(scene,FEATURES_GLOBAL,registry,seed=scene_i,global_theme=style_theme)
            bg=make_background((W,H),style,t,background_frame=bgframe)
            local=clamp(t-scene["start"],0,max(.001,scene["end"]-scene["start"]))
            scene_dur=max(.001,scene["end"]-scene["start"])
            final=render_scene_frame(scene,style,registry,W,H,local,scene_dur,bg)
            if scene_i>0 and 0<=t-scene["start"]<0.28:
                prev=scenes[scene_i-1]
                ps=choose_style(prev,FEATURES_GLOBAL,registry,seed=scene_i-1,global_theme=style_theme)
                pd=max(.001,prev["end"]-prev["start"])
                pb=make_background((W,H),ps,max(0,t-.02),background_frame=bgframe)
                outgoing=render_scene_frame(prev,ps,registry,W,H,pd,pd,pb)
                tr=ease_in_out(clamp((t-scene["start"])/.28,0,1))
                final=Image.blend(outgoing,final,tr)
            frame=np.asarray(final.convert('RGB'),dtype=np.uint8)
            proc.stdin.write(frame.tobytes())
            if progress and frame_i%max(1,fps)==0:
                progress.progress(min(.92,frame_i/total*.92),text=f"Renderizando {int(frame_i/total*100)}%")
        proc.stdin.close(); proc.stdin=None
        stderr=proc.stderr.read().decode('utf-8','replace')
        code=proc.wait()
        if code!=0:
            raise RuntimeError('FFmpeg não conseguiu codificar o vídeo em H.264.\n'+stderr[-5000:])
    except BrokenPipeError:
        try: proc.stdin.close()
        except Exception: pass
        err=proc.stderr.read().decode('utf-8','replace')
        proc.wait()
        raise RuntimeError('O FFmpeg encerrou durante a renderização.\n'+err[-5000:])
    finally:
        if bgcap: bgcap.release()

    # Add original audio without re-encoding the already high-quality video.
    cmd=[ff,'-y','-i',str(silent),'-i',audio_path,
         '-map','0:v:0','-map','1:a:0','-c:v','copy',
         '-c:a','aac','-b:a','256k','-shortest','-movflags','+faststart',str(out_path)]
    try:
        run_cmd(cmd,timeout=max(180,int(duration*8)))
    finally:
        silent.unlink(missing_ok=True)
    if progress: progress.progress(1.0,text='Concluído.')

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
    model=st.selectbox("Qualidade da transcrição",["small","medium","large-v3-turbo","large-v3"],index=2,
                       help="medium é o equilíbrio. large-v3-turbo pode ser pesado no Streamlit.")
with col2:
    theme=st.selectbox("Direção visual",["Auto","Black & White","Beige Editorial","Dark Editorial"])

col3,col4=st.columns(2)
with col3:
    res_label=st.selectbox("Resolução",["1080×1920 — recomendada","720×1280 — econômica"],index=0)
with col4:
    quality=st.selectbox("Render",["Equilibrado","Alta qualidade"],index=0)

st.info("Dica: se você tiver a letra oficial, cole-a. A IA agora reconhece o áudio mesmo assim e alinha a letra palavra por palavra aos timestamps reais do cantor.")

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

        try:
            # Always obtain real word timestamps from the audio. If the user supplied
            # lyrics, they replace the displayed spelling but inherit these timestamps.
            asr_words,lang,detdur=transcribe_audio(str(audio_path),model,status)
        except Exception as e:
            if model!="small":
                status.warning("O modelo escolhido não iniciou no ambiente. Tentando automaticamente o modelo small.")
                asr_words,lang,detdur=transcribe_audio(str(audio_path),"small",status)
            else:
                raise
        if lyrics.strip():
            status.write("Alinhando a letra fornecida palavra por palavra ao áudio…")
            words=align_manual_lyrics(lyrics,asr_words,duration)
        else:
            words=asr_words

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
