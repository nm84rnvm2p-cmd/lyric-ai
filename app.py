import os, re, math, shutil, subprocess, tempfile, urllib.request, difflib, unicodedata
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

APP_VERSION = "11.0-STABLE-LATIN-WORD-SYNC"
FPS = 30
BLACK = (5, 5, 7)
WHITE = (248, 248, 246)
ROYAL = (45, 92, 255)
CACHE = Path(".lyric_cache")
FONT_DIR = CACHE / "fonts"
FONT_DIR.mkdir(parents=True, exist_ok=True)

FONTS = {
    # Static TTF files are used instead of variable fonts. This is important on
    # Streamlit Cloud/Pillow because static files have much more predictable
    # Portuguese/Latin glyph rendering.
    "Anton": "https://raw.githubusercontent.com/google/fonts/main/ofl/anton/Anton-Regular.ttf",
    "Bebas Neue": "https://raw.githubusercontent.com/google/fonts/main/ofl/bebasneue/BebasNeue-Regular.ttf",
    "Montserrat": "https://raw.githubusercontent.com/google/fonts/main/ofl/montserrat/static/Montserrat-Bold.ttf",
    "Oswald": "https://raw.githubusercontent.com/google/fonts/main/ofl/oswald/static/Oswald-Bold.ttf",
    "Archivo Black": "https://raw.githubusercontent.com/google/fonts/main/ofl/archivoblack/ArchivoBlack-Regular.ttf",
    "DM Serif Display": "https://raw.githubusercontent.com/google/fonts/main/ofl/dmserifdisplay/DMSerifDisplay-Regular.ttf",
    "Playfair Display": "https://raw.githubusercontent.com/google/fonts/main/ofl/playfairdisplay/static/PlayfairDisplay-Bold.ttf",
    "Libre Baskerville": "https://raw.githubusercontent.com/google/fonts/main/ofl/librebaskerville/static/LibreBaskerville-Bold.ttf",
    "Space Mono": "https://raw.githubusercontent.com/google/fonts/main/ofl/spacemono/SpaceMono-Regular.ttf",
}
MAIN_FONTS = ["Anton", "Bebas Neue", "Archivo Black", "Montserrat", "Oswald"]
ALT_FONTS = ["Montserrat", "Oswald", "DM Serif Display", "Playfair Display", "Libre Baskerville", "Bebas Neue"]
SYSTEM_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
]


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, float(x)))


def ease(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def norm(s):
    """Normalize lyric text without removing Portuguese accents.
    Also removes invisible/control characters that can appear in ASR output.
    """
    s = unicodedata.normalize("NFC", s or "")
    s = s.replace("\u200b", "").replace("\ufeff", "")
    s = s.replace("â", '"').replace("â", '"').replace("â", "'").replace("â", "'")
    s = s.replace("â", "-").replace("â", "-").replace("â¦", "...")
    s = "".join(ch for ch in s if ch == "\n" or unicodedata.category(ch)[0] != "C")
    return re.sub(r"\s+", " ", s).strip()


def safe(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:100]


def token_norm(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-zA-Z0-9]", "", s).lower()


def similarity(a, b):
    a, b = token_norm(a), token_norm(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.90
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def ffmpeg():
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
    raise RuntimeError("FFmpeg nÃ£o encontrado. Instale imageio-ffmpeg no requirements.txt.")


def run(cmd, timeout=None):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-5000:])
    return p.stdout


def media_duration(path):
    p = subprocess.run([ffmpeg(), "-hide_banner", "-i", path, "-f", "null", "-"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=90)
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", p.stderr)
    if not m:
        return 0.0
    return int(m.group(1))*3600 + int(m.group(2))*60 + float(m.group(3))


def detect_audio_end(path, dur):
    # Conservative: only shorten when silence is genuinely at the very end.
    try:
        p = subprocess.run([
            ffmpeg(), "-hide_banner", "-i", path,
            "-af", "silencedetect=noise=-40dB:d=0.65", "-f", "null", "-"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=max(60, int(dur*2)))
        starts = [float(x) for x in re.findall(r"silence_start:\s*([\d.]+)", p.stderr)]
        ends = [float(x) for x in re.findall(r"silence_end:\s*([\d.]+)", p.stderr)]
        if starts and starts[-1] >= dur * 0.82:
            if ends and ends[-1] >= starts[-1]:
                return min(dur, ends[-1] + 0.12)
            return min(dur, starts[-1] + 0.12)
    except Exception:
        pass
    return dur


@st.cache_resource(show_spinner=False)
def font_registry():
    out = {}
    for name, url in FONTS.items():
        target = FONT_DIR / (safe(name) + ".ttf")
        if not target.exists() or target.stat().st_size < 10000:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "LyricAIStudio"})
                with urllib.request.urlopen(req, timeout=20) as r:
                    target.write_bytes(r.read())
            except Exception:
                target.unlink(missing_ok=True)
        if target.exists() and target.stat().st_size >= 10000:
            out[name] = str(target)
    # Always register a known-good Unicode fallback for Portuguese accents.
    for p in SYSTEM_FONTS:
        if os.path.exists(p):
            out["Unicode Safe"] = p
            break
    if not out:
        raise RuntimeError("Nenhuma fonte compatÃ­vel foi encontrada.")
    return out


def has_suspicious_glyphs(text):
    # Digits embedded inside alphabetic words are a common visible symptom of
    # bad ASR/font fallback (e.g. "A1A"). We do not rewrite the word here;
    # we simply force a known-good Unicode font for rendering.
    return any(ch.isalpha() and any(c.isdigit() for c in text) for ch in text)


def safe_font_name(text, preferred, reg):
    """Use the selected editorial font when safe, but guarantee Portuguese
    accented glyphs render with a Unicode-complete fallback when necessary."""
    preferred_path = reg.get(preferred)
    if preferred_path and not has_suspicious_glyphs(text):
        # DejaVu Sans Bold is retained in the registry as the hard Unicode fallback.
        return preferred
    return "Unicode Safe"


def get_font(name, reg, size):
    path = reg.get(name) or reg.get("Unicode Safe") or next(iter(reg.values()))
    return ImageFont.truetype(path, max(16, int(size)))


def bbox(text, font):
    b = font.getbbox(text)
    return b[0], b[1], b[2]-b[0], b[3]-b[1]


def fit_font(text, name, reg, size, max_width, min_size=30):
    size = int(size)
    while size >= min_size:
        f = get_font(name, reg, size)
        if bbox(text, f)[2] <= max_width:
            return f
        size -= 2
    return get_font(name, reg, min_size)


def transcribe(path, model_name, status):
    from faster_whisper import WhisperModel
    if status:
        status.write(f"ðï¸ Reconhecendo palavra por palavra com **{model_name}**â¦")
    model = WhisperModel(
        model_name, device="cpu", compute_type="int8",
        cpu_threads=max(2, min(8, os.cpu_count() or 4)), num_workers=1
    )
    segments, info = model.transcribe(
        path, language="pt", task="transcribe", word_timestamps=True,
        beam_size=5, best_of=5, patience=1.0, temperature=0.0,
        condition_on_previous_text=False, vad_filter=False,
        initial_prompt="Letra de mÃºsica brasileira em portuguÃªs. ReconheÃ§a todas as palavras, repetiÃ§Ãµes, gÃ­rias e contraÃ§Ãµes. NÃ£o resuma nem traduza."
    )
    words = []
    for seg in segments:
        if not seg.words:
            continue
        for w in seg.words:
            text = norm(w.word)
            if text and len(text) <= 40:
                # Do not display obvious control/gibberish artifacts from ASR.
                # Valid lyric text supplied by the user is never passed through this filter.
                letters = sum(ch.isalpha() for ch in text)
                digits = sum(ch.isdigit() for ch in text)
                if letters >= 1 and digits >= 1 and letters + digits >= 3:
                    continue
                words.append({
                    "word": text,
                    "start": float(w.start),
                    "end": float(w.end),
                    "prob": float(getattr(w, "probability", 0) or 0)
                })
    words.sort(key=lambda x: x["start"])
    return words, getattr(info, "language", "pt")


def parse_timed(text):
    result = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        m = re.match(r"^(\d{1,2}):(\d{2})(?:[.,](\d{1,3}))?\s*[-ââ]\s*(\d{1,2}):(\d{2})(?:[.,](\d{1,3}))?\s*\|\s*(.+)$", raw)
        if m:
            def tm(a,b,c): return int(a)*60 + int(b) + float("0."+(c or "0"))
            result.append({"start": tm(m.group(1),m.group(2),m.group(3)), "end": tm(m.group(4),m.group(5),m.group(6)), "text": norm(m.group(7))})
            continue
        m = re.match(r"^(\d{1,2}):(\d{2})(?:[.,](\d{1,3}))?\s*\|\s*(.+)$", raw)
        if m:
            result.append({"start": int(m.group(1))*60+int(m.group(2))+float("0."+(m.group(3) or "0")), "text": norm(m.group(4))})
            continue
        m = re.match(r"^(\d+(?:[.,]\d+)?)\s*\|\s*(.+)$", raw)
        if m:
            result.append({"start": float(m.group(1).replace(",", ".")), "text": norm(m.group(2))})
    return sorted(result, key=lambda x: x["start"])


def plain_lines(text):
    return [norm(x) for x in text.splitlines() if norm(x)]


def align_phrase(text, start, end, asr, cursor_hint=0):
    tokens = re.findall(r"\S+", text)
    candidates = [(i,w) for i,w in enumerate(asr) if w["end"] >= start-0.55 and w["start"] <= end+0.55]
    mapped = []
    cursor = 0
    for tok in tokens:
        best = None; best_score = 0
        for j in range(cursor, min(len(candidates), cursor+40)):
            _, w = candidates[j]
            score = similarity(tok, w["word"])
            if start <= w["start"] <= end:
                score += 0.06
            if score > best_score:
                best_score, best = score, (j,w)
            if score >= 1.0:
                break
        if best and best_score >= 0.40:
            j,w = best
            mapped.append((tok,w,best_score))
            cursor = j+1
        else:
            mapped.append((tok,None,0.0))
    known = [i for i,x in enumerate(mapped) if x[1] is not None]
    result=[]
    for i,(tok,w,score) in enumerate(mapped):
        if w is not None:
            st=max(start,float(w["start"])); en=min(end,max(st+0.055,float(w["end"])))
            result.append({"word":tok,"start":st,"end":max(st+0.055,en),"prob":max(float(w.get("prob",0)),score*0.75)})
            continue
        prev=max([k for k in known if k<i],default=-1)
        nxt=min([k for k in known if k>i],default=len(tokens))
        a=result[prev]["end"] if prev>=0 else start
        b=mapped[nxt][1]["start"] if nxt<len(tokens) else end
        if b < a: b=a+0.08
        total=max(1,nxt-prev)
        st=a+(b-a)*(i-prev)/total
        en=a+(b-a)*(i-prev+1)/total
        result.append({"word":tok,"start":clamp(st,start,end),"end":clamp(max(st+0.055,en),start+0.055,end),"prob":0.45})
    return result


def align_plain(text, asr, audio_end):
    lines=plain_lines(text)
    scenes=[]; cursor=0
    for pid,line in enumerate(lines):
        toks=re.findall(r"\S+",line)
        matches=[]
        for tok in toks:
            best=None; score_best=0
            for j in range(cursor,min(len(asr),cursor+50)):
                s=similarity(tok,asr[j]["word"])
                if s>score_best:
                    score_best=s; best=j
                if s>=0.99: break
            if best is not None and score_best>=0.40:
                matches.append((tok,best)); cursor=best+1
        if matches:
            st=asr[matches[0][1]]["start"]
            en=asr[matches[-1][1]]["end"]
        elif cursor < len(asr):
            st=asr[cursor]["start"]; en=min(audio_end,st+max(0.8,0.30*len(toks)))
        else:
            break
        en=min(audio_end,max(en,st+0.25))
        words=align_phrase(line,st,en,asr)
        if words:
            for w in words:
                w["phrase_id"]=pid; w["phrase_text"]=line
            scenes.append({"start":st,"end":min(audio_end,en+0.18),"words":words,"phrase_text":line,"phrase_id":pid})
    return scenes


def build_timed(lines, asr, audio_end):
    scenes=[]
    for i,line in enumerate(lines):
        st=float(line["start"])
        if st>=audio_end: continue
        en=float(line.get("end", lines[i+1]["start"] if i+1<len(lines) else audio_end))
        if i+1<len(lines): en=min(en,float(lines[i+1]["start"]))
        en=min(audio_end,en)
        if en<=st: continue
        words=align_phrase(line["text"],st,en,asr)
        if words:
            for w in words:
                w["phrase_id"]=i; w["phrase_text"]=line["text"]
            scenes.append({"start":st,"end":en,"words":words,"phrase_text":line["text"],"phrase_id":i})
    return scenes


def auto_scenes(asr,audio_end):
    if not asr: return []
    groups=[]; cur=[]
    for w in asr:
        if cur and w["start"]-cur[-1]["end"]>0.62:
            groups.append(cur); cur=[]
        cur.append(w)
    if cur: groups.append(cur)
    scenes=[]
    for pid,g in enumerate(groups):
        st=max(0,g[0]["start"]-0.03); en=min(audio_end,g[-1]["end"]+0.18)
        for w in g: w["phrase_id"]=pid; w["phrase_text"]=" ".join(x["word"] for x in g)
        scenes.append({"start":st,"end":en,"words":g,"phrase_text":" ".join(x["word"] for x in g),"phrase_id":pid})
    return scenes


def repair_scenes(scenes,audio_end):
    # Never allow zero-length, inverted or out-of-range scenes.
    fixed=[]
    for i,s in enumerate(sorted(scenes,key=lambda x:x["start"])):
        st=clamp(s["start"],0,audio_end); en=min(audio_end,max(st+0.08,float(s["end"])))
        words=[]
        for w in sorted(s.get("words",[]),key=lambda x:x["start"]):
            if w["start"]>=audio_end: continue
            ws=max(st,float(w["start"])); we=min(en,float(w["end"]))
            if we<=ws: we=min(en,ws+0.055)
            if we>ws:
                z=dict(w); z["start"]=ws; z["end"]=we; words.append(z)
        if words:
            s=dict(s); s["start"]=st; s["end"]=en; s["words"]=words; fixed.append(s)
    return fixed


def choose_font(scene_i, word_i, n, reg):
    main=[x for x in MAIN_FONTS if x in reg] or list(reg)
    alt=[x for x in ALT_FONTS if x in reg] or main
    # Thick font remains dominant; occasional alternate inside long phrases.
    if word_i==0: return main[scene_i % len(main)]
    if n>=6 and word_i in {2,n//2,n-2}:
        return alt[(scene_i+word_i)%len(alt)]
    return main[(scene_i + word_i//4)%len(main)]


def blue_word(scene_i, word_i, n):
    if scene_i % 2 != 0 or n < 3:
        return False
    return word_i in {1, n//2} and word_i < n


def draw_word(base,text,font,cx,y,color,progress):
    p=clamp(progress); e=ease(p); alpha=int(255*e)
    b=font.getbbox(text); x0,y0,x1,y1=b
    pad=max(26,int(font.size*0.22))
    lw=max(1,x1-x0)+pad*2; lh=max(1,y1-y0)+pad*2
    layer=Image.new("RGBA",(lw,lh),(0,0,0,0)); d=ImageDraw.Draw(layer)
    tx=pad-x0; ty=pad-y0
    shadow=(0,0,0,int(alpha*.20)) if color==WHITE else (255,255,255,int(alpha*.18))
    d.text((tx+2,ty+3),text,font=font,fill=shadow)
    d.text((tx,ty),text,font=font,fill=color+(alpha,))
    scale=0.94+0.06*e
    layer=layer.resize((max(1,int(lw*scale)),max(1,int(lh*scale))),Image.Resampling.LANCZOS)
    px=int(cx-layer.width/2); py=int(y+(1-e)*18)
    px=max(0,min(px,base.width-layer.width)); py=max(0,min(py,base.height-layer.height))
    base.alpha_composite(layer,(px,py))


def render_scene(scene, scene_i, W, H, t, reg, bg_blend=0.0):
    # IMPORTANT: the background is determined by the phrase index, not by render-loop state.
    # This guarantees black/white alternation even when there are timing gaps.
    target = BLACK if scene_i%2==0 else WHITE
    previous = WHITE if scene_i%2==0 else BLACK
    bg=np.full((H,W,3),target,dtype=np.float32)
    # Subtle transition at the END of a phrase, so the next frame naturally mixes into the next background.
    if bg_blend>0:
        bg=bg*(1-bg_blend)+np.full((H,W,3),previous,dtype=np.float32)*bg_blend
    image=Image.fromarray(np.uint8(bg)).convert("RGBA")
    words=scene.get("words",[])
    if not words:
        return image.convert("RGB")
    normal=WHITE if target==BLACK else BLACK
    visible=[]
    for i,w in enumerate(words):
        rel=t-(w["start"]-scene["start"])
        if rel>=0:
            visible.append((i,w,rel))
    if not visible:
        return image.convert("RGB")

    n=len(words)
    # Larger typography. Keep phrase on one composition, adding words one-by-one.
    max_width=int(W*0.88)
    base_size=int(H*0.078 if n<=5 else H*0.070 if n<=8 else H*0.060)
    # Build rows from the visible words only. A word may use a different font.
    rows=[]; row=[]; row_width=0; spacing=max(10,int(base_size*.075))
    for i,w,rel in visible:
        raw_word = norm(w["word"])
        name=choose_font(scene_i,i,n,reg)
        name=safe_font_name(raw_word, name, reg)
        f=fit_font(raw_word.upper(),name,reg,base_size,max_width*0.42,min_size=34)
        ww=bbox(w["word"].upper(),f)[2]
        if row and row_width+spacing+ww>max_width:
            rows.append(row); row=[]; row_width=0
        row.append((i,w,rel,f,ww)); row_width += ww + (spacing if len(row)>1 else 0)
    if row: rows.append(row)
    line_gap=int(base_size*1.04)
    total_h=len(rows)*line_gap
    start_y=int(H/2-total_h/2)

    for ri,row in enumerate(rows):
        spacing=max(8,int(base_size*.075))
        width=sum(x[4] for x in row)+spacing*max(0,len(row)-1)
        x=(W-width)/2
        y=start_y+ri*line_gap
        for i,w,rel,f,ww in row:
            text=norm(w["word"]).upper()
            color=ROYAL if blue_word(scene_i,i,n) else normal
            draw_word(image,text,f,x+ww/2,y,color,rel/0.18)
            x+=ww+spacing

    # Smooth phrase fade-out. This is visual only; next phrase is rendered independently.
    fade_start=max(0,scene["end"]-scene["start"]-0.20)
    if t>fade_start:
        a=int(255*clamp((t-fade_start)/0.20))
        overlay=Image.new("RGBA",(W,H),(0,0,0,a))
        image=Image.alpha_composite(image,overlay)
    return image.convert("RGB")


def render_video(audio_path, scenes, reg, output, resolution, quality, progress):
    W,H=resolution
    dur=media_duration(audio_path)
    if dur<=0: raise RuntimeError("NÃ£o foi possÃ­vel identificar a duraÃ§Ã£o da mÃºsica.")
    # Use audio duration as the hard upper bound. Captions never vanish at 10â15 seconds.
    total_frames=max(1,int(math.ceil(dur*FPS)))
    silent=Path(output).with_name("silent_lyric.mp4")
    crf="14" if quality=="Alta qualidade" else "17"
    preset="slow" if quality=="Alta qualidade" else "medium"
    cmd=[ffmpeg(),"-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{W}x{H}","-r",str(FPS),"-i","-","-an","-c:v","libx264","-preset",preset,"-crf",crf,"-pix_fmt","yuv420p","-movflags","+faststart",str(silent)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    scene_i=0
    try:
        for frame_no in range(total_frames):
            now=frame_no/FPS
            # Explicitly choose the active scene by time. Do NOT rely on a single moving index.
            while scene_i+1<len(scenes) and now>=scenes[scene_i]["end"]:
                scene_i+=1
            if scenes and scenes[scene_i]["start"]<=now<scenes[scene_i]["end"]:
                scene=scenes[scene_i]
                local=now-scene["start"]
                blend=0.0
                if local<0.12 and scene_i>0:
                    blend=1.0-clamp(local/0.12)
                frame=render_scene(scene,scene_i,W,H,local,reg,blend)
            else:
                # Gaps: keep a monochrome frame, alternating according to the NEXT phrase.
                next_i=scene_i if not scenes else min(scene_i+(1 if now>=scenes[scene_i]["end"] else 0),len(scenes)-1)
                bg=BLACK if next_i%2==0 else WHITE
                frame=np.full((H,W,3),bg,dtype=np.uint8)
            proc.stdin.write(np.asarray(frame,dtype=np.uint8).tobytes())
            if progress and frame_no%FPS==0:
                progress.progress(min(.93,frame_no/max(1,total_frames)*.93),text=f"Renderizando {int(frame_no/max(1,total_frames)*100)}%")
        proc.stdin.close()
        err=proc.stderr.read().decode("utf-8","replace")
        code=proc.wait()
        if code!=0: raise RuntimeError("FFmpeg falhou ao criar o vÃ­deo:\n"+err[-5000:])
    except Exception:
        try: proc.stdin.close()
        except Exception: pass
        try: proc.kill()
        except Exception: pass
        raise
    final=[ffmpeg(),"-y","-i",str(silent),"-i",audio_path,"-map","0:v:0","-map","1:a:0","-c:v","copy","-c:a","aac","-b:a","256k","-t",f"{dur:.3f}","-movflags","+faststart",str(output)]
    try:
        run(final,timeout=max(180,int(dur*8)))
    finally:
        silent.unlink(missing_ok=True)
    if progress: progress.progress(1.0,text="VÃ­deo concluÃ­do.")


st.set_page_config(page_title="Lyric AI Studio",page_icon="ðµ",layout="centered")
st.title("ðµ Lyric AI Studio")
st.caption(f"Word-Sync Kinetic Engine Â· {APP_VERSION}")

audio=st.file_uploader("1. MÃºsica ou vÃ­deo",type=["mp3","wav","m4a","mp4","mov","webm"])
lyrics=st.text_area("2. Letra oficial",height=240,placeholder="Uma frase por linha.\n\nOu com tempos:\n00:02.3 - 00:06.8 | Minha frase")

c1,c2=st.columns(2)
with c1:
    model=st.selectbox("Reconhecimento",["small","medium","large-v3-turbo","large-v3"],index=2)
with c2:
    quality=st.selectbox("Qualidade",["Equilibrado","Alta qualidade"],index=1)
resolution=st.selectbox("ResoluÃ§Ã£o",["1080 Ã 1920","720 Ã 1280"],index=0)

st.info("O fundo agora alterna PRETO/BRANCO por frase, com mistura suave na troca. As palavras aparecem uma por uma nos timestamps do cantor. Azul royal Ã© usado apenas em palavras selecionadas.")
reg=font_registry()
st.caption(f"Fontes disponÃ­veis: {len(reg)}")

if st.button("ð CRIAR LYRIC VIDEO",type="primary",use_container_width=True):
    if not audio:
        st.error("Envie a mÃºsica primeiro.")
        st.stop()
    temp=Path(tempfile.mkdtemp(prefix="lyric_ai_"))
    audio_path=temp/safe(audio.name)
    audio_path.write_bytes(audio.getbuffer())
    status=st.empty(); progress=st.progress(0,text="Preparandoâ¦")
    try:
        dur=media_duration(str(audio_path))
        if dur<=0: raise RuntimeError("NÃ£o foi possÃ­vel identificar a duraÃ§Ã£o do Ã¡udio.")
        end_time=detect_audio_end(str(audio_path),dur)
        status.write(f"â±ï¸ DuraÃ§Ã£o: {dur:.2f}s Â· fim Ãºtil detectado: {end_time:.2f}s")

        try:
            asr,lang=transcribe(str(audio_path),model,status)
        except Exception:
            if model=="small": raise
            status.warning("O modelo escolhido falhou no ambiente; tentando small automaticamenteâ¦")
            asr,lang=transcribe(str(audio_path),"small",status)
        asr=[w for w in asr if w["start"]<end_time]
        if not asr and not lyrics.strip():
            raise RuntimeError("Nenhuma palavra foi reconhecida. Cole a letra oficial e tente novamente.")

        timed=parse_timed(lyrics) if lyrics.strip() else []
        if timed:
            status.write("ð Tempos das frases encontrados. Refinando cada palavra pelo Ã¡udioâ¦")
            scenes=build_timed(timed,asr,end_time)
        elif lyrics.strip():
            status.write("ð§  Alinhando a letra oficial ao canto real palavra por palavraâ¦")
            scenes=align_plain(lyrics,asr,end_time)
        else:
            scenes=auto_scenes(asr,end_time)
        scenes=repair_scenes(scenes,end_time)
        if not scenes:
            raise RuntimeError("A IA nÃ£o conseguiu criar frases sincronizadas. Tente colar a letra oficial.")

        progress.progress(.20,text=f"SincronizaÃ§Ã£o pronta: {sum(len(s['words']) for s in scenes)} palavras em {len(scenes)} frases.")
        size=(1080,1920) if resolution.startswith("1080") else (720,1280)
        output=temp/"lyric_ai_final.mp4"
        status.write("ð¬ Renderizando palavra por palavra e alternando os fundosâ¦")
        render_video(str(audio_path),scenes,reg,str(output),size,quality,progress)
        status.success("â VÃ­deo criado.")
        st.video(str(output))
        st.download_button("â¬ï¸ BAIXAR MP4",data=output.read_bytes(),file_name="lyric_ai_final.mp4",mime="video/mp4",use_container_width=True)
        with st.expander("DiagnÃ³stico"):
            st.caption("A renderizaÃ§Ã£o usa NFC/Unicode e uma fonte de seguranÃ§a para caracteres portugueses. Se uma palavra estiver errada no diagnÃ³stico, o erro veio da transcriÃ§Ã£o; se estiver correta no diagnÃ³stico, ela deve ser renderizada exatamente com os acentos.")
            st.write(f"DuraÃ§Ã£o do arquivo: **{dur:.2f}s**")
            st.write(f"Fim Ãºtil detectado: **{end_time:.2f}s**")
            st.write(f"Frases renderizadas: **{len(scenes)}**")
            st.write(f"Palavras renderizadas: **{sum(len(s['words']) for s in scenes)}**")
            st.write(f"Idioma: **{lang}**")
            st.code("\n".join(f"{s['start']:.2f}-{s['end']:.2f} | {s['phrase_text']}" for s in scenes))
    except Exception as e:
        st.error("â A geraÃ§Ã£o falhou.")
        st.code(str(e))
