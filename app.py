import os, re, math, shutil, subprocess, tempfile, difflib, unicodedata
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

APP_VERSION = "15.0-LITE-FULL"
BLACK = (5, 5, 7)
WHITE = (248, 248, 246)
ROYAL = (45, 92, 255)
FPS = 24

# Apenas fontes locais: sem downloads durante a inicialização do Streamlit.
FONT_PATHS = {
    "Grossa": "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
    "Limpa": "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    "Serif": "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "Safe": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
}


# --------------------------------------------------------------------------------------
# Utilidades básicas
# --------------------------------------------------------------------------------------

def safe(s):
    """Sanitiza nomes de arquivo para evitar path traversal e caracteres inválidos."""
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s or "file")
    return s[:100] or "file"


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
    raise RuntimeError("FFmpeg não encontrado. Verifique imageio-ffmpeg no requirements.txt.")


def cmdrun(cmd, timeout=None):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError(p.stderr[-4000:])
    return p.stdout, p.stderr


def duration(path):
    _, err = cmdrun([ffmpeg(), "-hide_banner", "-i", path, "-f", "null", "-"], timeout=90)
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", err)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else 0.0


def detect_audio_end(path, dur):
    """Detecta silêncio real no final da faixa para não deixar a legenda sumir cedo demais
    nem o vídeo terminar com silêncio morto. Conservador: só encurta se o silêncio estiver
    genuinamente perto do fim."""
    try:
        _, err = cmdrun(
            [ffmpeg(), "-hide_banner", "-i", path, "-af", "silencedetect=noise=-40dB:d=0.65", "-f", "null", "-"],
            timeout=max(60, int(dur * 2)),
        )
        starts = [float(x) for x in re.findall(r"silence_start:\s*([\d.]+)", err)]
        ends = [float(x) for x in re.findall(r"silence_end:\s*([\d.]+)", err)]
        if starts and starts[-1] >= dur * 0.82:
            if ends and ends[-1] >= starts[-1]:
                return min(dur, ends[-1] + 0.12)
            return min(dur, starts[-1] + 0.12)
    except Exception:
        pass
    return dur


def norm(s):
    s = unicodedata.normalize("NFC", s or "")
    s = s.replace("\ufeff", "").replace("\u200b", "")
    s = "".join(c for c in s if c == "\n" or unicodedata.category(c)[0] != "C")
    return re.sub(r"\s+", " ", s).strip()


def key(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-zA-Z0-9]", "", s).lower()


def sim(a, b):
    a, b = key(a), key(b)
    if not a or not b:
        return 0
    if a == b:
        return 1
    if a in b or b in a:
        return 0.9
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, float(x)))


def parse_time(s):
    s = s.strip().replace(",", ".")
    if ":" in s:
        a, b = s.split(":", 1)
        return int(a) * 60 + float(b)
    return float(s)


def parse_lyrics(text):
    """Suporta 'tempo - tempo' na própria linha + letra na linha seguinte, ou forma inline."""
    lines = [x.strip() for x in (text or "").replace("\r", "").split("\n")]
    out = []
    i = 0
    pat = re.compile(
        r"^(\d{1,2}:\d{2}(?:[.,]\d{1,3})?|\d+(?:[.,]\d+)?)\s*[-–—]\s*"
        r"(\d{1,2}:\d{2}(?:[.,]\d{1,3})?|\d+(?:[.,]\d+)?)(?:\s*\|\s*(.*))?$"
    )
    while i < len(lines):
        if not lines[i]:
            i += 1
            continue
        m = pat.match(lines[i])
        if m:
            st_, en_ = parse_time(m.group(1)), parse_time(m.group(2))
            lyric = norm(m.group(3) or "")
            if not lyric and i + 1 < len(lines):
                j = i + 1
                while j < len(lines) and not lines[j]:
                    j += 1
                if j < len(lines) and not pat.match(lines[j]):
                    lyric = norm(lines[j])
                    i = j
            if lyric and en_ > st_:
                out.append({"start": st_, "end": en_, "text": lyric})
        i += 1
    return sorted(out, key=lambda x: x["start"])


def plain_lines(text):
    return [norm(x) for x in (text or "").splitlines() if norm(x)]


# --------------------------------------------------------------------------------------
# Reconhecimento de voz
# --------------------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_model(name):
    from faster_whisper import WhisperModel
    return WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=2, num_workers=1)


def transcribe(path, name, status):
    status.write(f"🎙️ Reconhecendo com **{name}**…")
    model = get_model(name)
    segs, info = model.transcribe(
        path, language="pt", word_timestamps=True, beam_size=1, best_of=1, patience=1.0,
        temperature=0.0, condition_on_previous_text=False, vad_filter=False,
        initial_prompt="Letra de música brasileira em português. Reconheça todas as palavras, "
                       "repetições, gírias e contrações. Não resuma nem traduza.",
    )
    words = []
    for seg in segs:
        for w in (seg.words or []):
            t = norm(w.word)
            if t and len(t) <= 40:
                letters = sum(ch.isalpha() for ch in t)
                digits = sum(ch.isdigit() for ch in t)
                if letters >= 1 and digits >= 1 and letters + digits >= 3:
                    continue  # artefato provável de ASR (ex: "A1A")
                words.append({
                    "word": t, "start": float(w.start), "end": float(w.end),
                    "prob": float(getattr(w, "probability", 0) or 0),
                })
    words.sort(key=lambda x: x["start"])
    return words, getattr(info, "language", "pt")


def transcribe_with_fallback(path, model_name, status):
    try:
        return transcribe(path, model_name, status)
    except Exception:
        if model_name == "small":
            raise
        status.warning("O modelo escolhido falhou neste ambiente; tentando **small** automaticamente…")
        return transcribe(path, "small", status)


# --------------------------------------------------------------------------------------
# Alinhamento palavra a palavra
# --------------------------------------------------------------------------------------

def align_phrase(text, st_, en_, asr):
    toks = re.findall(r"\S+", text)
    cand = [w for w in asr if w["end"] >= st_ - 0.4 and w["start"] <= en_ + 0.4]
    used = -1
    mapped = []
    for tok in toks:
        best = None
        bs = 0
        for j in range(used + 1, min(len(cand), used + 31)):
            s = sim(tok, cand[j]["word"])
            if st_ <= cand[j]["start"] <= en_:
                s += 0.05
            if s > bs:
                bs = s
                best = j
        if best is not None and bs >= 0.38:
            used = best
            mapped.append((tok, cand[best]))
        else:
            mapped.append((tok, None))
    known = [i for i, x in enumerate(mapped) if x[1] is not None]
    result = []
    for i, (tok, w) in enumerate(mapped):
        if w:
            a = max(st_, w["start"])
            b = min(en_, max(a + 0.06, w["end"]))
        else:
            p = max([k for k in known if k < i], default=-1)
            n = min([k for k in known if k > i], default=len(mapped))
            a = result[p]["end"] if p >= 0 else st_
            b = mapped[n][1]["start"] if n < len(mapped) else en_
            if b < a:
                b = a + 0.08
            a = a + (b - a) * (i - p) / max(1, n - p)
            b = a + (b - a) / max(1, n - p)
            a = max(st_, min(a, en_ - 0.06))
            b = min(en_, max(b, a + 0.06))
        result.append({"word": tok, "start": a, "end": b})
    return result


def build_scenes(lyrics, asr, dur):
    timed = parse_lyrics(lyrics)
    scenes = []
    if timed:
        for i, l in enumerate(timed):
            st_ = max(0, min(dur, l["start"]))
            en_ = min(dur, l["end"])
            if i + 1 < len(timed):
                en_ = min(en_, timed[i + 1]["start"])
            if en_ <= st_:
                continue
            ws = align_phrase(l["text"], st_, en_, asr)
            if ws:
                scenes.append({"start": st_, "end": en_, "words": ws, "text": l["text"]})
        return scenes
    lines = plain_lines(lyrics)
    cursor = 0
    for line in lines:
        toks = re.findall(r"\S+", line)
        matches = []
        for tok in toks:
            best = None
            bs = 0
            for j in range(cursor, min(len(asr), cursor + 60)):
                s = sim(tok, asr[j]["word"])
                if s > bs:
                    bs = s
                    best = j
                if s >= 0.99:
                    break
            if best is not None and bs >= 0.38:
                matches.append(best)
                cursor = best + 1
        if matches:
            st_ = asr[matches[0]]["start"]
            en_ = asr[matches[-1]]["end"]
        elif cursor < len(asr):
            st_ = asr[cursor]["start"]
            en_ = min(dur, st_ + max(0.8, 0.28 * len(toks)))
            cursor += 1
        else:
            break
        en_ = min(dur, max(en_, st_ + 0.4))
        ws = align_phrase(line, st_, en_, asr)
        if ws:
            scenes.append({"start": st_, "end": en_, "words": ws, "text": line})
    return scenes


def auto_scenes(asr, dur):
    groups = []
    cur = []
    for w in asr:
        if cur and w["start"] - cur[-1]["end"] > 0.65:
            groups.append(cur)
            cur = []
        cur.append(w)
    if cur:
        groups.append(cur)
    return [
        {
            "start": max(0, g[0]["start"] - 0.02),
            "end": min(dur, g[-1]["end"] + 0.18),
            "words": g,
            "text": " ".join(w["word"] for w in g),
        }
        for g in groups
    ]


def repair_scenes(scenes, dur):
    """Blinda contra cenas com duração zero/negativa ou fora do intervalo do áudio."""
    fixed = []
    for s in sorted(scenes, key=lambda x: x["start"]):
        st_ = clamp(s["start"], 0, dur)
        en_ = min(dur, max(st_ + 0.08, float(s["end"])))
        words = []
        for w in sorted(s.get("words", []), key=lambda x: x["start"]):
            if w["start"] >= dur:
                continue
            ws = max(st_, float(w["start"]))
            we = min(en_, float(w["end"]))
            if we <= ws:
                we = min(en_, ws + 0.055)
            if we > ws:
                z = dict(w)
                z["start"] = ws
                z["end"] = we
                words.append(z)
        if words:
            z = dict(s)
            z["start"] = st_
            z["end"] = en_
            z["words"] = words
            fixed.append(z)
    return fixed


# --------------------------------------------------------------------------------------
# Renderização (uma imagem por estado de legenda + concat no FFmpeg = leve)
# --------------------------------------------------------------------------------------

def font(path, size):
    return ImageFont.truetype(path, max(18, int(size)))


def choose_font_name(scene_i, word_i):
    names = ["Grossa", "Grossa", "Limpa", "Grossa", "Serif", "Grossa"]
    return names[(scene_i + word_i) % len(names)]


def is_blue_word(scene_i, word_i, n):
    if scene_i % 2 != 0 or n < 3:
        return False
    return word_i in {1, n // 2} and word_i < n


def render_image(scene, idx, t, W, H):
    bg = BLACK if idx % 2 == 0 else WHITE
    fg = WHITE if bg == BLACK else BLACK
    im = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(im)
    words = []
    for wi, w in enumerate(scene["words"]):
        if t >= w["start"] - scene["start"]:
            words.append((wi, w))
    if not words:
        return im
    n = len(scene["words"])
    size = int(H * (0.075 if n <= 5 else 0.062 if n <= 8 else 0.052))
    maxw = int(W * 0.88)
    rows = []
    row = []
    width = 0
    for wi, w in words:
        name = choose_font_name(idx, wi)
        path = FONT_PATHS.get(name, FONT_PATHS["Safe"])
        f = font(path, size)
        text = w["word"].upper()
        box = d.textbbox((0, 0), text, font=f)
        ww = box[2] - box[0]
        while ww > maxw * 0.43 and size > 32:
            size -= 2
            f = font(path, size)
            box = d.textbbox((0, 0), text, font=f)
            ww = box[2] - box[0]
        if row and width + ww + max(8, size // 14) > maxw:
            rows.append(row)
            row = []
            width = 0
        row.append((wi, w, f, ww))
        width += ww + (max(8, size // 14) if len(row) > 1 else 0)
    if row:
        rows.append(row)
    gap = int(size * 1.12)
    total = len(rows) * gap
    y = H // 2 - total // 2
    for ri, row in enumerate(rows):
        spacing = max(8, size // 14)
        totalw = sum(x[3] for x in row) + spacing * (len(row) - 1)
        x = (W - totalw) / 2
        for wi, w, f, ww in row:
            text = w["word"].upper()
            color = ROYAL if is_blue_word(idx, wi, n) else fg
            p = max(0, min(1, (t - (w["start"] - scene["start"])) / 0.16))
            alpha = int(255 * (1 - (1 - p) ** 3))
            layer = Image.new("RGBA", (ww + 24, f.size + 30), (0, 0, 0, 0))
            ld = ImageDraw.Draw(layer)
            ld.text((12, 8), text, font=f, fill=color + (alpha,))
            im = Image.alpha_composite(im.convert("RGBA"), layer, (int(x - 12), int(y + ri * gap))).convert("RGB")
            x += ww + spacing
    return im


def render_video(audio, scenes, out, resolution, quality, status, progress):
    W, H = resolution
    dur = duration(audio)
    tmp = Path(tempfile.mkdtemp(prefix="lyric_frames_"))
    idx = 0
    crf = "18" if quality == "Alta qualidade" else "22"
    preset = "medium" if quality == "Alta qualidade" else "veryfast"
    try:
        states = []
        total_scenes = max(1, len(scenes))
        for si, s in enumerate(scenes):
            starts = sorted(set([0.0] + [max(0, w["start"] - s["start"]) for w in s["words"]]))
            for k, st_ in enumerate(starts):
                en_ = starts[k + 1] if k + 1 < len(starts) else s["end"] - s["start"]
                if en_ - st_ < 0.04:
                    continue
                img = render_image(s, si, st_, W, H)
                p = tmp / f"f{idx:05d}.png"
                img.save(p, optimize=True)
                states.append((p, en_ - st_))
                idx += 1
            if progress:
                progress.progress(min(0.85, 0.05 + 0.55 * (si + 1) / total_scenes), text=f"Gerando quadros… {si + 1}/{total_scenes} frases")
        if not states:
            raise RuntimeError("Nenhum quadro de legenda foi criado.")
        concat = tmp / "concat.txt"
        with concat.open("w", encoding="utf-8") as f:
            for p, d in states:
                f.write(f"file '{p.as_posix()}'\nduration {d:.4f}\n")
            f.write(f"file '{states[-1][0].as_posix()}'\n")
        silent = tmp / "silent.mp4"
        status.write("🎬 Montando o vídeo…")
        if progress:
            progress.progress(0.88, text="Codificando vídeo…")
        cmd = [
            ffmpeg(), "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
            "-r", str(FPS), "-c:v", "libx264", "-preset", preset, "-crf", crf,
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(silent),
        ]
        cmdrun(cmd, timeout=max(180, int(dur * 10)))
        if progress:
            progress.progress(0.96, text="Adicionando áudio…")
        cmdrun(
            [
                ffmpeg(), "-y", "-i", str(silent), "-i", audio, "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-t", f"{dur:.3f}",
                "-movflags", "+faststart", out,
            ],
            timeout=max(120, int(dur * 5)),
        )
        if progress:
            progress.progress(1.0, text="Vídeo concluído.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------------------

st.set_page_config(page_title="Lyric AI Studio", page_icon="🎵")
st.title("🎵 Lyric AI Studio")
st.caption(f"Word-Sync Kinetic Engine · {APP_VERSION}")

audio = st.file_uploader("1. Música ou vídeo", type=["mp3", "wav", "m4a", "mp4", "mov", "webm"])
lyrics = st.text_area(
    "2. Letra oficial", height=220,
    placeholder="Uma frase por linha.\n\nOu:\n00:02.3 - 00:06.8\nMinha frase",
)

c1, c2 = st.columns(2)
with c1:
    model = st.selectbox("Reconhecimento", ["small", "medium", "large-v3-turbo", "large-v3"], index=1)
with c2:
    quality = st.selectbox("Qualidade", ["Equilibrado", "Alta qualidade"], index=0)
resolution = st.selectbox("Resolução", ["720 × 1280", "1080 × 1920"], index=0)

st.caption(
    "Versão leve: sem downloads de fontes na inicialização, modelo em cache, "
    "renderização por quadros estáticos + fallback automático de modelo."
)

if st.button("🚀 CRIAR LYRIC VIDEO", type="primary", use_container_width=True):
    if not audio:
        st.error("Envie a música primeiro.")
        st.stop()

    temp = Path(tempfile.mkdtemp(prefix="lyric_ai_"))
    ap = temp / (safe(Path(audio.name).stem) + Path(audio.name).suffix.lower())
    ap.write_bytes(audio.getbuffer())

    status = st.empty()
    progress = st.progress(0, text="Preparando…")
    try:
        dur = duration(str(ap))
        if dur <= 0:
            raise RuntimeError("Não foi possível identificar a duração do áudio/vídeo.")
        end_time = detect_audio_end(str(ap), dur)
        status.write(f"⏱️ Duração: {dur:.2f}s · fim útil detectado: {end_time:.2f}s")
        progress.progress(0.03, text="Transcrevendo áudio…")

        asr, lang = transcribe_with_fallback(str(ap), model, status)
        asr = [w for w in asr if w["start"] < end_time]
        if not asr and not lyrics.strip():
            raise RuntimeError("Nenhuma palavra foi reconhecida. Cole a letra oficial e tente novamente.")

        scenes = build_scenes(lyrics, asr, end_time) if lyrics.strip() else auto_scenes(asr, end_time)
        if not scenes:
            scenes = auto_scenes(asr, end_time)
        scenes = repair_scenes(scenes, end_time)
        if not scenes:
            raise RuntimeError("Não foi possível criar as frases sincronizadas.")

        total = sum(len(s["words"]) for s in scenes)
        status.write(f"📝 {len(scenes)} frases · {total} palavras")

        size = (720, 1280) if resolution.startswith("720") else (1080, 1920)
        out = temp / "lyric_ai_final.mp4"
        render_video(str(ap), scenes, str(out), size, quality, status, progress)

        status.success("✅ Vídeo criado.")
        st.video(str(out))
        st.download_button(
            "⬇️ BAIXAR MP4", out.read_bytes(), "lyric_ai_final.mp4", "video/mp4",
            use_container_width=True,
        )
        with st.expander("Diagnóstico"):
            st.write(f"Duração do arquivo: **{dur:.2f}s**")
            st.write(f"Fim útil detectado: **{end_time:.2f}s**")
            st.write(f"Frases: **{len(scenes)}**")
            st.write(f"Palavras: **{total}**")
            st.write(f"Idioma: **{lang}**")
            st.code("\n".join(f"{s['start']:.2f}-{s['end']:.2f} | {s['text']}" for s in scenes))
    except Exception as e:
        st.error("❌ A geração falhou.")
        st.code(str(e))
    finally:
        pass