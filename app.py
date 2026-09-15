# -*- coding: utf-8 -*-
import os, re, math, shutil, subprocess, tempfile, urllib.request, difflib, unicodedata
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

APP_VERSION = "10.0-DIRECTOR-CUT"
FPS = 30
W, H = 1080, 1920

CACHE = Path(".lyric_cache")
FONT_DIR = CACHE / "fonts"
FONT_DIR.mkdir(parents=True, exist_ok=True)

# Tipografia selecionada a dedo para Direção de Arte Elegante
FONT_SOURCES = {
    "Anton": "https://raw.githubusercontent.com/google/fonts/main/ofl/anton/Anton-Regular.ttf",
    "Playfair Display": "https://raw.githubusercontent.com/google/fonts/main/ofl/playfairdisplay/PlayfairDisplay%5Bwght%5D.ttf",
    "Montserrat": "https://raw.githubusercontent.com/google/fonts/main/ofl/montserrat/Montserrat%5Bwght%5D.ttf",
    "DM Serif Display": "https://raw.githubusercontent.com/google/fonts/main/ofl/dmserifdisplay/DMSerifDisplay-Regular.ttf",
    "Bebas Neue": "https://raw.githubusercontent.com/google/fonts/main/ofl/bebasneue/BebasNeue-Regular.ttf"
}

# Paleta Sofisticada
NAVY = (22, 28, 38)
CREAM = (248, 245, 238)
SAGE = (120, 138, 123)
BLACK_INK = (15, 15, 18)

# Dicionário Semântico e de Impacto
EMOTIONAL = {
    "amor": 3, "saudade": 3, "coração": 3, "coracao": 3, "vida": 2, "nunca": 2, "sempre": 2,
    "volta": 2, "voltar": 2, "paixão": 3, "paixao": 3, "desejo": 2, "chora": 2, "chorar": 2,
    "você": 1, "voce": 1, "céu": 2, "ceu": 2, "noite": 2, "luz": 2, "mundo": 1, "tempo": 1,
    "verdade": 2, "mentira": 2, "bebida": 2, "beber": 2, "sofrer": 3, "dor": 3, "lágrima": 3,
    "adeus": 3, "sol": 2, "embora": 2, "deus": 2
}

def clamp(x, a, b): return max(a, min(b, x))
def smooth(t): t = clamp(t, 0, 1); return t * t * (3 - 2 * t)
def ease_out(t): t = clamp(t, 0, 1); return 1 - (1 - t)**3
def safe(s): return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:100]
def norm(s): return re.sub(r"\s+", " ", s or "").strip()
def token(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-zA-Z0-9]", "", s).lower()
def sim(a, b):
    a, b = token(a), token(b)
    if not a or not b: return 0
    if a == b: return 1
    if a in b or b in a: return .90
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()

def ffmpeg():
    try:
        import imageio_ffmpeg
        p = imageio_ffmpeg.get_ffmpeg_exe()
        if p: return p
    except Exception: pass
    p = shutil.which("ffmpeg")
    if p: return p
    raise RuntimeError("FFmpeg não encontrado.")

def run(cmd, timeout=300):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if p.returncode: raise RuntimeError(p.stderr[-7000:])
    return p.stdout

def duration(path):
    ff = ffmpeg()
    p = subprocess.run([ff, "-hide_banner", "-i", path, "-f", "null", "-"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", timeout=60)
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", p.stderr)
    return int(m.group(1))*3600 + int(m.group(2))*60 + float(m.group(3)) if m else 0.0

@st.cache_resource(show_spinner=False)
def fonts():
    out = {}
    for name, url in FONT_SOURCES.items():
        p = FONT_DIR / (safe(name) + ".ttf")
        if not p.exists() or p.stat().st_size < 10000:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "LyricAIStudio"})
                with urllib.request.urlopen(req, timeout=25) as r: p.write_bytes(r.read())
            except Exception: pass
        if p.exists() and p.stat().st_size > 10000: out[name] = str(p)
    return out

@st.cache_resource(show_spinner=False)
def whisper(model):
    from faster_whisper import WhisperModel
    return WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=max(2, min(8, os.cpu_count() or 4)), num_workers=1)

def transcribe(path, model, status=None):
    m = whisper(model)
    if status: status.write(f"🎙️ Escutando com atenção ({model})…")
    segs, info = m.transcribe(
        path, language="pt", task="transcribe", beam_size=8,
        condition_on_previous_text=True, word_timestamps=True,
        initial_prompt="Música cantada. Letra precisa, detectando respirações e dinâmicas."
    )
    words = []
    for seg in segs:
        for w in seg.words:
            x = norm(w.word)
            if x: words.append({"word": x, "start": float(w.start), "end": float(w.end), "prob": float(getattr(w, "probability", 0) or 0)})
    return words, getattr(info, "language", "pt")

# ==========================================
# MOTOR DE DIREÇÃO: CHUNKING E ESTRUTURA
# ==========================================
def subdivide_scenes(scenes: List[Dict]) -> List[Dict]:
    """ Quebra blocos de texto grandes ou com pausas respiratórias, focando em legibilidade rápida """
    new_scenes = []
    for s in scenes:
        words = s["words"]
        cur = []
        for w in words:
            if not cur:
                cur.append(w)
                continue
            gap = w["start"] - cur[-1]["end"]
            # Quebra se o cantor pauser mais de 0.5s ou se tivermos mais de 4 palavras (chunk ideal)
            if gap > 0.5 or len(cur) >= 4:
                new_scenes.append({"start": cur[0]["start"], "end": cur[-1]["end"], "words": cur})
                cur = [w]
            else:
                cur.append(w)
        if cur:
            new_scenes.append({"start": cur[0]["start"], "end": cur[-1]["end"], "words": cur})
            
    # Fecha buracos de tempo entre frases para manter a tela fluindo (sem black screens)
    for i in range(len(new_scenes) - 1):
        mid = (new_scenes[i]["end"] + new_scenes[i+1]["start"]) / 2
        new_scenes[i]["end"] = max(new_scenes[i]["end"], mid)
        new_scenes[i+1]["start"] = min(new_scenes[i+1]["start"], mid)
        
    # Recalcula textos
    for i, s in enumerate(new_scenes):
        s["idx"] = i
        s["phrase_text"] = " ".join(w["word"] for w in s["words"])
    return new_scenes

def build_plain(text, asr, aend):
    # (Alinhamento simplificado preservado e envelopado pelo smart chunking)
    lines = [norm(x) for x in text.splitlines() if norm(x)]
    scenes = []; cursor = 0
    for pid, line in enumerate(lines):
        toks = re.findall(r"\S+", line)
        if not toks: continue
        best = None; score = 0
        for j in range(cursor, min(len(asr), cursor+45)):
            s = sim(toks[0], asr[j]["word"])
            if s > score: score = s; best = j
        if best is None or score < .35: continue
        st = asr[best]["start"]; idx = best
        for tok in toks[1:]:
            bi = None; bs = 0
            for j in range(idx+1, min(len(asr), idx+25)):
                s = sim(tok, asr[j]["word"])
                if s > bs: bs = s; bi = j
                if s >= .98: break
            if bi is not None and bs >= .35: idx = bi
        en = min(aend, asr[idx]["end"] + .18)
        words = [{"word": toks[i], "start": st + (en-st)*(i/len(toks)), "end": st + (en-st)*((i+1)/len(toks))} for i in range(len(toks))]
        # Overwrite starts/ends using ASR matches dynamically for better sync (omitted heavy DP for brevity, using approx map)
        scenes.append({"start": max(0, st-.02), "end": en, "words": words})
        cursor = idx + 1
    return subdivide_scenes(scenes)

def apply_direction(scenes: List[Dict], total_duration: float):
    """ Analisa a música para encontrar Clímax, Energia e Hierarquia """
    if not scenes: return
    
    # 1. First Frame Logic (Não deixar o início vazio)
    if scenes[0]["start"] > 0:
        scenes[0]["start"] = 0.0 # Estende o primeiro take para o zero
        
    climax_score = 0
    climax_idx = 0
    
    for i, s in enumerate(scenes):
        # Calcula densidade de energia (palavras faladas rapidamente = alta energia)
        dur = max(0.2, s["end"] - s["start"])
        density = len(s["words"]) / dur
        
        # O Clímax geralmente ocorre no terço final (50% a 85% do vídeo)
        rel_pos = s["start"] / total_duration
        pos_weight = 1.5 if 0.5 <= rel_pos <= 0.85 else 0.8
        
        energy = density * pos_weight
        s["energy"] = energy
        s["is_climax"] = False
        
        if energy > climax_score and rel_pos > 0.4:
            climax_score = energy
            climax_idx = i

    # Marcação da cena mais importante
    if scenes: scenes[climax_idx]["is_climax"] = True

# ==========================================
# MOTOR DE ARTE: CORES, FONTES, EFEITOS
# ==========================================
@dataclass
class Style:
    bg: Tuple[int,int,int]
    fg: Tuple[int,int,int]
    fg_dim: Tuple[int,int,int]
    accent: Tuple[int,int,int]
    font: str
    vfx: str

def style_for(scene, idx, reg):
    text_lower = scene.get("phrase_text", "").lower()
    
    # Alternância orgânica baseada em humor (evita estática excessiva)
    vibe = (idx // 3) % 3
    if vibe == 0:
        bg, fg, accent = BLACK_INK, CREAM, (110, 160, 255) # Dark Mood
    elif vibe == 1:
        bg, fg, accent = NAVY, CREAM, (250, 200, 100) # Emotional Mood (Gold accent)
    else:
        bg, fg, accent = CREAM, BLACK_INK, (45, 92, 255) # High Contrast Mood
        
    # Dim color for Pre-reveal (faint text waiting to be sung)
    fg_dim = (fg[0], fg[1], fg[2], 30)

    # Fonte dinâmica: Usa Playfair/DM Serif para mais lentas (elegantes) e Anton/Bebas para energia
    fonts = list(reg.keys())
    primary_font = "Playfair Display" if "Playfair Display" in reg else fonts[0]
    impact_font = "Anton" if "Anton" in reg else fonts[-1]
    
    font_choice = impact_font if scene.get("energy", 0) > 4.0 else primary_font

    # Análise semântica de VFX
    vfx = "none"
    if any(w in text_lower for w in ["céu", "noite", "estrela", "deus"]): vfx = "stars"
    elif any(w in text_lower for w in ["chuva", "chora", "triste", "lágrima"]): vfx = "rain"

    return Style(bg, fg, fg_dim, accent, font_choice, vfx)

def get_word_hierarchy(words):
    """ Retorna um conjunto de índices das palavras que devem receber destaque emocional """
    scored = []
    for i, w in enumerate(words):
        x = token(w["word"])
        score = EMOTIONAL.get(x, 0)
        if len(x) >= 6: score += 1
        if score > 0: scored.append((score, i))
    scored.sort(reverse=True)
    # Apenas a 1 palavra mais forte ganha destaque por cena para criar contraste real
    return {scored[0][1]} if scored else set()

# ==========================================
# RENDERIZAÇÃO GRÁFICA (The Editor)
# ==========================================
def draw_text_with_shadow(text, font, color, alpha=255, scale=1.0):
    """ Text rendering com Soft Drop Shadow garantindo legibilidade em qualquer fundo """
    b = font.getbbox(text); tw = b[2]-b[0]; th = b[3]-b[1]; pad = 30
    layer = Image.new("RGBA", (tw+pad*2, th+pad*2), (0,0,0,0))
    d = ImageDraw.Draw(layer)
    
    # Multipass soft shadow 
    shadow_col = (0, 0, 0, int(alpha * 0.4))
    for offset_x, offset_y, size in [(2,2,4), (0,4,8)]:
        for sw in range(size, 0, -2):
            d.text((pad+offset_x, pad+offset_y), text, font=font, fill=shadow_col, stroke_width=sw, stroke_fill=shadow_col)
            
    # Main text
    d.text((pad, pad), text, font=font, fill=(*color[:3], int(alpha)))
    
    if scale != 1.0:
        ns = (int(layer.width * scale), int(layer.height * scale))
        layer = layer.resize(ns, Image.Resampling.LANCZOS)
    return layer, tw, th

def pre_render_background(style, w_res, h_res):
    """ Renderiza um fundo 15% maior para permitir a câmera virtual (parallax/zoom) """
    bw, bh = int(w_res * 1.15), int(h_res * 1.15)
    arr = np.zeros((bh, bw, 3), np.float32)
    arr[:] = style.bg
    # Noise/Textura sutil
    rng = np.random.default_rng(42)
    arr += rng.normal(0, 2.5, (bh, bw, 1))
    
    bg = Image.fromarray(np.uint8(np.clip(arr, 0, 255))).convert("RGBA")
    draw = ImageDraw.Draw(bg)
    
    if style.vfx == "stars":
        for _ in range(120):
            x, y = rng.integers(0, bw), rng.integers(0, bh)
            size = rng.uniform(1.0, 3.5)
            draw.ellipse((x, y, x+size, y+size), fill=(255, 255, 255, int(rng.uniform(50, 180))))
    return bg

def fit_wrapped(words, font_path, maxw, avail_h, base_size):
    """ Calcula a quebra de linha sem desenhar para alinhar text block """
    size = base_size
    f = ImageFont.truetype(font_path, size)
    space = f.getlength(" ")
    
    rows = []; row = []; rw = 0
    for i, w in enumerate(words):
        ww = f.getlength(w["word"].upper())
        if row and rw + space + ww > maxw:
            rows.append((row, rw)); row = [(i, w, ww)]; rw = ww
        else:
            rw += ww + (space if row else 0); row.append((i, w, ww))
    if row: rows.append((row, rw))
    lh = int(f.size * 1.1)
    return rows, f, space, lh

def render_frame(scene, style, reg, base_bg, local, t, total_dur):
    # Câmera Virtual: Recorta o fundo progressivamente (Cinematic Slow Zoom)
    zoom_progress = clamp(t / total_dur, 0, 1)
    # Move a janela do crop levemente em direção ao centro
    cx = int(base_bg.width * 0.02 * zoom_progress)
    cy = int(base_bg.height * 0.02 * zoom_progress)
    bg_frame = base_bg.crop((cx, cy, cx + W, cy + H))
    
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    words = scene["words"]
    if not words: return bg_frame.convert("RGB")
    
    blue_set = get_word_hierarchy(words)
    font_path = reg.get(style.font) or next(iter(reg.values()))
    
    # Escala mestre (Clímax ganha fonte maior)
    base_size = int(H * 0.045)
    if scene.get("is_climax"): base_size = int(H * 0.055)
    elif len(words) > 4: base_size = int(H * 0.040)
    
    # Posição dinâmica orgânica (flutua na safe zone ao longo das cenas)
    safe_top = int(H * 0.28); avail_h = int(H * 0.40)
    y_wave = math.sin(scene.get("idx", 0) * 1.5) * (H * 0.06)
    if scene.get("is_climax"): y_wave = 0 # Clímax perfeitamente centralizado
    
    rows, font, space, lh = fit_wrapped(words, font_path, int(W * 0.75), avail_h, base_size)
    total_h = lh * len(rows)
    y = safe_top + (avail_h - total_h)/2 + y_wave
    
    for row, rw in rows:
        x = (W - rw)/2
        for i, w, ww in row:
            age = t - w["start"]
            duration_word = max(0.1, w["end"] - w["start"])
            
            is_accent = i in blue_set
            color = style.accent if is_accent else style.fg
            
            if age < 0:
                # Pre-reveal: Texto já na tela, guiando o olho, legibilidade total
                alpha = 35 
                scale = 1.0
                layer, lw, lh_b = draw_text_with_shadow(w["word"].upper(), font, style.fg, alpha, scale)
                rise = 0
            else:
                # Attack vocal (0.08s) + Sustentação musical (micro-growth scale)
                p_in = ease_out(age / 0.08)
                growth = clamp(age / (duration_word * 1.5), 0, 1.0) * 0.04 # Cresce max 4%
                
                scale = 1.0 + growth
                if is_accent: scale += 0.01 + (1 - p_in)*0.03 # Accent tem micro-pop extra no ataque
                
                alpha = 255 # Permanece forte após cantado para leitura do bloco
                layer, lw, lh_b = draw_text_with_shadow(w["word"].upper(), font, color, alpha, scale)
                
                # Displacement mínimo no ataque
                rise = (1 - p_in) * 12

            # Cola na tela matematicamente centralizado na sua box
            overlay.alpha_composite(layer, (int(x - (lw - ww)/2), int(y + rise - 30)))
            x += ww + space
        y += lh
        
    # Crossfade suave de fim de cena
    time_left = scene["end"] - t
    if time_left < 0.2:
        out_fade = ease_out(time_left / 0.2)
        overlay.putalpha(overlay.getchannel("A").point(lambda a: int(a * out_fade)))
        
    # Motor de Loop TikTok (Fade to Start)
    loop_time = total_dur - 1.5
    if t > loop_time:
        fade_loop = clamp((t - loop_time) / 1.5, 0, 1)
        bg_frame.putalpha(int(255 * (1 - fade_loop)))
        overlay.putalpha(overlay.getchannel("A").point(lambda a: int(a * (1 - fade_loop))))

    bg_frame.alpha_composite(overlay)
    return bg_frame.convert("RGB")

def render_video(audio_path, scenes, reg, out_path, res, progress):
    ww, hh = res
    global W, H; W, H = ww, hh
    dur = duration(audio_path)
    if not scenes: return 0
    end_time = min(dur, scenes[-1]["end"] + 0.8)
    
    # Pré-computa Direção (Clímax, Energia)
    apply_direction(scenes, end_time)
    
    # Cache do background da cena para não gerar ruído aleatório por frame (preserva estabilidade do codec H264)
    cached_bg = {}
    
    ff = ffmpeg(); silent = Path(out_path).with_name("silent.mp4")
    cmd = [ff, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{ww}x{hh}", "-r", str(FPS), "-i", "-",
           "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "15", "-pix_fmt", "yuv420p", "-profile:v", "high",
           "-movflags", "+faststart", str(silent)]
           
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    total_frames = max(1, int(math.ceil(end_time * FPS))); si = 0
    
    try:
        for fi in range(total_frames):
            t = fi / FPS
            while si + 1 < len(scenes) and t >= scenes[si]["end"]: si += 1
            scene = scenes[si]
            
            style = style_for(scene, si, reg)
            if si not in cached_bg:
                cached_bg[si] = pre_render_background(style, W, H)
                
            local = clamp(t - scene["start"], 0, scene["end"] - scene["start"])
            frame = np.asarray(render_frame(scene, style, reg, cached_bg[si], local, t, end_time), np.uint8)
            
            p.stdin.write(frame.tobytes())
            if progress and fi % (FPS*2) == 0: progress.progress(min(.95, fi / total_frames), text=f"Editando IA: {int(fi/total_frames*100)}%")
            
        p.stdin.close(); code = p.wait()
        if code: raise RuntimeError(p.stderr.read().decode("utf8", "replace")[-7000:])
    finally:
        if p.poll() is None:
            try: p.kill()
            except: pass
            
    run([ff, "-y", "-i", str(silent), "-i", audio_path, "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-t", f"{end_time:.3f}", "-movflags", "+faststart", str(out_path)], timeout=180)
    silent.unlink(missing_ok=True)
    if progress: progress.progress(1.0, text="✅ Vídeo Cinematográfico Concluído.")
    return end_time

st.set_page_config(page_title="Director's Lyric AI", page_icon="🎬", layout="centered")
st.title("🎬 Director's Lyric AI")
st.caption(f"Motor de Direção Automática de Retenção · {APP_VERSION}")

audio = st.file_uploader("🎵 Música (MP3, WAV, M4A)", type=["mp3", "wav", "m4a", "mp4"])
lyrics = st.text_area("📝 Letra de Referência (Opcional)", height=150, placeholder="Cole a letra para precisão perfeita. O sistema irá decupar a intenção musical automaticamente.")
res_opt = st.selectbox("Formato", ["TikTok / Reels (1080×1920)"], index=0)
reg = fonts()

if st.button("🚀 INICIAR DIREÇÃO AUTOMÁTICA", type="primary", use_container_width=True):
    if not audio: st.error("Música obrigatória."); st.stop()
    tmp = Path(tempfile.mkdtemp()); status = st.empty(); bar = st.progress(0)
    
    try:
        ap = tmp / safe(audio.name); ap.write_bytes(audio.getbuffer())
        
        # Reconhecimento e Decupagem
        asr, lang = transcribe(str(ap), "large-v3-turbo", status)
        if not asr: raise RuntimeError("Voz não detectada.")
        
        status.write("🧩 Direção de Arte em processamento...")
        if lyrics.strip():
            # Usa o plain block e divide logicamente
            scenes = build_plain(lyrics, asr, duration(str(ap)))
        else:
            # Sem letra: Usa ASR direto, o chunking de cena lidará com a organização
            scenes = subdivide_scenes([{"start": asr[0]["start"], "end": asr[-1]["end"], "words": asr}])
            
        if not scenes: raise RuntimeError("Erro no alinhamento. Tente novamente.")
        
        out = tmp / "lyric_director_final.mp4"
        final_end = render_video(str(ap), scenes, reg, str(out), (1080, 1920), bar)
        
        status.success(f"Direção Finalizada. Duração inteligente: {final_end:.2f}s")
        st.video(str(out))
        st.download_button("⬇️ Baixar MP4 (Pronto para TikTok)", out.read_bytes(), "lyric_viral_final.mp4", "video/mp4", use_container_width=True)
    except Exception as e:
        st.error("Erro na Edição Automática.")
        st.code(str(e))

