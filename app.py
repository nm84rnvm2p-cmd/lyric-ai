import os
import re
import math
import shutil
import subprocess
import tempfile
import urllib.request
import difflib
import unicodedata
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter


# ============================================================
# LYRIC AI STUDIO
# KINETIC WORD-SYNC ENGINE
# ============================================================

APP_VERSION = "7.0-ROYAL-WORDSYNC"

FPS = 30
DEFAULT_W = 1080
DEFAULT_H = 1920

CACHE_DIR = Path(".lyric_cache")
FONT_DIR = CACHE_DIR / "fonts"
FONT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FONTES
# ============================================================

FONT_SOURCES = {
    "Anton":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/anton/Anton-Regular.ttf",

    "Bebas Neue":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/bebasneue/BebasNeue-Regular.ttf",

    "Montserrat":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/montserrat/Montserrat%5Bwght%5D.ttf",

    "Oswald":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/oswald/Oswald%5Bwght%5D.ttf",

    "Archivo Black":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/archivoblack/ArchivoBlack-Regular.ttf",

    "DM Serif Display":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/dmserifdisplay/DMSerifDisplay-Regular.ttf",

    "Playfair Display":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/playfairdisplay/PlayfairDisplay%5Bwght%5D.ttf",

    "Libre Baskerville":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/librebaskerville/LibreBaskerville-Regular.ttf",

    "Space Mono":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/spacemono/SpaceMono-Regular.ttf",
}


SYSTEM_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
]


# ============================================================
# CORES — PRETO / BRANCO / AZUL ROYAL
# ============================================================

BLACK = (5, 5, 7)
BLACK_2 = (12, 12, 15)
WHITE = (248, 248, 246)
WHITE_SOFT = (222, 222, 220)

# Azul royal utilizado SOMENTE em palavras pontuais.
ROYAL_BLUE = (45, 92, 255)

TRANSPARENT = (0, 0, 0, 0)


# ============================================================
# UTILIDADES
# ============================================================

def clamp(v, a, b):
    return max(a, min(b, v))


def ease_out_cubic(t):
    t = clamp(t, 0.0, 1.0)
    return 1 - (1 - t) ** 3


def ease_in_out(t):
    t = clamp(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:100]


def normalize_text(s):
    return re.sub(r"\s+", " ", s or "").strip()


def normalize_token(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-zA-Z0-9]", "", s).lower()


def token_similarity(a, b):
    a = normalize_token(a)
    b = normalize_token(b)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    if a in b or b in a:
        return 0.90

    return difflib.SequenceMatcher(
        None,
        a,
        b,
        autojunk=False
    ).ratio()


def get_ffmpeg():
    try:
        import imageio_ffmpeg

        path = imageio_ffmpeg.get_ffmpeg_exe()

        if path and os.path.exists(path):
            return path

    except Exception:
        pass

    path = shutil.which("ffmpeg")

    if path:
        return path

    raise RuntimeError(
        "FFmpeg não encontrado. Verifique o requirements.txt."
    )


def run_cmd(cmd, timeout=None):
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout
    )

    if p.returncode != 0:
        raise RuntimeError(p.stderr[-6000:])

    return p.stdout


def media_duration(path):
    ff = get_ffmpeg()

    try:
        p = subprocess.run(
            [
                ff,
                "-hide_banner",
                "-i",
                path,
                "-f",
                "null",
                "-"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60
        )

        text = p.stderr + "\n" + p.stdout

        m = re.search(
            r"Duration:\s*(\d+):(\d+):([\d.]+)",
            text
        )

        if m:
            return (
                int(m.group(1)) * 3600 +
                int(m.group(2)) * 60 +
                float(m.group(3))
            )

    except Exception:
        pass

    return 0.0


# ============================================================
# FIM REAL DO ÁUDIO
# ============================================================

def detect_audio_end(path, duration):
    """
    Procura silêncio no final do áudio.

    Importante:
    Não usamos simplesmente a duração do arquivo.
    O lyric video termina próximo do último trecho vocal detectado.
    """

    ff = get_ffmpeg()

    try:

        cmd = [
            ff,
            "-hide_banner",
            "-i",
            path,
            "-af",
            "silencedetect=noise=-38dB:d=0.45",
            "-f",
            "null",
            "-"
        ]

        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=max(60, int(duration * 2))
        )

        text = p.stderr or ""

        starts = [
            float(x)
            for x in re.findall(
                r"silence_start:\s*([\d.]+)",
                text
            )
        ]

        ends = [
            float(x)
            for x in re.findall(
                r"silence_end:\s*([\d.]+)",
                text
            )
        ]

        if starts:

            last_start = starts[-1]

            # Só consideramos o último silêncio se ele realmente
            # estiver no final do áudio.
            if last_start > duration * 0.70:

                if ends and ends[-1] > last_start:
                    return min(
                        duration,
                        ends[-1] + 0.15
                    )

                return min(
                    duration,
                    last_start + 0.20
                )

    except Exception:
        pass

    return duration


# ============================================================
# FONTES
# ============================================================

def download_font(name):

    target = FONT_DIR / (
        safe_name(name) + ".ttf"
    )

    if target.exists() and target.stat().st_size > 10000:
        return str(target)

    url = FONT_SOURCES.get(name)

    if not url:
        return None

    try:

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "LyricAIStudio/7.0"
            }
        )

        with urllib.request.urlopen(
            req,
            timeout=30
        ) as response:

            data = response.read()

        target.write_bytes(data)

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

        path = download_font(name)

        if path:
            registry[name] = path

    if not registry:

        for path in SYSTEM_FONT_CANDIDATES:

            if os.path.exists(path):

                registry["System"] = path
                break

    return registry


def font_path(name, registry):

    if name in registry:
        return registry[name]

    for path in SYSTEM_FONT_CANDIDATES:

        if os.path.exists(path):
            return path

    raise RuntimeError(
        "Nenhuma fonte disponível."
    )


def fit_font(
    text,
    max_width,
    start_size,
    path,
    minimum=28
):

    size = int(start_size)

    while size >= minimum:

        font = ImageFont.truetype(
            path,
            size=size
        )

        box = font.getbbox(text)

        width = box[2] - box[0]

        if width <= max_width:
            return font

        size -= 2

    return ImageFont.truetype(
        path,
        size=minimum
    )


# ============================================================
# WHISPER
# ============================================================

@st.cache_resource(show_spinner=False)
def get_whisper(model_name):

    from faster_whisper import WhisperModel

    return WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
        cpu_threads=max(
            2,
            min(8, os.cpu_count() or 4)
        ),
        num_workers=1
    )


def transcribe_audio(
    path,
    model_name,
    status=None
):

    model = get_whisper(model_name)

    if status:
        status.write(
            f"🎙️ Analisando cada palavra com **{model_name}**..."
        )

    segments, info = model.transcribe(

        path,

        language="pt",

        task="transcribe",

        beam_size=8,

        best_of=8,

        patience=1.2,

        temperature=0.0,

        compression_ratio_threshold=2.4,

        log_prob_threshold=-1.0,

        no_speech_threshold=0.25,

        condition_on_previous_text=False,

        vad_filter=False,

        word_timestamps=True,

        initial_prompt=(
            "Letra de música brasileira cantada em português. "
            "Reconheça todas as palavras possíveis. "
            "Não resuma. "
            "Não traduza. "
            "Preserve repetições. "
            "Preserve contrações, gírias e nomes próprios."
        )
    )

    words = []

    for seg in segments:

        if not seg.words:
            continue

        for word in seg.words:

            txt = normalize_text(
                word.word
            )

            if not txt:
                continue

            words.append({

                "word": txt,

                "start": float(word.start),

                "end": float(word.end),

                "prob": float(
                    getattr(
                        word,
                        "probability",
                        0.0
                    ) or 0.0
                )
            })

    return (
        words,
        getattr(info, "language", "pt"),
        float(
            getattr(
                info,
                "duration",
                0.0
            ) or 0.0
        )
    )


# ============================================================
# LETRA COM TEMPOS
# ============================================================

def parse_timed_lyrics(text):
    """
    Formato aceito:

    00:12.30 | Eu sei que vou te amar
    00:16.80 | Por toda a minha vida

    Também aceita:

    12.30 | Eu sei que vou te amar
    """

    lines = []

    for raw in text.splitlines():

        raw = raw.strip()

        if not raw:
            continue

        match = re.match(
            r"^\s*(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?:[.,](\d{1,3}))?\s*\|\s*(.+)$",
            raw
        )

        if match:

            hours = (
                int(match.group(1))
                if match.group(1)
                else 0
            )

            minutes = int(match.group(2))
            seconds = int(match.group(3))

            fraction = match.group(4) or "0"

            fraction_value = float(
                "0." + fraction
            )

            total = (
                hours * 3600 +
                minutes * 60 +
                seconds +
                fraction_value
            )

            phrase = normalize_text(
                match.group(5)
            )

            if phrase:
                lines.append({
                    "start": total,
                    "text": phrase
                })

            continue

        # Também aceita timestamp simples:
        # 12.30 | texto

        match = re.match(
            r"^\s*(\d+(?:[.,]\d+)?)\s*\|\s*(.+)$",
            raw
        )

        if match:

            start = float(
                match.group(1).replace(",", ".")
            )

            phrase = normalize_text(
                match.group(2)
            )

            if phrase:
                lines.append({
                    "start": start,
                    "text": phrase
                })

    lines.sort(
        key=lambda x: x["start"]
    )

    return lines


def plain_lyrics(text):

    return [
        normalize_text(line)
        for line in text.splitlines()
        if normalize_text(line)
    ]


# ============================================================
# ALINHAMENTO DA LETRA
# ============================================================

def align_phrase_to_asr(
    phrase_text,
    phrase_start,
    phrase_end,
    asr_words
):

    tokens = re.findall(
        r"\S+",
        phrase_text
    )

    if not tokens:
        return []

    candidates = []

    for index, word in enumerate(asr_words):

        if word["end"] < phrase_start - 0.50:
            continue

        if word["start"] > phrase_end + 0.50:
            break

        candidates.append(
            (index, word)
        )

    result = []

    cursor = 0

    for token in tokens:

        best = None
        best_score = 0

        # Busca à frente.
        # A janela maior é importante para músicas rápidas.
        for j in range(
            cursor,
            min(len(candidates), cursor + 15)
        ):

            idx, word = candidates[j]

            score = token_similarity(
                token,
                word["word"]
            )

            # Pequeno bônus se o tempo estiver
            # dentro da frase.
            if (
                word["start"] >= phrase_start and
                word["start"] <= phrase_end
            ):
                score += 0.05

            if score > best_score:

                best_score = score
                best = (j, idx, word)

        if best and best_score >= 0.48:

            j, idx, word = best

            result.append({
                "word": token,
                "start": word["start"],
                "end": word["end"],
                "prob": max(
                    word.get("prob", 0.0),
                    best_score * 0.75
                )
            })

            cursor = j + 1

        else:

            result.append(None)

    # ========================================================
    # PALAVRAS NÃO RECONHECIDAS
    # Distribuição inteligente dentro da frase
    # ========================================================

    known = [
        i for i, x in enumerate(result)
        if x is not None
    ]

    for i, item in enumerate(result):

        if item is not None:
            continue

        previous = [
            k for k in known
            if k < i
        ]

        following = [
            k for k in known
            if k > i
        ]

        if previous:

            p = previous[-1]
            start = result[p]["end"]

        else:

            p = -1
            start = phrase_start

        if following:

            n = following[0]
            end = result[n]["start"]

        else:

            n = len(tokens)
            end = phrase_end

        gap = max(
            0.10,
            end - start
        )

        position = i - p
        total = max(
            1,
            n - p
        )

        word_start = (
            start +
            gap * (
                position / total
            )
        )

        word_end = (
            start +
            gap * (
                (position + 1) / total
            )
        )

        result[i] = {

            "word": tokens[i],

            "start": clamp(
                word_start,
                phrase_start,
                phrase_end
            ),

            "end": clamp(
                max(
                    word_start + 0.055,
                    word_end
                ),
                phrase_start + 0.055,
                phrase_end
            ),

            "prob": 0.45
        }

    return result


def build_timed_lyrics(
    timed_lines,
    asr_words,
    audio_end
):

    scenes = []

    for i, line in enumerate(timed_lines):

        start = float(
            line["start"]
        )

        if start >= audio_end:
            continue

        if i + 1 < len(timed_lines):

            next_start = float(
                timed_lines[i + 1]["start"]
            )

            end = min(
                next_start,
                audio_end
            )

        else:

            end = audio_end

        if end <= start:
            continue

        words = align_phrase_to_asr(
            line["text"],
            start,
            end,
            asr_words
        )

        # Remove palavras que caíram
        # claramente fora da frase.
        cleaned = []

        for w in words:

            w["start"] = clamp(
                w["start"],
                start,
                end
            )

            w["end"] = clamp(
                max(
                    w["start"] + 0.055,
                    w["end"]
                ),
                w["start"] + 0.055,
                end
            )

            cleaned.append(w)

        if cleaned:

            for w in cleaned:
                w["phrase_id"] = i
                w["phrase_text"] = line["text"]

            scenes.append({
                "start": start,
                "end": end,
                "words": cleaned,
                "phrase_text": line["text"],
                "instrumental": False
            })

    return scenes


# ============================================================
# LETRA SEM TEMPO
# ============================================================

def align_plain_lyrics(
    lyrics,
    asr_words,
    audio_end
):

    lines = plain_lyrics(
        lyrics
    )

    if not lines:
        return []

    scenes = []

    cursor = 0

    for phrase_id, text in enumerate(lines):

        tokens = re.findall(
            r"\S+",
            text
        )

        if not tokens:
            continue

        mapped = []

        for token in tokens:

            best = None
            best_score = 0

            for j in range(
                cursor,
                min(
                    len(asr_words),
                    cursor + 30
                )
            ):

                score = token_similarity(
                    token,
                    asr_words[j]["word"]
                )

                if score > best_score:

                    best_score = score
                    best = j

                if score >= 0.98:
                    break

            if (
                best is not None and
                best_score >= 0.45
            ):

                mapped.append(
                    (
                        token,
                        best,
                        best_score
                    )
                )

                cursor = best + 1

        if mapped:

            start = asr_words[
                mapped[0][1]
            ]["start"]

            end = asr_words[
                mapped[-1][1]
            ]["end"]

        else:

            if cursor < len(asr_words):

                start = asr_words[
                    cursor
                ]["start"]

            else:

                break

            duration_guess = max(
                0.8,
                0.32 * len(tokens)
            )

            end = min(
                audio_end,
                start + duration_guess
            )

        words = []

        known = {
            token: (
                asr_words[idx],
                score
            )
            for token, idx, score
            in mapped
        }

        for i, token in enumerate(tokens):

            if token in known:

                asr, score = known[token]

                words.append({
                    "word": token,
                    "start": asr["start"],
                    "end": asr["end"],
                    "prob": max(
                        asr.get("prob", 0.0),
                        score * 0.75
                    )
                })

            else:

                total = len(tokens)

                stime = (
                    start +
                    (end - start) *
                    i / total
                )

                etime = (
                    start +
                    (end - start) *
                    (i + 1) / total
                )

                words.append({
                    "word": token,
                    "start": stime,
                    "end": max(
                        stime + 0.055,
                        etime
                    ),
                    "prob": 0.45
                })

        for word in words:

            word["phrase_id"] = phrase_id
            word["phrase_text"] = text

        scenes.append({
            "start": max(
                0,
                start - 0.03
            ),

            "end": min(
                audio_end,
                end + 0.18
            ),

            "words": words,

            "phrase_text": text,

            "instrumental": False
        })

    return scenes


# ============================================================
# AUTO MODE
# ============================================================

def build_auto_scenes(
    asr_words,
    audio_end
):

    if not asr_words:
        return []

    scenes = []
    current = []

    for word in asr_words:

        if word["start"] >= audio_end:
            break

        if not current:

            current = [word]
            continue

        gap = (
            word["start"] -
            current[-1]["end"]
        )

        current_duration = (
            word["end"] -
            current[0]["start"]
        )

        punctuation = bool(
            re.search(
                r"[.!?;:]$",
                current[-1]["word"]
            )
        )

        # Importante:
        # não quebrar frases cedo demais.
        should_break = (
            gap > 0.62 or
            punctuation or
            len(current) >= 16 or
            current_duration > 7.5
        )

        if should_break:

            scenes.append(
                current
            )

            current = [word]

        else:

            current.append(
                word
            )

    if current:
        scenes.append(
            current
        )

    result = []

    for i, words in enumerate(scenes):

        start = words[0]["start"]
        end = min(
            words[-1]["end"] + 0.16,
            audio_end
        )

        result.append({
            "start": start,
            "end": end,
            "words": words,
            "phrase_text": " ".join(
                w["word"]
                for w in words
            ),
            "instrumental": False
        })

    return result


# ============================================================
# DETECÇÃO DE FRASES
# ============================================================

def trim_scenes_to_audio(
    scenes,
    audio_end
):

    output = []

    for scene in scenes:

        if scene["start"] >= audio_end:
            continue

        scene = dict(scene)

        scene["end"] = min(
            scene["end"],
            audio_end
        )

        valid_words = []

        for word in scene.get(
            "words",
            []
        ):

            if word["start"] < audio_end:

                word = dict(word)

                word["end"] = min(
                    word["end"],
                    audio_end
                )

                valid_words.append(
                    word
                )

        scene["words"] = valid_words

        if (
            scene["end"] >
            scene["start"] + 0.05
        ):

            output.append(
                scene
            )

    return output


# ============================================================
# DIREÇÃO VISUAL
# ============================================================

@dataclass
class Style:

    background: Tuple[int, int, int]

    foreground: Tuple[int, int, int]

    font: str

    layout: str

    blue_words: bool

    grain: float


SANS_FONTS = [
    "Anton",
    "Bebas Neue",
    "Montserrat",
    "Oswald",
    "Archivo Black"
]

SERIF_FONTS = [
    "DM Serif Display",
    "Playfair Display",
    "Libre Baskerville"
]


def choose_style(
    scene,
    registry,
    index
):

    phrase = scene.get(
        "phrase_text",
        ""
    )

    energy = len(
        scene.get(
            "words",
            []
        )
    )

    seed = (
        sum(
            ord(c)
            for c in phrase
        )
        + index * 31
    )

    # Fundo sempre monocromático.
    # Predominância de preto.
    background_options = [
        BLACK,
        BLACK,
        BLACK_2,
        (18, 18, 20),
        (245, 245, 243),
    ]

    background = background_options[
        seed % len(background_options)
    ]

    available_sans = [
        x for x in SANS_FONTS
        if x in registry
    ]

    available_serif = [
        x for x in SERIF_FONTS
        if x in registry
    ]

    available = (
        available_sans +
        available_serif
    )

    if not available:
        available = list(
            registry.keys()
        )

    # Frases mais fortes recebem tipografia
    # mais pesada.
    if energy <= 4:

        fonts = (
            available_sans
            or available
        )

    else:

        fonts = (
            available_sans
            or available
        )

    font = fonts[
        seed % len(fonts)
    ]

    # Layouts:
    # center = tradicional
    # stack = palavras empilhadas
    # editorial = composição assimétrica
    # hero = poucas palavras gigantes

    layouts = [
        "center",
        "editorial",
        "stack",
        "center",
        "editorial",
        "hero"
    ]

    layout = layouts[
        seed % len(layouts)
    ]

    # Azul não aparece sempre.
    use_blue = (
        seed % 4 == 0
    )

    return Style(
        background=background,
        foreground=(
            WHITE
            if sum(background) < 300
            else BLACK
        ),
        font=font,
        layout=layout,
        blue_words=use_blue,
        grain=0.008
    )


# ============================================================
# BACKGROUND
# ============================================================

def make_background(
    W,
    H,
    style,
    t
):

    base = np.zeros(
        (H, W, 3),
        dtype=np.float32
    )

    color = np.array(
        style.background,
        dtype=np.float32
    )

    base[:] = color

    yy, xx = np.mgrid[
        0:H,
        0:W
    ]

    # Detalhes monocromáticos.
    # Não usamos azul no fundo.
    if sum(style.background) < 300:

        center_x = W * (
            0.5 +
            0.08 *
            math.sin(t * 0.35)
        )

        center_y = H * (
            0.45 +
            0.08 *
            math.cos(t * 0.27)
        )

        dist = (
            ((xx - center_x) / (W * 0.65)) ** 2 +
            ((yy - center_y) / (H * 0.65)) ** 2
        )

        glow = np.exp(
            -2.2 * dist
        )[..., None]

        base += glow * 8

    else:

        center_x = W * (
            0.5 +
            0.05 *
            math.sin(t * 0.3)
        )

        center_y = H * 0.5

        dist = (
            ((xx - center_x) / (W * 0.7)) ** 2 +
            ((yy - center_y) / (H * 0.7)) ** 2
        )

        shade = np.exp(
            -2.0 * dist
        )[..., None]

        base -= shade * 8

    # Granulação extremamente discreta.
    rng = np.random.default_rng(
        int(t * 1000) % 999983
    )

    noise = rng.normal(
        0,
        255 * style.grain,
        (H, W, 1)
    )

    base += noise

    return Image.fromarray(
        np.uint8(
            np.clip(
                base,
                0,
                255
            )
        )
    )


# ============================================================
# ESCOLHA DE PALAVRAS AZUIS
# ============================================================

EMOTIONAL_WORDS = {
    "amor",
    "saudade",
    "coração",
    "coracao",
    "beijo",
    "beijos",
    "vida",
    "nunca",
    "sempre",
    "volta",
    "voltar",
    "embora",
    "paixão",
    "paixao",
    "desejo",
    "perfume",
    "chora",
    "chorar",
    "quero",
    "querer",
    "você",
    "voce",
    "meu",
    "minha",
    "tudo",
    "nada"
}


def choose_blue_indices(
    words,
    enabled,
    seed
):

    if not enabled:
        return set()

    if not words:
        return set()

    # Nunca pinta uma frase inteira de azul.
    # Normalmente 1 palavra.
    candidates = []

    for i, word in enumerate(words):

        clean = normalize_token(
            word["word"]
        )

        score = (
            len(clean) * 0.35
        )

        if clean in EMOTIONAL_WORDS:
            score += 4

        if len(clean) >= 8:
            score += 1.5

        candidates.append(
            (
                score,
                i
            )
        )

    candidates.sort(
        reverse=True
    )

    # Azul aparece de maneira ocasional.
    if seed % 3 != 0:
        return set()

    amount = 1

    if len(words) >= 11 and seed % 5 == 0:
        amount = 2

    return {
        i
        for _, i in candidates[:amount]
    }


# ============================================================
# TEXTO
# ============================================================

def text_size(
    draw,
    text,
    font
):

    box = draw.textbbox(
        (0, 0),
        text,
        font=font
    )

    return (
        box[2] - box[0],
        box[3] - box[1]
    )


def draw_word(
    draw,
    text,
    x,
    y,
    font,
    color,
    alpha,
    scale=1.0,
    shadow=True
):

    bbox = font.getbbox(text)

    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    pad = 35

    layer = Image.new(
        "RGBA",
        (
            tw + pad * 2,
            th + pad * 2
        ),
        (0, 0, 0, 0)
    )

    ld = ImageDraw.Draw(
        layer
    )

    if shadow:

        ld.text(
            (
                pad + 4,
                pad + 6
            ),
            text,
            font=font,
            fill=(0, 0, 0, 90)
        )

    ld.text(
        (
            pad,
            pad
        ),
        text,
        font=font,
        fill=(
            color[0],
            color[1],
            color[2],
            int(alpha)
        )
    )

    if scale != 1.0:

        layer = layer.resize(
            (
                max(
                    1,
                    int(
                        layer.width *
                        scale
                    )
                ),
                max(
                    1,
                    int(
                        layer.height *
                        scale
                    )
                )
            ),
            Image.Resampling.LANCZOS
        )

    return layer


# ============================================================
# LAYOUT — NORMAL
# ============================================================

def render_center_layout(
    overlay,
    words,
    font,
    W,
    H,
    local_time,
    scene_start,
    fg,
    blue_indices
):

    draw = ImageDraw.Draw(
        overlay
    )

    spoken = []

    for i, word in enumerate(words):

        relative = (
            word["start"] -
            scene_start
        )

        if local_time >= relative:

            spoken.append(
                (
                    i,
                    word,
                    relative
                )
            )

    if not spoken:
        return

    # ========================================================
    # QUEBRA POR LARGURA REAL
    # ========================================================

    max_width = int(
        W * 0.88
    )

    # Letras propositalmente maiores.
    base_size = int(
        H * 0.083
    )

    if len(spoken) >= 8:
        base_size = int(
            H * 0.072
        )

    if len(spoken) >= 12:
        base_size = int(
            H * 0.064
        )

    font = fit_font(
        " ".join(
            x[1]["word"].upper()
            for x in spoken
        ),
        max_width,
        base_size,
        font.path
        if hasattr(font, "path")
        else FONT_ACTIVE_PATH,
        minimum=32
    )

    rows = []

    current = []
    width = 0

    space_width = text_size(
        draw,
        " ",
        font
    )[0]

    for item in spoken:

        word = item[1]["word"].upper()

        ww = text_size(
            draw,
            word,
            font
        )[0]

        if (
            current and
            width +
            space_width +
            ww >
            max_width
        ):

            rows.append(
                (
                    current,
                    width
                )
            )

            current = [
                item
            ]

            width = ww

        else:

            if current:
                width += space_width

            current.append(
                item
            )

            width += ww

    if current:

        rows.append(
            (
                current,
                width
            )
        )

    line_height = int(
        font.size * 1.05
    )

    total_height = (
        len(rows) *
        line_height
    )

    start_y = (
        H / 2 -
        total_height / 2
    )

    for row_index, (
        row,
        row_width
    ) in enumerate(rows):

        x = (
            W -
            row_width
        ) / 2

        y = (
            start_y +
            row_index *
            line_height
        )

        for index, word, relative in row:

            token = word["word"].upper()

            ww = text_size(
                draw,
                token,
                font
            )[0]

            age = (
                local_time -
                relative
            )

            progress = clamp(
                age / 0.24,
                0,
                1
            )

            eased = ease_out_cubic(
                progress
            )

            alpha = int(
                255 * eased
            )

            rise = (
                20 *
                (1 - eased)
            )

            scale = (
                0.94 +
                0.06 * eased
            )

            color = (
                ROYAL_BLUE
                if index in blue_indices
                else fg
            )

            layer = draw_word(
                draw,
                token,
                x,
                y + rise,
                font,
                color,
                alpha,
                scale=scale
            )

            overlay.alpha_composite(
                layer,
                (
                    int(
                        x -
                        (
                            layer.width -
                            ww
                        ) / 2
                    ),
                    int(
                        y +
                        rise -
                        35
                    )
                )
            )

            x += (
                ww +
                space_width
            )


# ============================================================
# LAYOUT — STACK
# ============================================================

def render_stack_layout(
    overlay,
    words,
    font_path_value,
    W,
    H,
    local_time,
    scene_start,
    fg,
    blue_indices
):

    draw = ImageDraw.Draw(
        overlay
    )

    spoken = []

    for i, word in enumerate(words):

        relative = (
            word["start"] -
            scene_start
        )

        if local_time >= relative:

            spoken.append(
                (
                    i,
                    word,
                    relative
                )
            )

    if not spoken:
        return

    # ========================================================
    # FRASES GRANDES — PALAVRAS EMPILHADAS
    # ========================================================

    size = int(
        H * 0.075
    )

    font = fit_font(
        "AAAAAA",
        int(W * 0.70),
        size,
        font_path_value,
        minimum=34
    )

    max_visible = min(
        8,
        len(spoken)
    )

    visible = spoken[
        -max_visible:
    ]

    gap = int(
        font.size * 0.95
    )

    total = (
        len(visible) *
        gap
    )

    y = (
        H / 2 -
        total / 2
    )

    for index, word, relative in visible:

        token = word["word"].upper()

        age = (
            local_time -
            relative
        )

        p = clamp(
            age / 0.22,
            0,
            1
        )

        e = ease_out_cubic(
            p
        )

        alpha = int(
            255 * e
        )

        slide = (
            45 *
            (1 - e)
        )

        bbox = font.getbbox(
            token
        )

        width = (
            bbox[2] -
            bbox[0]
        )

        x = (
            W -
            width
        ) / 2

        color = (
            ROYAL_BLUE
            if index in blue_indices
            else fg
        )

        layer = draw_word(
            draw,
            token,
            x + slide,
            y,
            font,
            color,
            alpha,
            scale=0.96 + 0.04 * e
        )

        overlay.alpha_composite(
            layer,
            (
                int(
                    x +
                    slide -
                    35
                ),
                int(
                    y -
                    35
                )
            )
        )

        y += gap


# ============================================================
# LAYOUT — EDITORIAL
# ============================================================

def render_editorial_layout(
    overlay,
    words,
    font_path_value,
    W,
    H,
    local_time,
    scene_start,
    fg,
    blue_indices
):

    draw = ImageDraw.Draw(
        overlay
    )

    spoken = []

    for i, word in enumerate(words):

        relative = (
            word["start"] -
            scene_start
        )

        if local_time >= relative:

            spoken.append(
                (
                    i,
                    word,
                    relative
                )
            )

    if not spoken:
        return

    size = int(
        H * 0.078
    )

    font = fit_font(
        " ".join(
            x[1]["word"].upper()
            for x in spoken[-7:]
        ),
        int(W * 0.80),
        size,
        font_path_value,
        minimum=32
    )

    # ========================================================
    # Composição diagonal / assimétrica.
    # ========================================================

    anchor_y = H * 0.43

    for position, (
        index,
        word,
        relative
    ) in enumerate(
        spoken[-8:]
    ):

        token = word["word"].upper()

        age = (
            local_time -
            relative
        )

        p = clamp(
            age / 0.25,
            0,
            1
        )

        e = ease_out_cubic(
            p
        )

        alpha = int(
            255 * e
        )

        bbox = font.getbbox(
            token
        )

        width = (
            bbox[2] -
            bbox[0]
        )

        # Alternância de posição.
        if position % 2 == 0:

            x = W * 0.08

        else:

            x = W * 0.92 - width

        y = (
            anchor_y +
            (
                position -
                len(spoken[-8:]) / 2
            ) *
            font.size *
            0.82
        )

        y += (
            35 *
            (1 - e)
        )

        color = (
            ROYAL_BLUE
            if index in blue_indices
            else fg
        )

        layer = draw_word(
            draw,
            token,
            x,
            y,
            font,
            color,
            alpha,
            scale=0.95 + 0.05 * e
        )

        overlay.alpha_composite(
            layer,
            (
                int(
                    x -
                    35
                ),
                int(
                    y -
                    35
                )
            )
        )


# ============================================================
# RENDER DA CENA
# ============================================================

def render_scene(
    scene,
    style,
    registry,
    W,
    H,
    local_time,
    bg
):

    base = bg.convert(
        "RGBA"
    )

    overlay = Image.new(
        "RGBA",
        (W, H),
        (0, 0, 0, 0)
    )

    words = scene.get(
        "words",
        []
    )

    if not words:
        return base.convert(
            "RGB"
        )

    font_path_value = font_path(
        style.font,
        registry
    )

    global FONT_ACTIVE_PATH
    FONT_ACTIVE_PATH = font_path_value

    # ========================================================
    # Escolhe palavras azuis de maneira ocasional.
    # ========================================================

    seed = sum(
        ord(c)
        for c in scene.get(
            "phrase_text",
            ""
        )
    )

    blue_indices = choose_blue_indices(
        words,
        style.blue_words,
        seed
    )

    # ========================================================
    # Elementos decorativos discretos.
    # ========================================================

    draw = ImageDraw.Draw(
        overlay
    )

    pulse = (
        0.5 +
        0.5 *
        math.sin(
            local_time *
            1.7
        )
    )

    # Linha extremamente discreta.
    line_y = int(
        H * 0.82
    )

    draw.line(
        (
            W * 0.18,
            line_y,
            W * (
                0.82 +
                0.02 * pulse
            ),
            line_y
        ),
        fill=(
            style.foreground[0],
            style.foreground[1],
            style.foreground[2],
            35
        ),
        width=2
    )

    # Pequenos pontos monocromáticos.
    for k in range(3):

        angle = (
            local_time *
            (0.35 + k * 0.09)
            + seed * 0.01
            + k
        )

        x = W * (
            0.12 +
            0.76 *
            (
                math.sin(angle) + 1
            ) / 2
        )

        y = H * (
            0.18 +
            0.64 *
            (
                math.cos(angle * 1.15) + 1
            ) / 2
        )

        radius = int(
            W *
            0.003
        )

        draw.ellipse(
            (
                x - radius,
                y - radius,
                x + radius,
                y + radius
            ),
            fill=(
                style.foreground[0],
                style.foreground[1],
                style.foreground[2],
                28
            )
        )

    # ========================================================
    # Escolha do layout.
    # ========================================================

    if style.layout == "stack":

        render_stack_layout(
            overlay,
            words,
            font_path_value,
            W,
            H,
            local_time,
            scene["start"],
            style.foreground,
            blue_indices
        )

    elif style.layout == "editorial":

        render_editorial_layout(
            overlay,
            words,
            font_path_value,
            W,
            H,
            local_time,
            scene["start"],
            style.foreground,
            blue_indices
        )

    else:

        # hero e center usam composição central,
        # mas hero possui letras maiores.
        render_center_layout(
            overlay,
            words,
            ImageFont.truetype(
                font_path_value,
                size=int(H * 0.08)
            ),
            W,
            H,
            local_time,
            scene["start"],
            style.foreground,
            blue_indices
        )

    # ========================================================
    # Fade suave da composição.
    # ========================================================

    fade_duration = 0.22

    scene_duration = max(
        0.1,
        scene["end"] -
        scene["start"]
    )

    # Fade inicial muito sutil.
    if local_time < fade_duration:

        alpha = int(
            255 *
            ease_out_cubic(
                local_time /
                fade_duration
            )
        )

        overlay.putalpha(
            overlay.getchannel(
                "A"
            ).point(
                lambda x:
                int(
                    x *
                    alpha /
                    255
                )
            )
        )

    # Fade final.
    remaining = (
        scene_duration -
        local_time
    )

    if remaining < 0.20:

        factor = clamp(
            remaining / 0.20,
            0,
            1
        )

        overlay.putalpha(
            overlay.getchannel(
                "A"
            ).point(
                lambda x:
                int(
                    x *
                    factor
                )
            )
        )

    return Image.alpha_composite(
        base,
        overlay
    ).convert(
        "RGB"
    )


# ============================================================
# FUNDO DE VÍDEO OPCIONAL
# ============================================================

def video_info(path):

    import cv2

    cap = cv2.VideoCapture(
        path
    )

    if not cap.isOpened():
        return None

    fps = (
        cap.get(
            cv2.CAP_PROP_FPS
        )
        or 30
    )

    frames = (
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
        or 0
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
        or 0
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
        or 0
    )

    duration = (
        frames / fps
        if fps
        else 0
    )

    cap.release()

    return {
        "fps": fps,
        "frames": frames,
        "w": width,
        "h": height,
        "duration": duration
    }


def fit_crop_frame(
    frame,
    W,
    H
):

    import cv2

    if frame is None:
        return None

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    h, w = frame.shape[:2]

    target = W / H

    ratio = w / h

    if ratio > target:

        new_width = int(
            h * target
        )

        x = (
            w -
            new_width
        ) // 2

        frame = frame[
            :,
            x:x + new_width
        ]

    else:

        new_height = int(
            w / target
        )

        y = (
            h -
            new_height
        ) // 2

        frame = frame[
            y:y + new_height,
            :
        ]

    frame = cv2.resize(
        frame,
        (W, H),
        interpolation=cv2.INTER_LANCZOS4
    )

    return frame


# ============================================================
# RENDER VIDEO
# ============================================================

def render_video(
    audio_path,
    background_path,
    scenes,
    registry,
    output_path,
    resolution,
    quality,
    progress
):

    import cv2

    W, H = resolution

    audio_duration = media_duration(
        audio_path
    )

    # ========================================================
    # MUITO IMPORTANTE:
    # Não usamos simplesmente a duração do arquivo.
    # ========================================================

    audio_end = detect_audio_end(
        audio_path,
        audio_duration
    )

    if scenes:

        latest_word_end = max(
            (
                w["end"]
                for scene in scenes
                for w in scene.get(
                    "words",
                    []
                )
            ),
            default=0
        )

        # O vídeo não precisa continuar
        # depois do último vocal.
        if latest_word_end > 0:

            audio_end = min(
                audio_end,
                latest_word_end + 0.35
            )

    duration = max(
        0.5,
        audio_end
    )

    # ========================================================
    # BACKGROUND
    # ========================================================

    bgcap = None
    bg_static = None

    if background_path:

        info = video_info(
            background_path
        )

        if info:

            bgcap = cv2.VideoCapture(
                background_path
            )

        else:

            try:

                bg_static = np.asarray(
                    Image.open(
                        background_path
                    ).convert(
                        "RGB"
                    )
                )

            except Exception:

                bg_static = None

    # ========================================================
    # ENCODER
    # ========================================================

    ff = get_ffmpeg()

    silent = Path(
        output_path
    ).with_name(
        "silent_video.mp4"
    )

    if quality == "Alta qualidade":

        crf = "13"
        preset = "slow"

    else:

        crf = "16"
        preset = "medium"

    command = [
        ff,
        "-y",

        "-f",
        "rawvideo",

        "-vcodec",
        "rawvideo",

        "-pix_fmt",
        "rgb24",

        "-s",
        f"{W}x{H}",

        "-r",
        str(FPS),

        "-i",
        "-",

        "-an",

        "-c:v",
        "libx264",

        "-preset",
        preset,

        "-crf",
        crf,

        "-pix_fmt",
        "yuv420p",

        "-movflags",
        "+faststart",

        str(silent)
    ]

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    total_frames = max(
        1,
        int(
            math.ceil(
                duration *
                FPS
            )
        )
    )

    scene_index = 0

    try:

        for frame_index in range(
            total_frames
        ):

            t = (
                frame_index /
                FPS
            )

            # =================================================
            # CENA ATUAL
            # =================================================

            while (
                scene_index + 1 <
                len(scenes)
                and
                t >=
                scenes[
                    scene_index
                ]["end"]
            ):

                scene_index += 1

            if scenes:

                scene = scenes[
                    min(
                        scene_index,
                        len(scenes) - 1
                    )
                ]

            else:

                scene = {
                    "start": 0,
                    "end": duration,
                    "words": [],
                    "instrumental": True
                }

            # =================================================
            # BACKGROUND
            # =================================================

            bgframe = None

            if bgcap is not None:

                bgcap.set(
                    cv2.CAP_PROP_POS_MSEC,
                    t * 1000
                )

                ok, frame = (
                    bgcap.read()
                )

                if ok:

                    bgframe = fit_crop_frame(
                        frame,
                        W,
                        H
                    )

            elif bg_static is not None:

                bgframe = fit_crop_frame(
                    bg_static,
                    W,
                    H
                )

            if bgframe is not None:

                bg = Image.fromarray(
                    bgframe
                )

                # Monocromatização do fundo.
                gray = ImageEnhance.Color(
                    bg
                ).enhance(
                    0.0
                )

                gray = ImageEnhance.Contrast(
                    gray
                ).enhance(
                    1.10
                )

                bg = ImageEnhance.Brightness(
                    gray
                ).enhance(
                    0.70
                )

            else:

                style = choose_style(
                    scene,
                    registry,
                    scene_index
                )

                bg = make_background(
                    W,
                    H,
                    style,
                    t
                )

            style = choose_style(
                scene,
                registry,
                scene_index
            )

            local_time = clamp(
                t -
                scene["start"],
                0,
                max(
                    0.001,
                    scene["end"] -
                    scene["start"]
                )
            )

            final = render_scene(
                scene,
                style,
                registry,
                W,
                H,
                local_time,
                bg
            )

            # =================================================
            # TRANSIÇÃO FLUIDA
            # =================================================

            # Mistura curta entre cenas.
            # Nada de corte seco.
            transition_duration = 0.24

            if (
                scene_index > 0
                and
                0 <=
                t -
                scene["start"]
                <
                transition_duration
            ):

                previous = scenes[
                    scene_index - 1
                ]

                previous_style = choose_style(
                    previous,
                    registry,
                    scene_index - 1
                )

                previous_duration = max(
                    0.001,
                    previous["end"] -
                    previous["start"]
                )

                previous_bg = bg

                previous_frame = render_scene(
                    previous,
                    previous_style,
                    registry,
                    W,
                    H,
                    previous_duration,
                    previous_bg
                )

                transition = ease_in_out(
                    (
                        t -
                        scene["start"]
                    ) /
                    transition_duration
                )

                final = Image.blend(
                    previous_frame,
                    final,
                    transition
                )

            frame = np.asarray(
                final,
                dtype=np.uint8
            )

            process.stdin.write(
                frame.tobytes()
            )

            if progress and (
                frame_index %
                max(
                    1,
                    FPS
                ) == 0
            ):

                percent = (
                    frame_index /
                    total_frames
                )

                progress.progress(
                    min(
                        0.92,
                        percent * 0.92
                    ),
                    text=(
                        f"Renderizando "
                        f"{int(percent * 100)}%"
                    )
                )

        process.stdin.close()
        process.stdin = None

        stderr = (
            process.stderr
            .read()
            .decode(
                "utf-8",
                "replace"
            )
        )

        code = process.wait()

        if code != 0:

            raise RuntimeError(
                "FFmpeg falhou:\n" +
                stderr[-6000:]
            )

    except Exception:

        try:

            if process.stdin:
                process.stdin.close()

        except Exception:
            pass

        process.kill()

        raise

    finally:

        if bgcap:
            bgcap.release()

    # ========================================================
    # ÁUDIO ORIGINAL
    # ========================================================

    final_command = [

        ff,
        "-y",

        "-i",
        str(silent),

        "-i",
        audio_path,

        "-map",
        "0:v:0",

        "-map",
        "1:a:0",

        "-c:v",
        "copy",

        "-c:a",
        "aac",

        "-b:a",
        "256k",

        "-shortest",

        "-movflags",
        "+faststart",

        str(output_path)
    ]

    run_cmd(
        final_command,
        timeout=max(
            180,
            int(duration * 10)
        )
    )

    silent.unlink(
        missing_ok=True
    )

    if progress:

        progress.progress(
            1.0,
            text="✅ Vídeo concluído."
        )

    return duration


# ============================================================
# STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Lyric AI Studio",
    page_icon="🎵",
    layout="centered"
)

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1050px;
        padding-top: 1rem;
    }

    h1 {
        letter-spacing: -0.045em;
    }

    </style>
    """,
    unsafe_allow_html=True
)


st.title(
    "🎵 Lyric AI Studio"
)

st.caption(
    f"Kinetic Word-Sync Engine · {APP_VERSION}"
)


# ============================================================
# EXPLICAÇÃO
# ============================================================

with st.expander(
    "Como usar esta versão",
    expanded=False
):

    st.markdown(
        """
        **Para obter a melhor sincronização possível:**

        1. Envie a música.
        2. Cole a letra oficial.
        3. Se puder, informe o tempo de início de cada frase.
        4. A IA usa esses tempos como guia.
        5. O Whisper tenta identificar cada palavra dentro da frase.
        6. As palavras aparecem uma por uma.
        7. A frase permanece na tela enquanto está sendo cantada.
        8. O vídeo para próximo do último trecho realmente cantado.

        **Formato recomendado para letra com tempo:**

        `00:12.30 | Primeira frase da música`

        `00:16.80 | Segunda frase da música`

        `00:21.45 | Terceira frase da música`
        """
    )


# ============================================================
# UPLOADS
# ============================================================

audio_file = st.file_uploader(
    "1. Música ou vídeo com a música",
    type=[
        "mp3",
        "wav",
        "m4a",
        "mp4",
        "mov",
        "webm"
    ]
)


bg_file = st.file_uploader(
    "2. Fundo opcional",
    type=[
        "mp4",
        "mov",
        "webm",
        "jpg",
        "jpeg",
        "png"
    ]
)


lyrics = st.text_area(
    "3. Letra da música",
    height=180,
    placeholder=(
        "Cole a letra aqui.\n\n"
        "Se tiver os tempos das frases, use:\n\n"
        "00:12.30 | Eu sei que vou te amar\n"
        "00:16.80 | Por toda a minha vida\n"
        "00:21.40 | Eu vou te amar"
    )
)


st.caption(
    "💡 Os tempos das frases são opcionais, mas altamente recomendados."
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

col1, col2 = st.columns(2)


with col1:

    model = st.selectbox(
        "Modelo de reconhecimento",
        [
            "small",
            "medium",
            "large-v3-turbo",
            "large-v3"
        ],
        index=2
    )


with col2:

    quality = st.selectbox(
        "Qualidade do vídeo",
        [
            "Equilibrado",
            "Alta qualidade"
        ],
        index=1
    )


col3, col4 = st.columns(2)


with col3:

    resolution = st.selectbox(
        "Resolução",
        [
            "1080×1920",
            "720×1280"
        ],
        index=0
    )


with col4:

    style_mode = st.selectbox(
        "Estilo",
        [
            "IA — Preto/Branco/Royal",
            "Mais minimalista",
            "Mais dinâmico"
        ],
        index=0
    )


# ============================================================
# FONTES
# ============================================================

registry = load_font_registry()

st.caption(
    f"Fontes disponíveis: {len(registry)}"
)


# ============================================================
# BOTÃO
# ============================================================

if st.button(
    "🚀 CRIAR LYRIC VIDEO",
    type="primary",
    use_container_width=True
):

    if not audio_file:

        st.error(
            "Envie a música primeiro."
        )

        st.stop()

    tmpdir = Path(
        tempfile.mkdtemp(
            prefix="lyric_ai_"
        )
    )

    try:

        # ====================================================
        # ARQUIVOS
        # ====================================================

        audio_path = (
            tmpdir /
            safe_name(
                audio_file.name
            )
        )

        audio_path.write_bytes(
            audio_file.getbuffer()
        )

        bg_path = None

        if bg_file:

            bg_path = (
                tmpdir /
                safe_name(
                    bg_file.name
                )
            )

            bg_path.write_bytes(
                bg_file.getbuffer()
            )

        status = st.empty()

        progress = st.progress(
            0.0
        )

        # ====================================================
        # DURAÇÃO
        # ====================================================

        status.write(
            "⏱️ Analisando duração do áudio..."
        )

        duration = media_duration(
            str(audio_path)
        )

        if duration <= 0:

            raise RuntimeError(
                "Não foi possível determinar a duração do áudio."
            )

        audio_end = detect_audio_end(
            str(audio_path),
            duration
        )

        # ====================================================
        # TRANSCRIÇÃO
        # ====================================================

        try:

            asr_words, language, detected_duration = (
                transcribe_audio(
                    str(audio_path),
                    model,
                    status
                )
            )

        except Exception:

            if model != "small":

                status.warning(
                    "O modelo escolhido não iniciou. "
                    "Tentando automaticamente o modelo small."
                )

                asr_words, language, detected_duration = (
                    transcribe_audio(
                        str(audio_path),
                        "small",
                        status
                    )
                )

            else:

                raise

        # ====================================================
        # REMOVE PALAVRAS FORA DO FINAL REAL
        # ====================================================

        asr_words = [
            w
            for w in asr_words
            if w["start"] < audio_end
        ]

        if not asr_words:

            raise RuntimeError(
                "Nenhuma palavra foi reconhecida."
            )

        progress.progress(
            0.20,
            text="Reconhecimento concluído."
        )

        # ====================================================
        # IDENTIFICA SE A LETRA TEM TEMPOS
        # ====================================================

        timed_lines = parse_timed_lyrics(
            lyrics
        ) if lyrics.strip() else []

        # ====================================================
        # CONSTRUÇÃO DAS CENAS
        # ====================================================

        if timed_lines:

            status.write(
                "🎯 Usando os tempos fornecidos para sincronizar cada frase..."
            )

            scenes = build_timed_lyrics(
                timed_lines,
                asr_words,
                audio_end
            )

        elif lyrics.strip():

            status.write(
                "🎯 Alinhando a letra oficial ao canto..."
            )

            scenes = align_plain_lyrics(
                lyrics,
                asr_words,
                audio_end
            )

        else:

            status.write(
                "🎙️ Criando frases automaticamente..."
            )

            scenes = build_auto_scenes(
                asr_words,
                audio_end
            )

        # ====================================================
        # GARANTIA DE FINAL
        # ====================================================

        scenes = trim_scenes_to_audio(
            scenes,
            audio_end
        )

        if not scenes:

            raise RuntimeError(
                "Não foi possível criar as frases."
            )

        # ====================================================
        # RESOLUÇÃO
        # ====================================================

        if resolution.startswith(
            "1080"
        ):

            render_resolution = (
                1080,
                1920
            )

        else:

            render_resolution = (
                720,
                1280
            )

        progress.progress(
            0.30,
            text="Direção visual preparada."
        )

        # ====================================================
        # SAÍDA
        # ====================================================

        output_path = (
            tmpdir /
            "lyric_ai_final.mp4"
        )

        status.write(
            "🎬 Renderizando tipografia cinética..."
        )

        final_duration = render_video(
            str(audio_path),
            str(bg_path)
            if bg_path
            else None,
            scenes,
            registry,
            str(output_path),
            render_resolution,
            quality,
            progress
        )

        # ====================================================
        # RESULTADO
        # ====================================================

        status.success(
            "🎉 Vídeo criado com sucesso."
        )

        st.video(
            str(output_path)
        )

        st.download_button(
            "⬇️ BAIXAR LYRIC VIDEO",
            data=output_path.read_bytes(),
            file_name="lyric_ai_final.mp4",
            mime="video/mp4",
            use_container_width=True
        )

        # ====================================================
        # DIAGNÓSTICO
        # ====================================================

        with st.expander(
            "🔎 Diagnóstico da IA"
        ):

            all_words = [
                word
                for scene in scenes
                for word in scene.get(
                    "words",
                    []
                )
            ]

            confidence = (
                np.mean(
                    [
                        w.get(
                            "prob",
                            0
                        )
                        for w in all_words
                    ]
                )
                if all_words
                else 0
            )

            st.write(
                f"**Palavras sincronizadas:** "
                f"{len(all_words)}"
            )

            st.write(
                f"**Frases:** "
                f"{len(scenes)}"
            )

            st.write(
                f"**Confiança média:** "
                f"{confidence:.2f}"
            )

            st.write(
                f"**Idioma:** "
                f"{language}"
            )

            st.write(
                f"**Duração original:** "
                f"{duration:.2f}s"
            )

            st.write(
                f"**Final detectado:** "
                f"{audio_end:.2f}s"
            )

            st.write(
                f"**Duração final do vídeo:** "
                f"{final_duration:.2f}s"
            )

            st.write(
                "**Letra sincronizada:**"
            )

            st.code(
                "\n".join(
                    f"{scene['start']:.2f}s → "
                    f"{scene['end']:.2f}s | "
                    f"{scene.get('phrase_text', '')}"
                    for scene in scenes
                )
            )

    except Exception as error:

        st.error(
            "❌ A geração falhou."
        )

        st.code(
            str(error)
        )

        st.info(
            "Se o erro mencionar memória, "
            "troque o modelo para small ou medium."
        )

    finally:

        # Os arquivos permanecem disponíveis
        # enquanto o Streamlit precisar deles.
        pass