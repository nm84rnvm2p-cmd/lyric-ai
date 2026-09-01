# -*- coding: utf-8 -*-

import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
import math
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# CONFIGURAÇÃO
# ============================================================

APP_VERSION = "19.0-FADE-IN-PHRASE"

W = 720
H = 1280
FPS = 20

# ============================================================
# FADE DE ENTRADA DAS PALAVRAS
#
# Cada palavra começa no seu timestamp e faz fade in.
# Depois de aparecer, permanece na tela até o FINAL DA FRASE.
# ============================================================

WORD_FADE = 0.12

# Transição entre fundos
BACKGROUND_FADE = 0.20

# Cor azul
ROYAL = (45, 92, 255)

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


# ============================================================
# FONTES
# ============================================================

FONT_CANDIDATES = [

    (
        "Grossa",
        "/usr/share/fonts/truetype/dejavu/"
        "DejaVuSansCondensed-Bold.ttf",
    ),

    (
        "Sans",
        "/usr/share/fonts/truetype/dejavu/"
        "DejaVuSans-Bold.ttf",
    ),

    (
        "Serif",
        "/usr/share/fonts/truetype/dejavu/"
        "DejaVuSerif-Bold.ttf",
    ),

    (
        "Liberation",
        "/usr/share/fonts/truetype/liberation2/"
        "LiberationSans-Bold.ttf",
    ),

    (
        "FreeSans",
        "/usr/share/fonts/truetype/freefont/"
        "FreeSansBold.ttf",
    ),

    (
        "Arimo",
        "/usr/share/fonts/truetype/croscore/"
        "Arimo-Bold.ttf",
    ),

    (
        "Carlito",
        "/usr/share/fonts/truetype/crosextra/"
        "Carlito-Bold.ttf",
    ),

    (
        "Tinos",
        "/usr/share/fonts/truetype/croscore/"
        "Tinos-Bold.ttf",
    ),

]


# ============================================================
# TEXTO / UTF-8
# ============================================================

def clean_text(value):

    if value is None:
        return ""

    text = unicodedata.normalize(
        "NFC",
        str(value),
    )

    text = (
        text
        .replace("\ufeff", "")
        .replace("\ufffd", "")
    )

    text = "".join(
        ch
        for ch in text
        if ch in "\n\t"
        or unicodedata.category(ch)
        not in {"Cc", "Cf"}
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def normalize_for_match(text):

    text = clean_text(
        text
    ).lower()

    text = unicodedata.normalize(
        "NFKD",
        text,
    )

    text = "".join(
        c
        for c in text
        if not unicodedata.combining(c)
    )

    return text


def safe_filename(name):

    return (
        re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            name or "audio",
        )[:100]
        or "audio"
    )


# ============================================================
# FFMPEG
# ============================================================

def get_ffmpeg():

    try:

        import imageio_ffmpeg

        path = (
            imageio_ffmpeg
            .get_ffmpeg_exe()
        )

        if (
            path
            and os.path.isfile(path)
        ):
            return path

    except Exception:
        pass

    path = shutil.which(
        "ffmpeg"
    )

    if path:
        return path

    raise RuntimeError(
        "FFmpeg não foi encontrado."
    )


def run_cmd(
    cmd,
    timeout=300,
):

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )

    if result.returncode != 0:

        raise RuntimeError(
            result.stderr[-7000:]
            or
            "O comando terminou com erro."
        )

    return (
        result.stdout,
        result.stderr,
    )


def media_info(path):

    _, stderr = run_cmd(
        [
            get_ffmpeg(),
            "-hide_banner",
            "-i",
            str(path),
            "-f",
            "null",
            "-",
        ],
        timeout=90,
    )

    match = re.search(
        r"Duration:\s*(\d+):(\d+):([\d.]+)",
        stderr,
    )

    if not match:

        raise RuntimeError(
            "Não foi possível determinar "
            "a duração do arquivo."
        )

    duration = (
        int(match.group(1)) * 3600
        + int(match.group(2)) * 60
        + float(match.group(3))
    )

    has_audio = bool(
        re.search(
            r"Stream #.*Audio:",
            stderr,
        )
    )

    return duration, has_audio


# ============================================================
# ESTIMATIVA LEVE DE BPM
# ============================================================

def estimate_bpm(audio_path):

    """
    Estimativa leve de BPM.

    Não utiliza librosa.
    Não adiciona uma dependência pesada.

    Analisa somente os primeiros segundos
    da música e usa autocorrelação da energia.
    """

    try:

        import numpy as np

        ff = get_ffmpeg()

        command = [
            ff,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(audio_path),
            "-t",
            "18",
            "-ac",
            "1",
            "-ar",
            "8000",
            "-f",
            "s16le",
            "-",
            "pipe:1",
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=45,
        )

        if (
            result.returncode != 0
            or not result.stdout
        ):
            return 100.0

        audio = np.frombuffer(
            result.stdout,
            dtype=np.int16,
        ).astype(
            np.float32
        )

        if len(audio) < 8000:
            return 100.0

        audio /= 32768.0

        block = 400

        count = (
            len(audio)
            // block
        )

        if count < 30:
            return 100.0

        envelope = np.zeros(
            count,
            dtype=np.float32,
        )

        for i in range(count):

            section = audio[
                i * block:
                (i + 1) * block
            ]

            envelope[i] = np.sqrt(
                np.mean(
                    section * section
                )
                + 1e-9
            )

        envelope -= np.mean(
            envelope
        )

        std = np.std(
            envelope
        )

        if std < 1e-6:
            return 100.0

        envelope /= std

        sample_rate = (
            8000 / block
        )

        min_bpm = 60
        max_bpm = 180

        min_lag = int(
            sample_rate
            * 60
            / max_bpm
        )

        max_lag = int(
            sample_rate
            * 60
            / min_bpm
        )

        if max_lag >= len(envelope):
            max_lag = (
                len(envelope) - 1
            )

        best_lag = None
        best_score = -1e9

        for lag in range(
            min_lag,
            max_lag + 1,
        ):

            a = envelope[
                :-lag
            ]

            b = envelope[
                lag:
            ]

            if len(a) < 10:
                continue

            score = float(
                np.mean(a * b)
            )

            if score > best_score:

                best_score = score
                best_lag = lag

        if not best_lag:

            return 100.0

        bpm = (
            60
            * sample_rate
            / best_lag
        )

        while bpm < 75:
            bpm *= 2

        while bpm > 160:
            bpm /= 2

        return float(
            max(
                70,
                min(
                    160,
                    bpm,
                ),
            )
        )

    except Exception:

        return 100.0


# ============================================================
# INTRO
# ============================================================

def calculate_intro_duration(
    bpm,
    first_lyric_start,
):

    """
    Intro baseada em 2 batidas.

    Música rápida:
    intro menor.

    Música lenta:
    intro maior.

    Quando existe timestamp da primeira
    frase, a intro tenta terminar antes dela.
    """

    bpm = max(
        70.0,
        min(
            160.0,
            float(bpm),
        ),
    )

    beat = (
        60.0 / bpm
    )

    desired = (
        beat * 2.0
    )

    desired = max(
        0.85,
        min(
            2.40,
            desired,
        ),
    )

    if (
        first_lyric_start is not None
        and first_lyric_start > 0.65
    ):

        desired = min(
            desired,
            max(
                0.55,
                first_lyric_start - 0.05,
            ),
        )

    return desired


def smoothstep(value):

    value = max(
        0.0,
        min(
            1.0,
            value,
        ),
    )

    return (
        value
        * value
        * (
            3
            - 2 * value
        )
    )


@st.cache_data(
    show_spinner=False
)
def load_logo_from_file():

    candidates = [

        Path(
            "logo_perfil.png"
        ),

        Path(
            "logo.png"
        ),

    ]

    for path in candidates:

        if path.exists():

            try:

                image = Image.open(
                    path
                ).convert(
                    "RGBA"
                )

                return image

            except Exception:
                pass

    return None


def resize_logo(
    logo,
    max_width=460,
    max_height=380,
):

    if logo is None:
        return None

    image = logo.copy()

    ratio = min(
        max_width
        / image.width,
        max_height
        / image.height,
        1.0,
    )

    if ratio < 1:

        image = image.resize(
            (
                int(
                    image.width
                    * ratio
                ),
                int(
                    image.height
                    * ratio
                ),
            ),
            Image.Resampling.LANCZOS,
        )

    return image


def draw_intro_frame(
    logo,
    time_position,
    intro_duration,
):

    image = Image.new(
        "RGBA",
        (W, H),
        BLACK + (255,),
    )

    draw = ImageDraw.Draw(
        image
    )

    if intro_duration <= 0:

        return image.convert(
            "RGB"
        )

    progress = (
        time_position
        / intro_duration
    )

    progress = max(
        0.0,
        min(
            1.0,
            progress,
        ),
    )

    # --------------------------------------------------------
    # LOGO
    # --------------------------------------------------------

    logo = resize_logo(
        logo
    )

    if logo is not None:

        logo_end = 0.42

        if progress < logo_end:

            logo_progress = (
                progress
                / logo_end
            )

            logo_alpha = int(
                255
                * (
                    1
                    - smoothstep(
                        logo_progress
                    )
                    * 0.85
                )
            )

            logo_layer = (
                logo.copy()
            )

            if logo_layer.mode != "RGBA":

                logo_layer = (
                    logo_layer.convert(
                        "RGBA"
                    )
                )

            alpha = (
                logo_layer
                .getchannel(
                    "A"
                )
            )

            alpha = alpha.point(
                lambda p:
                int(
                    p
                    * logo_alpha
                    / 255
                )
            )

            logo_layer.putalpha(
                alpha
            )

            x = (
                W
                - logo_layer.width
            ) // 2

            y = (
                H
                - logo_layer.height
            ) // 2

            image.alpha_composite(
                logo_layer,
                (
                    x,
                    y,
                ),
            )

    # --------------------------------------------------------
    # CÍRCULO BRANCO
    # --------------------------------------------------------

    circle_start = 0.25

    if progress >= circle_start:

        circle_progress = (
            progress
            - circle_start
        ) / (
            1.0
            - circle_start
        )

        circle_progress = smoothstep(
            circle_progress
        )

        max_radius = math.sqrt(
            (
                W / 2
            ) ** 2
            + (
                H / 2
            ) ** 2
        )

        radius = (
            max_radius
            * circle_progress
        )

        cx = W // 2
        cy = H // 2

        draw.ellipse(
            (
                int(
                    cx - radius
                ),
                int(
                    cy - radius
                ),
                int(
                    cx + radius
                ),
                int(
                    cy + radius
                ),
            ),
            fill=WHITE + (255,),
        )

    return image.convert(
        "RGB"
    )


# ============================================================
# WHISPER
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def load_whisper(
    model_name
):

    from faster_whisper import (
        WhisperModel
    )

    return WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
        cpu_threads=2,
        num_workers=1,
    )


def transcribe(
    path,
    model_name,
    status,
):

    model = load_whisper(
        model_name
    )

    status.write(
        f"Transcrevendo com "
        f"**{model_name}**..."
    )

    segments_iter, info = (
        model.transcribe(
            str(path),
            language="pt",
            task="transcribe",
            word_timestamps=True,
            beam_size=3,
            best_of=3,
            temperature=0.0,
            condition_on_previous_text=True,
            vad_filter=False,
            initial_prompt=(
                "Letra de música brasileira "
                "em português. "
                "Reconheça todas as palavras, "
                "repetições, gírias e acentos. "
                "Não traduza."
            ),
        )
    )

    segments = []
    words = []

    for segment in segments_iter:

        start = float(
            segment.start
        )

        end = float(
            segment.end
        )

        text = clean_text(
            segment.text
        )

        if (
            text
            and end > start
        ):

            segments.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                }
            )

        for word in (
            segment.words
            or []
        ):

            word_text = clean_text(
                word.word
            )

            if not word_text:
                continue

            word_start = float(
                word.start
            )

            word_end = float(
                word.end
            )

            if (
                word_end
                <= word_start
            ):
                continue

            words.append(
                {
                    "word": word_text,
                    "start": word_start,
                    "end": word_end,
                }
            )

    return (
        segments,
        words,
        getattr(
            info,
            "language",
            "pt",
        )
        or "pt",
    )


# ============================================================
# FALLBACK DE PALAVRAS
# ============================================================

def distribute_words(
    text,
    start,
    end,
):

    tokens = re.findall(
        r"\S+",
        clean_text(text),
    )

    if not tokens:
        return []

    duration = (
        end - start
    )

    if duration <= 0:
        return []

    step = (
        duration
        / len(tokens)
    )

    result = []

    for i, token in enumerate(
        tokens
    ):

        word_start = (
            start
            + i * step
        )

        if (
            i
            == len(tokens) - 1
        ):

            word_end = end

        else:

            word_end = (
                start
                + (
                    i + 1
                )
                * step
            )

        result.append(
            {
                "word": token,
                "start": word_start,
                "end": max(
                    word_start + 0.06,
                    word_end,
                ),
            }
        )

    return result


# ============================================================
# SIMILARIDADE
# ============================================================

def similarity(
    a,
    b,
):

    import difflib

    aa = normalize_for_match(
        a
    )

    bb = normalize_for_match(
        b
    )

    if not aa or not bb:
        return 0.0

    if aa == bb:
        return 1.0

    return difflib.SequenceMatcher(
        None,
        aa,
        bb,
        autojunk=False,
    ).ratio()


# ============================================================
# ALINHAMENTO
# ============================================================

def align_phrase(
    text,
    start,
    end,
    asr_words,
):

    tokens = re.findall(
        r"\S+",
        clean_text(text),
    )

    if not tokens:
        return []

    candidates = [
        word
        for word in asr_words
        if (
            word["end"]
            > start - 0.20
        )
        and (
            word["start"]
            < end + 0.20
        )
    ]

    result = [
        None
        for _ in tokens
    ]

    used = set()

    for i, token in enumerate(
        tokens
    ):

        best_index = None
        best_score = 0.0

        for j, candidate in enumerate(
            candidates
        ):

            if j in used:
                continue

            score = similarity(
                token,
                candidate["word"],
            )

            if score > best_score:

                best_score = score
                best_index = j

        if (
            best_index is not None
            and best_score >= 0.42
        ):

            candidate = candidates[
                best_index
            ]

            used.add(
                best_index
            )

            result[i] = {
                "word": token,
                "start": max(
                    start,
                    min(
                        end,
                        candidate["start"],
                    ),
                ),
                "end": max(
                    start,
                    min(
                        end,
                        candidate["end"],
                    ),
                ),
            }

    known = [
        i
        for i, item in enumerate(
            result
        )
        if item is not None
    ]

    for i in range(
        len(tokens)
    ):

        if result[i] is not None:
            continue

        previous = max(
            (
                k
                for k in known
                if k < i
            ),
            default=-1,
        )

        following = min(
            (
                k
                for k in known
                if k > i
            ),
            default=len(tokens),
        )

        left = (
            result[
                previous
            ]["end"]
            if previous >= 0
            else start
        )

        right = (
            result[
                following
            ]["start"]
            if following < len(tokens)
            else end
        )

        count = max(
            1,
            following - previous,
        )

        word_start = (
            left
            + (
                right - left
            )
            * (
                (
                    i
                    - previous
                    - 1
                )
                / count
            )
        )

        word_end = (
            left
            + (
                right - left
            )
            * (
                (
                    i
                    - previous
                )
                / count
            )
        )

        word_start = max(
            start,
            min(
                end,
                word_start,
            ),
        )

        word_end = max(
            word_start + 0.06,
            min(
                end,
                word_end,
            ),
        )

        result[i] = {
            "word": tokens[i],
            "start": word_start,
            "end": word_end,
        }

    return result


# ============================================================
# TIMESTAMPS
# ============================================================

def parse_time(
    value
):

    value = (
        value
        .strip()
        .replace(",", ".")
    )

    if ":" not in value:

        return float(value)

    parts = value.split(":")

    if len(parts) == 2:

        return (
            int(parts[0]) * 60
            + float(parts[1])
        )

    return (
        int(parts[-3]) * 3600
        + int(parts[-2]) * 60
        + float(parts[-1])
    )


def parse_timed_lyrics(
    text
):

    lines = (
        text or ""
    ).replace(
        "\r",
        "",
    ).split("\n")

    result = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        match = re.match(
            r"^"
            r"(\d{1,2}:\d{2}(?:[.,]\d{1,3})?"
            r"|\d+(?:[.,]\d+)?)"
            r"\s*[-–—]\s*"
            r"(\d{1,2}:\d{2}(?:[.,]\d{1,3})?"
            r"|\d+(?:[.,]\d+)?)"
            r"\s*(?:\|\s*)?"
            r"(.*)$",
            line,
        )

        if match:

            start = parse_time(
                match.group(1)
            )

            end = parse_time(
                match.group(2)
            )

            phrase = clean_text(
                match.group(3)
            )

            if (
                phrase
                and end > start
            ):

                result.append(
                    {
                        "start": start,
                        "end": end,
                        "text": phrase,
                    }
                )

            continue

        match = re.match(
            r"^"
            r"(\d{1,2}:\d{2}(?:[.,]\d{1,3})?"
            r"|\d+(?:[.,]\d+)?)"
            r"\s*\|\s*"
            r"(.+)$",
            line,
        )

        if match:

            result.append(
                {
                    "start": parse_time(
                        match.group(1)
                    ),
                    "end": None,
                    "text": clean_text(
                        match.group(2)
                    ),
                }
            )

    result.sort(
        key=lambda x:
        x["start"]
    )

    return result


# ============================================================
# CONSTRUÇÃO DAS CENAS
# ============================================================

def build_scenes(
    timed,
    segments,
    asr_words,
    sung_end,
):

    scenes = []

    if timed:

        for i, line in enumerate(
            timed
        ):

            start = max(
                0.0,
                float(
                    line["start"]
                ),
            )

            if start >= sung_end:
                break

            if (
                line.get("end")
                is not None
            ):

                end = min(
                    float(
                        line["end"]
                    ),
                    sung_end,
                )

            else:

                if (
                    i + 1
                    < len(timed)
                ):

                    next_start = (
                        timed[
                            i + 1
                        ]["start"]
                    )

                else:

                    next_start = sung_end

                end = min(
                    float(next_start),
                    sung_end,
                )

            if (
                end
                <= start + 0.05
            ):
                continue

            words = align_phrase(
                line["text"],
                start,
                end,
                asr_words,
            )

            if not words:

                words = distribute_words(
                    line["text"],
                    start,
                    end,
                )

            if words:

                scenes.append(
                    {
                        "start": start,
                        "end": end,
                        "text": clean_text(
                            line["text"]
                        ),
                        "words": words,
                    }
                )

        return scenes

    for segment in segments:

        start = max(
            0.0,
            float(
                segment["start"]
            ),
        )

        end = min(
            sung_end,
            float(
                segment["end"]
            ),
        )

        if (
            end
            <= start + 0.05
        ):
            continue

        segment_words = [
            dict(word)
            for word in asr_words
            if (
                word["end"] > start
                and word["start"] < end
            )
        ]

        if not segment_words:

            segment_words = distribute_words(
                segment["text"],
                start,
                end,
            )

        if segment_words:

            scenes.append(
                {
                    "start": start,
                    "end": end,
                    "text": clean_text(
                        segment["text"]
                    ),
                    "words": segment_words,
                }
            )

    return scenes


# ============================================================
# FONTES
# ============================================================

@st.cache_data(
    show_spinner=False
)
def local_fonts():

    found = []

    paths_seen = set()

    for name, path in FONT_CANDIDATES:

        if (
            os.path.isfile(path)
            and path not in paths_seen
        ):

            found.append(
                (
                    name,
                    path,
                )
            )

            paths_seen.add(
                path
            )

    if not found:

        raise RuntimeError(
            "Nenhuma fonte local segura "
            "foi encontrada."
        )

    return found


def load_font(
    path,
    size,
):

    return ImageFont.truetype(
        path,
        max(
            20,
            int(size),
        ),
    )


def text_size(
    font,
    text,
):

    bbox = font.getbbox(
        text
    )

    return (
        max(
            1,
            bbox[2] - bbox[0],
        ),
        max(
            1,
            bbox[3] - bbox[1],
        ),
        bbox,
    )


# ============================================================
# FONTES — VARIAÇÃO PALAVRA POR PALAVRA
# ============================================================

def choose_word_font(
    fonts,
    scene_index,
    word_index,
):

    if len(fonts) == 1:
        return fonts[0][1]

    """
    A fonte muda a cada palavra.

    Dentro da mesma frase:

    palavra 1 → fonte A
    palavra 2 → fonte B
    palavra 3 → fonte C
    palavra 4 → fonte D
    ...
    """

    position = (
        scene_index * 3
        + word_index
    )

    index = (
        position
        % len(fonts)
    )

    return fonts[index][1]


def fit_word_font(
    text,
    path,
    start_size,
    max_width,
    min_size=30,
):

    size = int(
        start_size
    )

    while size >= min_size:

        font = load_font(
            path,
            size,
        )

        width, _, _ = text_size(
            font,
            text,
        )

        if width <= max_width:

            return font

        size -= 2

    return load_font(
        path,
        min_size,
    )


# ============================================================
# AZUL
# ============================================================

def choose_blue_indexes(
    scene_index,
    word_count,
):

    if word_count <= 0:
        return set()

    """
    Aproximadamente 35–40% das palavras
    podem receber azul.

    Não existe fundo azul.
    É SOMENTE a cor da palavra.
    """

    indexes = set()

    for i in range(
        word_count
    ):

        value = (
            scene_index * 17
            + i * 7
            + i * i
        ) % 10

        if value < 4:

            indexes.add(i)

    if not indexes:

        indexes.add(
            word_count // 2
        )

    return indexes


# ============================================================
# LAYOUT
# ============================================================

def prepare_scene(
    scene,
    scene_index,
    fonts,
):

    words = scene["words"]

    count = len(words)

    # 50% horizontal / 50% vertical
    if scene_index % 2 == 0:

        mode = "horizontal"

    else:

        mode = "vertical"

    blue_indexes = (
        choose_blue_indexes(
            scene_index,
            count,
        )
    )

    safe_width = int(
        W * 0.86
    )

    safe_height = int(
        H * 0.68
    )

    if count <= 3:

        base_size = 112

    elif count <= 5:

        base_size = 98

    elif count <= 8:

        base_size = 82

    else:

        base_size = 68

    items = []

    for i, word in enumerate(
        words
    ):

        text = clean_text(
            word["word"]
        ).upper()

        font_path = (
            choose_word_font(
                fonts,
                scene_index,
                i,
            )
        )

        font = fit_word_font(
            text,
            font_path,
            base_size,
            safe_width,
            30,
        )

        tw, th, bbox = text_size(
            font,
            text,
        )

        items.append(
            {
                "index": i,
                "text": text,
                "font": font,
                "tw": tw,
                "th": th,
                "bbox": bbox,
                "blue": (
                    i
                    in blue_indexes
                ),
                "start": float(
                    word["start"]
                ),
                "end": float(
                    word["end"]
                ),
                "font_path": font_path,
            }
        )

    placements = []

    # ========================================================
    # VERTICAL
    # ========================================================

    if mode == "vertical":

        gap = 8

        total_height = (
            sum(
                item["th"]
                for item in items
            )
            + gap
            * max(
                0,
                len(items) - 1,
            )
        )

        if (
            total_height
            > safe_height
            and items
        ):

            scale = max(
                0.58,
                safe_height
                / total_height,
            )

            for item in items:

                new_size = max(
                    30,
                    int(
                        item[
                            "font"
                        ].size
                        * scale
                    ),
                )

                item["font"] = (
                    load_font(
                        item[
                            "font_path"
                        ],
                        new_size,
                    )
                )

                (
                    item["tw"],
                    item["th"],
                    item["bbox"],
                ) = text_size(
                    item["font"],
                    item["text"],
                )

            total_height = (
                sum(
                    item["th"]
                    for item in items
                )
                + gap
                * max(
                    0,
                    len(items) - 1,
                )
            )

        y = max(
            70,
            (
                H
                - total_height
            )
            / 2,
       