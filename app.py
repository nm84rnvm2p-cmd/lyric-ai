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

APP_VERSION = "19.0-EXACT-TIMESTAMP-FAST-FADE"

W = 720
H = 1280
FPS = 20

# ============================================================
# FADE DE ENTRADA DAS PALAVRAS
#
# Antes estava lento.
# Agora cada palavra faz fade-in muito rápido.
# ============================================================

WORD_FADE = 0.06

# ============================================================
# TRANSIÇÃO ENTRE FUNDOS
# ============================================================

BACKGROUND_FADE = 0.20

# ============================================================
# COR AZUL
# ============================================================

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

    # ========================================================
    # LOGO
    # ========================================================

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

    # ========================================================
    # CÍRCULO BRANCO
    # ========================================================

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
                    word_start + 0.01,
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
# ALINHAMENTO DO WHISPER
#
# ESTA FUNÇÃO CONTINUA EXISTINDO PARA O MODO AUTOMÁTICO.
#
# IMPORTANTE:
# quando o usuário fornece timestamps,
# ela NÃO será utilizada.
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
            word_start + 0.01,
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

    # ========================================================
    # MODO COM TIMESTAMPS
    #
    # AQUI ESTÁ A PRINCIPAL CORREÇÃO.
    #
    # O Whisper NÃO altera mais os tempos.
    #
    # O intervalo fornecido pelo usuário é absoluto.
    # ========================================================

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

            # =================================================
            # IMPORTANTE:
            #
            # NÃO usamos align_phrase().
            #
            # As palavras são distribuídas diretamente
            # entre o início e o fim fornecidos pelo usuário.
            # =================================================

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

                        # Guarda informação de que
                        # esta frase veio de timestamp manual.
                        "exact_timestamp": True,
                    }
                )

        return scenes

    # ========================================================
    # MODO AUTOMÁTICO
    #
    # Somente aqui o Whisper controla o tempo.
    # ========================================================

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

                    "exact_timestamp": False,
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
        )

        for item in items:

            x = (
                W
                - item["tw"]
            ) / 2

            placements.append(
                {
                    **item,
                    "x": int(x),
                    "y": int(y),
                }
            )

            y += (
                item["th"]
                + gap
            )

    # ========================================================
    # HORIZONTAL
    # ========================================================

    else:

        rows = []

        current = []

        row_width = 0

        space = 14

        target = (
            2
            if count % 3
            else 3
        )

        for item in items:

            required = (
                item["tw"]
                if not current
                else (
                    row_width
                    + space
                    + item["tw"]
                )
            )

            if (
                current
                and (
                    len(current)
                    >= target
                    or required
                    > safe_width
                )
            ):

                rows.append(
                    current
                )

                current = []

                row_width = 0

            current.append(
                item
            )

            if len(current) == 1:

                row_width = (
                    item["tw"]
                )

            else:

                row_width += (
                    space
                    + item["tw"]
                )

            if (
                len(current)
                == target
            ):

                rows.append(
                    current
                )

                current = []

                row_width = 0

        if current:

            rows.append(
                current
            )

        gap_y = 18

        total_height = (
            sum(
                max(
                    item["th"]
                    for item in row
                )
                for row in rows
            )
            + gap_y
            * max(
                0,
                len(rows) - 1,
            )
        )

        if (
            total_height
            > safe_height
            and items
        ):

            scale = max(
                0.55,
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

            rows = []

            current = []

            row_width = 0

            for item in items:

                required = (
                    item["tw"]
                    if not current
                    else (
                        row_width
                        + space
                        + item["tw"]
                    )
                )

                if (
                    current
                    and (
                        len(current)
                        >= target
                        or required
                        > safe_width
                    )
                ):

                    rows.append(
                        current
                    )

                    current = []

                    row_width = 0

                current.append(
                    item
                )

                if len(current) == 1:

                    row_width = (
                        item["tw"]
                    )

                else:

                    row_width += (
                        space
                        + item["tw"]
                    )

                if (
                    len(current)
                    == target
                ):

                    rows.append(
                        current
                    )

                    current = []

                    row_width = 0

            if current:

                rows.append(
                    current
                )

            total_height = (
                sum(
                    max(
                        item["th"]
                        for item in row
                    )
                    for row in rows
                )
                + gap_y
                * max(
                    0,
                    len(rows) - 1,
                )
            )

        y = max(
            70,
            (
                H
                - total_height
            )
            / 2,
        )

        for row in rows:

            row_height = max(
                item["th"]
                for item in row
            )

            row_width = (
                sum(
                    item["tw"]
                    for item in row
                )
                + space
                * max(
                    0,
                    len(row) - 1,
                )
            )

            x = (
                W
                - row_width
            ) / 2

            for item in row:

                placements.append(
                    {
                        **item,
                        "x": int(x),
                        "y": int(
                            y
                            + (
                                row_height
                                - item[
                                    "th"
                                ]
                            )
                            / 2
                        ),
                    }
                )

                x += (
                    item["tw"]
                    + space
                )

            y += (
                row_height
                + gap_y
            )

    scene["mode"] = mode

    scene["placements"] = (
        placements
    )


# ============================================================
# CENA NO TEMPO
# ============================================================

def scene_at_time(
    scenes,
    t,
    hint=0,
):

    if not scenes:
        return -1

    index = min(
        max(
            0,
            hint,
        ),
        len(scenes) - 1,
    )

    while (
        index + 1
        < len(scenes)
        and t >= scenes[index]["end"]
    ):

        index += 1

    while (
        index > 0
        and t < scenes[index]["start"]
    ):

        index -= 1

    return index


# ============================================================
# FUNDO
# ============================================================

def background_for(
    scene_index
):

    if scene_index % 2 == 0:

        return BLACK

    return WHITE


def blend_background(
    first,
    second,
    amount,
):

    amount = max(
        0.0,
        min(
            1.0,
            amount,
        ),
    )

    amount = smoothstep(
        amount
    )

    return tuple(
        int(
            first[i]
            * (
                1
                - amount
            )
            + second[i]
            * amount
        )
        for i in range(3)
    )


# ============================================================
# DESENHA TEXTO COM ALPHA
# ============================================================

def draw_text_with_alpha(
    image,
    item,
    color,
    alpha,
):

    if alpha >= 255:

        draw = ImageDraw.Draw(
            image
        )

        draw.text(
            (
                item["x"],
                item["y"],
            ),
            item["text"],
            font=item["font"],
            fill=color,
        )

        return image

    if alpha <= 0:
        return image

    padding = 8

    width = (
        item["tw"]
        + padding * 2
    )

    height = (
        item["th"]
        + padding * 2
    )

    layer = Image.new(
        "RGBA",
        (
            width,
            height,
        ),
        (0, 0, 0, 0),
    )

    layer_draw = (
        ImageDraw.Draw(
            layer
        )
    )

    layer_draw.text(
        (
            padding
            - item["bbox"][0],
            padding
            - item["bbox"][1],
        ),
        item["text"],
        font=item["font"],
        fill=(
            color[0],
            color[1],
            color[2],
            alpha,
        ),
    )

    image.paste(
        layer,
        (
            item["x"]
            - padding,
            item["y"]
            - padding,
        ),
        layer,
    )

    return image


# ============================================================
# FRAME
# ============================================================

def draw_frame(
    scene,
    scene_index,
    time_position,
):

    current_background = (
        background_for(
            scene_index
        )
    )

    background = (
        current_background
    )

    # ========================================================
    # TRANSIÇÃO PRETO ↔ BRANCO
    # ========================================================

    if (
        scene_index > 0
        and (
            time_position
            - scene["start"]
        )
        < BACKGROUND_FADE
    ):

        previous_background = (
            background_for(
                scene_index - 1
            )
        )

        progress = (
            time_position
            - scene["start"]
        ) / BACKGROUND_FADE

        background = (
            blend_background(
                previous_background,
                current_background,
                progress,
            )
        )

    image = Image.new(
        "RGB",
        (W, H),
        background,
    )

    brightness = (
        sum(background)
        / 3
    )

    main_color = (
        BLACK
        if brightness > 128
        else WHITE
    )

    rendered = False

    # ========================================================
    # PALAVRAS
    #
    # Cada palavra entra no seu próprio momento.
    #
    # Depois do fade:
    # permanece 100% visível.
    #
    # Não existe desaparecimento individual.
    # ========================================================

    for item in scene[
        "placements"
    ]:

        if (
            time_position
            < item["start"]
        ):

            continue

        rendered = True

        # ====================================================
        # FADE-IN MUITO RÁPIDO
        # ====================================================

        age = (
            time_position
            - item["start"]
        )

        if age >= WORD_FADE:

            alpha = 255

        else:

            alpha = int(
                255
                * (
                    age
                    / WORD_FADE
                )
            )

        # ====================================================
        # COR
        # ====================================================

        if item["blue"]:

            color = ROYAL

        else:

            color = main_color

        # ====================================================
        # DESENHA
        # ====================================================

        image = draw_text_with_alpha(
            image,
            item,
            color,
            alpha,
        )

    # ========================================================
    # DESAPARECIMENTO DA FRASE INTEIRA
    #
    # Todas as palavras continuam na tela até o final.
    #
    # Quando chega ao final:
    # a frase some inteira.
    # ========================================================

    remaining = (
        scene["end"]
        - time_position
    )

    if (
        rendered
        and remaining
        < BACKGROUND_FADE
    ):

        factor = max(
            0.0,
            min(
                1.0,
                remaining
                / BACKGROUND_FADE,
            ),
        )

        if factor < 0.999:

            flat = Image.new(
                "RGB",
                image.size,
                background,
            )

            image = Image.blend(
                flat,
                image,
                factor,
            )

    return (
        image,
        rendered,
    )


# ============================================================
# VALIDAÇÃO
# ============================================================

def validate_text_visibility(
    scenes
):

    if not scenes:
        return False

    samples = []

    for scene in scenes:

        if not scene[
            "placements"
        ]:

            continue

        first_word = min(
            scene["placements"],
            key=lambda x:
            x["start"],
        )

        samples.append(
            max(
                scene["start"],
                first_word["start"]
                + 0.08,
            )
        )

    for time_position in samples[
        :8
    ]:

        index = scene_at_time(
            scenes,
            time_position,
        )

        if index < 0:
            continue

        frame, rendered = (
            draw_frame(
                scenes[index],
                index,
                time_position,
            )
        )

        if not rendered:
            continue

        background = (
            background_for(
                index
            )
        )

        pixels = frame.load()

        changed = 0

        for y in range(
            0,
            H,
            8,
        ):

            for x in range(
                0,
                W,
                8,
            ):

                if (
                    pixels[x, y]
                    != background
                ):

                    changed += 1

                    if changed > 20:

                        return True

    return False


# ============================================================
# RENDERIZAÇÃO
# ============================================================

def render_video(
    audio_path,
    scenes,
    output_path,
    quality,
    progress,
    logo,
    intro_duration,
):

    duration, has_audio = (
        media_info(
            audio_path
        )
    )

    if not has_audio:

        raise RuntimeError(
            "O arquivo enviado "
            "não possui faixa de áudio."
        )

    fonts = local_fonts()

    # ========================================================
    # PREPARA LAYOUT UMA VEZ
    # ========================================================

    for i, scene in enumerate(
        scenes
    ):

        prepare_scene(
            scene,
            i,
            fonts,
        )

    if not validate_text_visibility(
        scenes
    ):

        raise RuntimeError(
            "A validação detectou "
            "que nenhuma legenda seria "
            "visível."
        )

    # ========================================================
    # FINAL REAL DA MÚSICA
    # ========================================================

    last_word_end = max(
        (
            float(word["end"])
            for scene in scenes
            for word in scene[
                "words"
            ]
        ),
        default=0.0,
    )

    if last_word_end > 0:

        final_end = min(
            duration,
            last_word_end
            + 0.65,
        )

    else:

        final_end = duration

    if final_end < 0.5:

        final_end = duration

    ff = get_ffmpeg()

    silent_video = (
        Path(output_path)
        .with_name(
            "video_sem_audio.mp4"
        )
    )

    if quality == "Alta qualidade":

        crf = "16"
        preset = "fast"

    else:

        crf = "18"
        preset = "veryfast"

    command = [
        ff,
        "-y",
        "-f",
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
        str(silent_video),
    ]

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # ========================================================
    # INTRO
    # ========================================================

    total_duration = final_end

    total_frames = max(
        1,
        int(
            total_duration
            * FPS
        ),
    )

    scene_hint = 0

    rendered_frames = 0

    try:

        for frame_index in range(
            total_frames
        ):

            time_position = (
                frame_index
                / FPS
            )

            # =================================================
            # INTRO
            # =================================================

            if (
                time_position
                < intro_duration
            ):

                frame = (
                    draw_intro_frame(
                        logo,
                        time_position,
                        intro_duration,
                    )
                )

                rendered = False

            # =================================================
            # LETRA
            # =================================================

            else:

                scene_hint = (
                    scene_at_time(
                        scenes,
                        time_position,
                        scene_hint,
                    )
                )

                if scene_hint < 0:

                    frame = Image.new(
                        "RGB",
                        (W, H),
                        BLACK,
                    )

                    rendered = False

                else:

                    frame, rendered = (
                        draw_frame(
                            scenes[
                                scene_hint
                            ],
                            scene_hint,
                            time_position,
                        )
                    )

            if rendered:

                rendered_frames += 1

            process.stdin.write(
                frame.tobytes()
            )

            if (
                progress
                and frame_index
                % FPS
                == 0
            ):

                percent = int(
                    frame_index
                    / total_frames
                    * 100
                )

                progress.progress(
                    min(
                        0.92,
                        frame_index
                        / total_frames
                        * 0.92,
                    ),
                    text=(
                        f"Renderizando "
                        f"{percent}%"
                    ),
                )

        process.stdin.close()

        error_text = (
            process.stderr
            .read()
            .decode(
                "utf-8",
                "replace",
            )
        )

        code = process.wait()

        if code != 0:

            raise RuntimeError(
                error_text[-7000:]
                or
                "FFmpeg falhou."
            )

    finally:

        if process.poll() is None:

            try:
                process.kill()

            except Exception:
                pass

    if rendered_frames <= 0:

        silent_video.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            "Nenhum frame com "
            "legenda foi renderizado."
        )

    # ========================================================
    # ÁUDIO ORIGINAL
    # ========================================================

    run_cmd(
        [
            ff,
            "-y",
            "-i",
            str(silent_video),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-t",
            f"{final_end:.3f}",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        timeout=max(
            180,
            int(
                final_end * 8
            ),
        ),
    )

    silent_video.unlink(
        missing_ok=True
    )

    if progress:

        progress.progress(
            1.0,
            text="Vídeo concluído.",
        )

    return (
        final_end,
        rendered_frames,
    )


# ============================================================
# STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Lyric AI Studio",
    page_icon="🎵",
    layout="centered",
)

st.title(
    "🎵 Lyric AI Studio"
)

st.caption(
    f"Typography Sync · "
    f"{APP_VERSION}"
)


with st.expander(
    "Como usar",
    expanded=False,
):

    st.markdown(
        """
Cole a letra oficial com timestamps.

Exemplo:

`00:11.700 - 00:15.200 | É o que eu quero pra nós`

`00:15.200 - 00:22.900 | E que nada nesse mundo cale a nossa voz`

`00:22.900 - 00:26.000 | Céu e mar`

O vídeo possui:

- intro com logo;
- círculo expansivo;
- fade-in rápido palavra por palavra;
- todas as palavras permanecem na tela;
- a frase inteira desaparece de uma vez;
- fontes alternadas;
- fundo preto/branco;
- palavras azuis;
- transição entre fundos;
- timestamps fornecidos pelo usuário têm prioridade absoluta.
        """
    )


# ============================================================
# LOGO
# ============================================================

logo = load_logo_from_file()

if logo is not None:

    st.caption(
        "✓ Logo do perfil carregada "
        "automaticamente."
    )

else:

    st.warning(
        "A logo não foi encontrada. "
        "Coloque `logo_perfil.png` na "
        "mesma pasta do app.py. "
        "O vídeo continuará funcionando "
        "sem a logo."
    )


# ============================================================
# ARQUIVOS
# ============================================================

audio = st.file_uploader(
    "1. Música ou vídeo com a música",
    type=[
        "mp3",
        "wav",
        "m4a",
        "mp4",
        "mov",
        "webm",
    ],
)


lyrics = st.text_area(
    "2. Letra oficial (recomendada)",
    height=230,
    placeholder=(
        "00:11.700 - 00:15.200 | "
        "É o que eu quero pra nós\n"
        "00:15.200 - 00:22.900 | "
        "E que nada nesse mundo cale a nossa voz\n"
        "00:22.900 - 00:26.000 | "
        "Céu e mar"
    ),
)


column1, column2 = st.columns(
    2
)


with column1:

    model_name = st.selectbox(
        "Reconhecimento",
        [
            "small",
            "medium",
            "large-v3-turbo",
            "large-v3",
        ],
        index=0,
    )


with column2:

    quality = st.selectbox(
        "Qualidade",
        [
            "Equilibrado",
            "Alta qualidade",
        ],
        index=0,
    )


st.selectbox(
    "Resolução",
    [
        "720 × 1280",
    ],
    index=0,
    disabled=True,
)


# ============================================================
# FONTES
# ============================================================

try:

    fonts_available = (
        local_fonts()
    )

    st.caption(
        f"Fontes locais disponíveis: "
        f"{len(fonts_available)} · "
        f"variação palavra por palavra · "
        f"sem download de fontes"
    )

except Exception as exc:

    st.error(
        str(exc)
    )

    st.stop()


# ============================================================
# BOTÃO
# ============================================================

if st.button(
    "🚀 CRIAR LYRIC VIDEO",
    type="primary",
    use_container_width=True,
):

    if not audio:

        st.error(
            "Envie a música primeiro."
        )

        st.stop()

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="lyric_ai_"
        )
    )

    status = st.empty()

    progress = st.progress(
        0
    )

    try:

        # ====================================================
        # ARQUIVO
        # ====================================================

        input_path = (
            temp_dir
            / safe_filename(
                audio.name
            )
        )

        input_path.write_bytes(
            audio.getbuffer()
        )

        duration, has_audio = (
            media_info(
                input_path
            )
        )

        if not has_audio:

            raise RuntimeError(
                "O arquivo enviado "
                "não possui áudio."
            )

        # ====================================================
        # BPM
        # ====================================================

        status.write(
            "Analisando o ritmo da música..."
        )

        bpm = estimate_bpm(
            input_path
        )

        # ====================================================
        # TRANSCRIÇÃO
        # ====================================================

        try:

            (
                segments,
                asr_words,
                language,
            ) = transcribe(
                input_path,
                model_name,
                status,
            )

            used_model = model_name

        except Exception as first_error:

            if model_name == "small":

                raise RuntimeError(
                    "Falha na transcrição: "
                    f"{first_error}"
                )

            status.warning(
                "O modelo escolhido "
                "falhou. Tentando "
                "small para manter "
                "o Streamlit leve..."
            )

            (
                segments,
                asr_words,
                language,
            ) = transcribe(
                input_path,
                "small",
                status,
            )

            used_model = "small"

        if (
            not segments
            and not asr_words
        ):

            raise RuntimeError(
                "O reconhecimento "
                "não encontrou "
                "segmentos nem palavras."
            )

        # ====================================================
        # FIM DA VOZ
        # ====================================================

        last_asr_word = max(
            (
                word["end"]
                for word in asr_words
            ),
            default=0.0,
        )

        if last_asr_word > 0:

            sung_end = min(
                duration,
                last_asr_word
                + 0.65,
            )

        else:

            sung_end = duration

        # ====================================================
        # LETRA
        # ====================================================

        timed = (
            parse_timed_lyrics(
                lyrics
            )
            if lyrics.strip()
            else []
        )

        scenes = build_scenes(
            timed,
            segments,
            asr_words,
            sung_end,
        )

        if not scenes:

            raise RuntimeError(
                "Não foi possível "
                "criar as legendas. "
                "Use a letra com "
                "timestamps."
            )

        # ====================================================
        # INTRO
        # ====================================================

        first_lyric_start = (
            scenes[0]["start"]
            if scenes
            else None
        )

        intro_duration = (
            calculate_intro_duration(
                bpm,
                first_lyric_start,
            )
        )

        # ====================================================
        # LAYOUT
        # ====================================================

        for i, scene in enumerate(
            scenes
        ):

            prepare_scene(
                scene,
                i,
                fonts_available,
            )

        total_rendered_words = sum(
            len(scene["words"])
            for scene in scenes
        )

        first_caption = (
            scenes[0]["text"]
        )

        last_caption = (
            scenes[-1]["text"]
        )

        # ====================================================
        # DIAGNÓSTICO
        # ====================================================

        st.subheader(
            "Diagnóstico"
        )

        diag1, diag2 = (
            st.columns(2)
        )

        with diag1:

            st.write(
                f"**Duração:** "
                f"{duration:.2f}s"
            )

            st.write(
                f"**BPM estimado:** "
                f"{bpm:.1f}"
            )

            st.write(
                f"**Duração da intro:** "
                f"{intro_duration:.2f}s"
            )

            st.write(
                f"**Fim detectado:** "
                f"{sung_end:.2f}s"
            )

            st.write(
                f"**Segmentos:** "
                f"{len(segments)}"
            )

            st.write(
                f"**Palavras reconhecidas:** "
                f"{len(asr_words)}"
            )

        with diag2:

            st.write(
                f"**Frases:** "
                f"{len(scenes)}"
            )

            st.write(
                f"**Palavras renderizadas:** "
                f"{total_rendered_words}"
            )

            st.write(
                f"**Primeira legenda:** "
                f"{first_caption}"
            )

            st.write(
                f"**Última legenda:** "
                f"{last_caption}"
            )

            if timed:

                st.write(
                    "**Sincronização:** "
                    "TIMESTAMPS EXATOS"
                )

                st.write(
                    "**Whisper:** "
                    "não altera os tempos da letra"
                )

            else:

                st.write(
                    "**Sincronização:** "
                    "Whisper automático"
                )

        st.write(
            f"**Modelo:** "
            f"{used_model}"
            f" · idioma: {language}"
        )

        # ====================================================
        # RENDER
        # ====================================================

        output_path = (
            temp_dir
            / "lyric_ai_final.mp4"
        )

        (
            final_end,
            rendered_frames,
        ) = render_video(
            input_path,
            scenes,
            output_path,
            quality,
            progress,
            logo,
            intro_duration,
        )

        # ====================================================
        # VALIDAÇÃO FINAL
        # ====================================================

        if (
            not output_path.exists()
            or output_path.stat().st_size
            < 10000
        ):

            raise RuntimeError(
                "O MP4 final não foi "
                "criado corretamente."
            )

        final_duration, final_audio = (
            media_info(
                output_path
            )
        )

        if not final_audio:

            raise RuntimeError(
                "O MP4 final foi "
                "criado sem áudio."
            )

        if rendered_frames <= 0:

            raise RuntimeError(
                "Nenhuma legenda "
                "foi efetivamente "
                "renderizada."
            )

        st.success(
            f"Vídeo pronto · "
            f"{final_duration:.2f}s · "
            f"frames com legenda: "
            f"{rendered_frames} · "
            f"áudio: OK"
        )

        video_bytes = (
            output_path.read_bytes()
        )

        st.video(
            video_bytes
        )

        st.download_button(
            "⬇️ BAIXAR MP4",
            data=video_bytes,
            file_name=(
                "lyric_ai_video.mp4"
            ),
            mime="video/mp4",
            use_container_width=True,
        )

    except Exception as exc:

        st.error(
            "A renderização não "
            "foi concluída."
        )

        st.exception(
            exc
        )

    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )