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

APP_VERSION = "19.0-AI-WORD-SYNC"

W = 720
H = 1280
FPS = 20

# Entrada MUITO rápida das palavras.
WORD_FADE = 0.055

# Duração do pequeno "pop" de entrada.
WORD_POP_DURATION = 0.10

# Tamanho máximo do pequeno pop.
WORD_POP_SCALE = 1.08

# Pequeno deslocamento vertical durante o pop.
WORD_RISE = 10

# Fade conjunto da frase ao terminar.
PHRASE_FADE = 0.10

# Transição entre fundos.
BACKGROUND_FADE = 0.20

# Cor azul.
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

        for i in range(
            count
        ):

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
            8000
            / block
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

        if max_lag >= len(
            envelope
        ):

            max_lag = (
                len(envelope)
                - 1
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
            beam_size=5,
            best_of=5,
            temperature=0.0,
            condition_on_previous_text=True,
            vad_filter=False,
            initial_prompt=(
                "Letra de música brasileira "
                "em português. "
                "Reconheça todas as palavras "
                "exatamente como são cantadas, "
                "inclusive repetições, gírias, "
                "contrações e acentos. "
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

            if (
                word.start is None
                or word.end is None
            ):
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
# FALLBACK DESATIVADO
# ============================================================

def distribute_words(
    *args,
    **kwargs,
):

    """
    NÃO distribui mais o tempo da frase.

    Mantida somente para compatibilidade
    com versões anteriores.

    Retorna lista vazia propositalmente.
    """

    return []


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
# ALINHAMENTO REAL DAS PALAVRAS
# ============================================================

def align_phrase(
    text,
    start,
    end,
    asr_words,
):

    """
    A letra oficial define:

        INÍCIO da frase
        FIM da frase

    O Whisper define:

        INÍCIO de cada palavra
        FIM de cada palavra

    NÃO existe divisão matemática do tempo.

    NÃO existe:

        duração / número_de_palavras

    Cada palavra precisa ser encontrada
    no áudio reconhecido.
    """

    tokens = re.findall(
        r"\S+",
        clean_text(
            text
        ),
    )

    if not tokens:

        return []

    # --------------------------------------------------------
    # PALAVRAS CANDIDATAS
    # --------------------------------------------------------

    candidates = [
        word
        for word in asr_words
        if (
            word["end"]
            > start - 0.35
        )
        and (
            word["start"]
            < end + 0.35
        )
    ]

    if not candidates:

        return []

    result = []

    candidate_index = 0

    # --------------------------------------------------------
    # BUSCA SEQUENCIAL
    # --------------------------------------------------------

    for token in tokens:

        best_index = None
        best_score = 0.0

        search_end = min(
            len(candidates),
            candidate_index + 10,
        )

        for j in range(
            candidate_index,
            search_end,
        ):

            candidate = candidates[
                j
            ]

            score = similarity(
                token,
                candidate["word"],
            )

            normalized_token = (
                normalize_for_match(
                    token
                )
            )