# -*- coding: utf-8 -*-

import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# CONFIGURAÇÃO
# ============================================================

APP_VERSION = "17.0-VISUAL-FINAL"

W = 720
H = 1280
FPS = 20

FADE = 0.18

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
ROYAL = (45, 92, 255)


# ============================================================
# FONTES LOCAIS
# Não baixa nenhuma fonte durante a execução.
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
        "Liberation",
        "/usr/share/fonts/truetype/liberation/"
        "LiberationSans-Bold.ttf",
    ),
]


# ============================================================
# TEXTO / UTF-8
# ============================================================

def clean_text(value):
    """
    Mantém Unicode real.
    Remove somente caracteres de controle problemáticos.
    NÃO faz conversões latin1/utf8 perigosas.
    """

    if value is None:
        return ""

    text = unicodedata.normalize("NFC", str(value))

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

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_for_match(text):
    """
    Versão somente para comparar palavras.
    A palavra original nunca é alterada.
    """

    text = clean_text(text).lower()

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

        path = imageio_ffmpeg.get_ffmpeg_exe()

        if path and os.path.isfile(path):
            return path

    except Exception:
        pass

    path = shutil.which("ffmpeg")

    if path:
        return path

    raise RuntimeError(
        "FFmpeg não foi encontrado. "
        "Verifique o requirements.txt."
    )


def run_cmd(cmd, timeout=300):

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
            or "O comando terminou com erro."
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
# WHISPER
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def load_whisper(model_name):

    from faster_whisper import WhisperModel

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
        f"Transcrevendo com **{model_name}**..."
    )

    segments_iter, info = model.transcribe(
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

        if text and end > start:

            segments.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                }
            )

        for word in segment.words or []:

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

            if word_end <= word_start:
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

    if end <= start:
        return []

    duration = end - start

    step = duration / len(tokens)

    result = []

    for i, token in enumerate(tokens):

        word_start = (
            start
            + i * step
        )

        if i == len(tokens) - 1:

            word_end = end

        else:

            word_end = (
                start
                + (i + 1) * step
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
# COMPARAÇÃO DE PALAVRAS
# ============================================================

def similarity(a, b):

    import difflib

    aa = normalize_for_match(a)
    bb = normalize_for_match(b)

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
# ALINHAMENTO DA LETRA OFICIAL
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
        if word["end"] > start - 0.20
        and word["start"] < end + 0.20
    ]

    result = [
        None
        for _ in tokens
    ]

    used = set()

    # Tenta aproveitar os timestamps reais do Whisper.
    for i, token in enumerate(tokens):

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

            used.add(best_index)

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

    # Fallback para palavras que o Whisper não alinhou.
    known = [
        i
        for i, item in enumerate(result)
        if item is not None
    ]

    for i in range(len(tokens)):

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
            result[previous]["end"]
            if previous >= 0
            else start
        )

        right = (
            result[following]["start"]
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
                (i - previous - 1)
                / count
            )
        )

        word_end = (
            left
            + (
                right - left
            )
            * (
                (i - previous)
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
# LEITURA DOS TIMESTAMPS
# ============================================================

def parse_time(value):

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


def parse_timed_lyrics(text):

    """
    Aceita:

    00:02.3 - 00:06.8 | frase

    ou:

    00:02.3 | frase
    """

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

        # START - END
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

        # START | TEXTO
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
        key=lambda x: x["start"]
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

    # --------------------------------------------------------
    # LETRA COM TIMESTAMPS
    # --------------------------------------------------------

    if timed:

        for i, line in enumerate(
            timed
        ):

            start = max(
                0.0,
                float(line["start"]),
            )

            if start >= sung_end:
                break

            if line.get("end") is not None:

                end = min(
                    float(line["end"]),
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

            if end <= start + 0.05:
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

    # --------------------------------------------------------
    # SEM TIMESTAMPS OFICIAIS
    # --------------------------------------------------------

    for segment in segments:

        start = max(
            0.0,
            float(segment["start"]),
        )

        end = min(
            sung_end,
            float(segment["end"]),
        )

        if end <= start + 0.05:
            continue

        segment_words = [
            dict(word)
            for word in asr_words
            if word["end"] > start
            and word["start"] < end
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

    for name, path in FONT_CANDIDATES:

        if (
            os.path.isfile(path)
            and path
            not in [
                item[1]
                for item in found
            ]
        ):

            found.append(
                (
                    name,
                    path,
                )
            )

    if not found:

        raise RuntimeError(
            "Nenhuma fonte local segura "
            "foi encontrada no Streamlit Cloud."
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
# VARIAÇÃO DE FONTES
# ============================================================

def choose_word_font(
    fonts,
    word_index,
):

    if len(fonts) == 1:
        return fonts[0][1]

    # Grossa continua majoritária.
    pattern = [
        0,
        0,
        1,
        0,
        2,
        0,
        0,
        1,
        0,
        2,
    ]

    index = (
        pattern[
            word_index
            % len(pattern)
        ]
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
# AZUL ROYAL
# ============================================================

def choose_blue_indexes(
    scene_index,
    word_count,
):

    if word_count == 0:
        return set()

    # Frase sim / frase não.
    if scene_index % 2 == 0:
        return set()

    if word_count == 1:
        return {0}

    if word_count <= 4:
        return {
            word_count - 1
        }

    return {
        max(
            0,
            word_count - 2,
        ),
        word_count - 1,
    }


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

    # 50% horizontal / 50% vertical.
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

    # --------------------------------------------------------
    # VERTICAL
    # --------------------------------------------------------

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

                item["font"] = load_font(
                    item["font_path"],
                    new_size,
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

    # --------------------------------------------------------
    # HORIZONTAL
    # 2 ou 3 palavras por linha.
    # --------------------------------------------------------

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

                item["font"] = load_font(
                    item["font_path"],
                    new_size,
                )

                (
                    item["tw"],
                    item["th"],
                    item["bbox"],
                ) = text_size(
                    item["font"],
                    item["text"],
                )

            # Reconstruir as linhas depois da redução.
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

    # A posição final é congelada aqui.
    scene["mode"] = mode
    scene["placements"] = placements


# ============================================================
# CENA ATUAL
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
# SOMENTE PRETO / BRANCO
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

    return tuple(
        int(
            first[i]
            * (1 - amount)
            + second[i]
            * amount
        )
        for i in range(3)
    )


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

    # Fade simples entre preto e branco.
    if (
        scene_index > 0
        and (
            time_position
            - scene["start"]
        )
        < FADE
    ):

        previous_background = (
            background_for(
                scene_index - 1
            )
        )

        background = (
            blend_background(
                previous_background,
                current_background,
                (
                    time_position
                    - scene["start"]
                )
                / FADE,
            )
        )

    image = Image.new(
        "RGB",
        (W, H),
        background,
    )

    draw = ImageDraw.Draw(
        image
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

    # --------------------------------------------------------
    # PALAVRAS
    # --------------------------------------------------------

    for item in scene[
        "placements"
    ]:

        if (
            time_position
            < item["start"]
        ):

            continue

        rendered = True

        # Fade somente de opacidade.
        # NÃO existe movimento.
        alpha = 255

        age = (
            time_position
            - item["start"]
        )

        if age < 0.10:

            alpha = max(
                0,
                min(
                    255,
                    int(
                        255
                        * age
                        / 0.10
                    ),
                ),
            )

        if item["blue"]:

            color = ROYAL

        else:

            color = main_color

        # Texto azul é SOMENTE texto.
        # Nenhum retângulo, glow ou fundo.
        if alpha >= 255:

            draw.text(
                (
                    item["x"],
                    item["y"],
                ),
                item["text"],
                font=item["font"],
                fill=color,
            )

        else:

            layer = Image.new(
                "RGBA",
                (W, H),
                (0, 0, 0, 0),
            )

            layer_draw = (
                ImageDraw.Draw(
                    layer
                )
            )

            layer_draw.text(
                (
                    item["x"],
                    item["y"],
                ),
                item["text"],
                font=item["font"],
                fill=(
                    *color,
                    alpha,
                ),
            )

            image = (
                Image.alpha_composite(
                    image.convert(
                        "RGBA"
                    ),
                    layer,
                ).convert(
                    "RGB"
                )
            )

    # --------------------------------------------------------
    # FADE OUT DA FRASE
    # --------------------------------------------------------

    remaining = (
        scene["end"]
        - time_position
    )

    if (
        rendered
        and remaining < FADE
    ):

        factor = max(
            0.0,
            min(
                1.0,
                remaining / FADE,
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
# TESTE REAL DE VISIBILIDADE
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
            key=lambda x: x[
                "start"
            ],
        )

        samples.append(
            max(
                scene["start"],
                first_word["start"]
                + 0.12,
            )
        )

    for time_position in samples[:8]:

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

        # Amostragem leve para não gastar memória.
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

    # Prepara todos os layouts uma única vez.
    for i, scene in enumerate(
        scenes
    ):

        prepare_scene(
            scene,
            i,
            fonts,
        )

    # Não permite gerar um vídeo sem texto.
    if not validate_text_visibility(
        scenes
    ):

        raise RuntimeError(
            "A validação detectou "
            "que nenhuma legenda seria "
            "visível. A renderização foi "
            "cancelada."
        )

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

    # Detecta o final real da fala.
    # Dá pequena margem para a última sílaba.
    if last_word_end > 0:

        final_end = min(
            duration,
            last_word_end + 0.65,
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

    total_frames = max(
        1,
        int(
            final_end
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
                and frame_index % FPS
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
                "FFmpeg falhou "
                "ao renderizar."
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

    # --------------------------------------------------------
    # ADICIONA O ÁUDIO ORIGINAL
    # --------------------------------------------------------

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
# INTERFACE STREAMLIT
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
Cole a letra oficial com timestamps para máxima precisão.

Exemplo:

`00:02.30 - 00:06.80 | Não tenho vergonha de dizer que sou maluco por você`

Também funciona:

`00:02.30 | Não tenho vergonha de dizer que sou maluco por você`

O fundo alterna automaticamente entre preto e branco.
        """
    )


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
        "00:02.30 - 00:06.80 | "
        "Não tenho vergonha de dizer que sou maluco por você\n"
        "00:06.80 - 00:10.50 | "
        "É, sou maluco, deixa o mundo perceber"
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


try:

    fonts_available = (
        local_fonts()
    )

    st.caption(
        f"Fontes locais disponíveis: "
        f"{len(fonts_available)} · "
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

        # ----------------------------------------------------
        # ARQUIVO ORIGINAL
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # TRANSCRIÇÃO
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # FIM REAL DA VOZ
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # CONSTRUÇÃO DAS CENAS
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # PREPARA LAYOUT
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # DIAGNÓSTICO
        # ----------------------------------------------------

        st.subheader(
            "Diagnóstico"
        )

        diag1, diag2 = (
            st.columns(2)
        )

        with diag1:

            st.write(
                f"**Duração do arquivo:** "
                f"{duration:.2f}s"
            )

            st.write(
                f"**Fim detectado:** "
                f"{sung_end:.2f}s"
            )

            st.write(
                f"**Segmentos reconhecidos:** "
                f"{len(segments)}"
            )

            st.write(
                f"**Palavras reconhecidas:** "
                f"{len(asr_words)}"
            )

        with diag2:

            st.write(
                f"**Frases renderizadas:** "
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

        st.write(
            f"**Modelo utilizado:** "
            f"{used_model}"
            f" · idioma: {language}"
        )

        # ----------------------------------------------------
        # RENDERIZAÇÃO
        # ----------------------------------------------------

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
        )

        # ----------------------------------------------------
        # VALIDAÇÃO FINAL
        # ----------------------------------------------------

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