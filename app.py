import os
import re
import math
import shutil
import subprocess
import tempfile
import urllib.request
import unicodedata
import difflib
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# LYRIC AI STUDIO — FINAL STREAMLIT VERSION
# ============================================================

APP_VERSION = "7.0-FINAL"

FPS = 30

CACHE_DIR = Path(".lyric_cache")
FONT_DIR = CACHE_DIR / "fonts"
FONT_DIR.mkdir(parents=True, exist_ok=True)

ROYAL_BLUE = (65, 105, 225)

# ------------------------------------------------------------
# FONTES
# ------------------------------------------------------------

FONT_URLS = {
    "Montserrat":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/montserrat/Montserrat%5Bwght%5D.ttf",

    "Oswald":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/oswald/Oswald%5Bwght%5D.ttf",

    "Anton":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/anton/Anton-Regular.ttf",

    "Bebas Neue":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/bebasneue/BebasNeue-Regular.ttf",

    "Playfair Display":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/playfairdisplay/PlayfairDisplay%5Bwght%5D.ttf",

    "DM Serif Display":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/dmserifdisplay/DMSerifDisplay-Regular.ttf",

    "Archivo Black":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/archivoblack/ArchivoBlack-Regular.ttf",
}

SYSTEM_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
]


# ------------------------------------------------------------
# UTILIDADES
# ------------------------------------------------------------

def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_word(word):
    word = unicodedata.normalize("NFKD", word or "")
    word = "".join(c for c in word if not unicodedata.combining(c))
    return re.sub(r"[^a-zA-Z0-9]", "", word).lower()


def similarity(a, b):
    a = normalize_word(a)
    b = normalize_word(b)

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


def clamp(value, minimum=0.0, maximum=1.0):
    return max(minimum, min(maximum, value))


def ease_out(value):
    value = clamp(value)
    return 1 - (1 - value) ** 3


def ease_in_out(value):
    value = clamp(value)
    return value * value * (3 - 2 * value)


# ------------------------------------------------------------
# FFMPEG
# ------------------------------------------------------------

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
        "FFmpeg não foi encontrado. "
        "Verifique se imageio-ffmpeg está no requirements.txt."
    )


def run_command(command, timeout=None):
    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout
    )

    if process.returncode != 0:
        raise RuntimeError(process.stderr[-5000:])

    return process.stdout


# ------------------------------------------------------------
# DURAÇÃO / ÁUDIO
# ------------------------------------------------------------

def get_duration(path):
    ffmpeg = get_ffmpeg()

    process = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-i",
            str(path),
            "-f",
            "null",
            "-"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60
    )

    text = process.stderr

    match = re.search(
        r"Duration:\s*(\d+):(\d+):([\d.]+)",
        text
    )

    if not match:
        return 0.0

    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))

    return hours * 3600 + minutes * 60 + seconds


def extract_audio(input_path, output_path):
    ffmpeg = get_ffmpeg()

    run_command(
        [
            ffmpeg,
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_path)
        ],
        timeout=180
    )


# ------------------------------------------------------------
# FONTES
# ------------------------------------------------------------

def download_font(name):
    target = FONT_DIR / (
        re.sub(r"[^A-Za-z0-9_-]", "_", name) + ".ttf"
    )

    if target.exists() and target.stat().st_size > 10000:
        return str(target)

    url = FONT_URLS.get(name)

    if not url:
        return None

    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "LyricAI"}
        )

        with urllib.request.urlopen(request, timeout=20) as response:
            data = response.read()

        if len(data) > 10000:
            target.write_bytes(data)
            return str(target)

    except Exception:
        pass

    return None


@st.cache_resource(show_spinner=False)
def load_fonts():
    fonts = {}

    for name in FONT_URLS:
        path = download_font(name)

        if path:
            fonts[name] = path

    if not fonts:
        for path in SYSTEM_FONTS:
            if os.path.exists(path):
                fonts["System"] = path
                break

    return fonts


def get_font_path(name, fonts):
    if name in fonts:
        return fonts[name]

    if fonts:
        return next(iter(fonts.values()))

    raise RuntimeError("Nenhuma fonte disponível.")


def load_font(path, size):
    return ImageFont.truetype(
        path,
        max(20, int(size))
    )


def text_size(draw, text, font):
    box = draw.textbbox(
        (0, 0),
        text,
        font=font
    )

    return (
        box[2] - box[0],
        box[3] - box[1]
    )


# ------------------------------------------------------------
# WHISPER
# ------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_whisper(model_name):
    from faster_whisper import WhisperModel

    return WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
        cpu_threads=max(
            2,
            min(6, os.cpu_count() or 4)
        ),
        num_workers=1
    )


def transcribe(audio_path, model_name, status):
    model = load_whisper(model_name)

    status.write(
        f"🎤 Transcrevendo com **{model_name}**..."
    )

    segments, info = model.transcribe(
        str(audio_path),
        language="pt",
        task="transcribe",

        beam_size=5,
        best_of=5,
        patience=1.0,
        temperature=0.0,

        word_timestamps=True,

        # Importante para música:
        # não deixar o VAD apagar palavras cantadas rapidamente.
        vad_filter=False,

        condition_on_previous_text=True,

        initial_prompt=(
            "Letra de música brasileira em português. "
            "Preserve palavras, repetições, gírias, "
            "contrações e nomes próprios. "
            "Não traduza."
        )
    )

    words = []

    for segment in segments:

        if not segment.words:
            continue

        for word in segment.words:

            text = clean_text(word.word)

            if not text:
                continue

            words.append(
                {
                    "word": text,
                    "start": float(word.start),
                    "end": float(word.end),
                    "prob": float(
                        getattr(
                            word,
                            "probability",
                            0.0
                        ) or 0.0
                    )
                }
            )

    return words, getattr(info, "language", "pt")


# ------------------------------------------------------------
# ALINHAMENTO DA LETRA MANUAL
# ------------------------------------------------------------

def align_lyrics(lyrics, asr_words, duration):
    """
    A letra fornecida pelo usuário é usada como texto oficial.

    O áudio continua sendo responsável pelo tempo.

    Isso evita que erros ortográficos do Whisper apareçam
    no vídeo quando o usuário conhece a letra correta.
    """

    lines = [
        clean_text(line)
        for line in lyrics.splitlines()
        if clean_text(line)
    ]

    if not lines:
        return asr_words

    if not asr_words:
        return fallback_timing(
            lyrics,
            duration
        )

    result = []

    cursor = 0

    total_asr = len(asr_words)

    for phrase_id, line in enumerate(lines):

        tokens = re.findall(
            r"\S+",
            line
        )

        if not tokens:
            continue

        matched = []

        search_from = cursor

        for token in tokens:

            best_index = None
            best_score = 0.0

            # Janela relativamente grande para músicas rápidas.
            search_to = min(
                total_asr,
                search_from + 35
            )

            for index in range(
                search_from,
                search_to
            ):

                score = similarity(
                    token,
                    asr_words[index]["word"]
                )

                if score > best_score:
                    best_score = score
                    best_index = index

                if score >= 0.98:
                    break

            if (
                best_index is not None
                and best_score >= 0.52
            ):

                matched.append(
                    (
                        token,
                        best_index,
                        best_score
                    )
                )

                search_from = best_index + 1

        if matched:

            first = matched[0][1]
            last = matched[-1][1]

            phrase_start = asr_words[first]["start"]
            phrase_end = asr_words[last]["end"]

            cursor = last + 1

        else:

            if cursor < total_asr:
                phrase_start = asr_words[cursor]["start"]
            else:
                phrase_start = duration

            phrase_end = min(
                duration,
                phrase_start + 1.5
            )

        # Mapa de posições.
        mapped = {}

        for token, index, score in matched:
            mapped.setdefault(
                normalize_word(token),
                []
            ).append(
                (index, score)
            )

        for token_index, token in enumerate(tokens):

            key = normalize_word(token)

            if key in mapped and mapped[key]:

                index, score = mapped[key].pop(0)

                start = asr_words[index]["start"]
                end = asr_words[index]["end"]

                result.append(
                    {
                        "word": token,
                        "start": start,
                        "end": max(
                            end,
                            start + 0.055
                        ),
                        "prob": max(
                            asr_words[index]["prob"],
                            score * 0.75
                        ),
                        "phrase_id": phrase_id,
                        "phrase_text": line
                    }
                )

            else:

                # Palavra não encontrada:
                # interpolar somente dentro da frase.
                previous = [
                    x for x in result
                    if x.get("phrase_id") == phrase_id
                    and x["word"] != token
                ]

                future_indices = [
                    index
                    for index in range(
                        cursor,
                        total_asr
                    )
                    if similarity(
                        token,
                        asr_words[index]["word"]
                    ) >= 0.52
                ]

                if previous:
                    start = previous[-1]["end"]
                else:
                    start = phrase_start

                if future_indices:
                    target = asr_words[
                        future_indices[0]
                    ]["start"]
                else:
                    target = phrase_end

                remaining = max(
                    0.10,
                    target - start
                )

                missing_count = max(
                    1,
                    len(tokens) - token_index
                )

                end = start + (
                    remaining / missing_count
                )

                end = min(
                    max(end, start + 0.055),
                    phrase_end
                )

                result.append(
                    {
                        "word": token,
                        "start": start,
                        "end": end,
                        "prob": 0.45,
                        "phrase_id": phrase_id,
                        "phrase_text": line
                    }
                )

    result.sort(
        key=lambda x: (
            x.get("phrase_id", 0),
            x["start"]
        )
    )

    return repair_timestamps(result)


def fallback_timing(lyrics, duration):
    lines = [
        clean_text(line)
        for line in lyrics.splitlines()
        if clean_text(line)
    ]

    if not lines:
        return []

    total_words = sum(
        len(line.split())
        for line in lines
    )

    if total_words == 0:
        return []

    result = []

    current = 0.0

    for phrase_id, line in enumerate(lines):

        words = line.split()

        phrase_duration = (
            duration * len(words) / total_words
        )

        word_duration = (
            phrase_duration / len(words)
        )

        for index, word in enumerate(words):

            start = (
                current +
                index * word_duration
            )

            end = (
                start +
                word_duration
            )

            result.append(
                {
                    "word": word,
                    "start": start,
                    "end": max(
                        start + 0.055,
                        end
                    ),
                    "prob": 0.25,
                    "phrase_id": phrase_id,
                    "phrase_text": line
                }
            )

        current += phrase_duration

    return result


def repair_timestamps(words):
    previous = 0.0

    for word in words:

        start = max(
            previous,
            float(word["start"])
        )

        end = max(
            start + 0.055,
            float(word["end"])
        )

        word["start"] = start
        word["end"] = end

        previous = start

    return words


# ------------------------------------------------------------
# LIMPEZA
# ------------------------------------------------------------

def clean_words(words):
    result = []

    for word in words:

        text = clean_text(word["word"])

        if not text:
            continue

        if len(text) > 40:
            continue

        item = dict(word)
        item["word"] = text

        result.append(item)

    return repair_timestamps(result)


# ------------------------------------------------------------
# CRIAÇÃO DAS FRASES
# ------------------------------------------------------------

def create_scenes(words):
    """
    Se houver letra manual:
    cada linha da letra vira uma frase.

    Se não houver:
    pausas naturais e pontuação determinam as frases.
    """

    if not words:
        return []

    manual = any(
        "phrase_id" in word
        for word in words
    )

    scenes = []

    current = []
    current_id = None

    for word in words:

        if not current:

            current = [word]
            current_id = word.get(
                "phrase_id"
            )

            continue

        if manual:

            if word.get("phrase_id") != current_id:

                scenes.append(current)

                current = [word]
                current_id = word.get(
                    "phrase_id"
                )

                continue

        gap = (
            word["start"]
            - current[-1]["end"]
        )

        punctuation = bool(
            re.search(
                r"[.!?;:]$",
                current[-1]["word"]
            )
        )

        phrase_duration = (
            word["end"]
            - current[0]["start"]
        )

        # Não quebrar frases rapidamente.
        if (
            not manual
            and (
                gap > 0.65
                or punctuation
                or len(current) >= 16
                or phrase_duration >= 8.5
            )
        ):

            scenes.append(current)
            current = [word]

        else:

            current.append(word)

    if current:
        scenes.append(current)

    output = []

    for index, scene in enumerate(scenes):

        start = float(
            scene[0]["start"]
        )

        end = float(
            scene[-1]["end"]
        )

        output.append(
            {
                "words": scene,
                "start": max(
                    0.0,
                    start - 0.03
                ),
                "end": end + 0.18,
                "phrase": " ".join(
                    word["word"]
                    for word in scene
                ),
                "index": index
            }
        )

    return output


# ------------------------------------------------------------
# FIM REAL DA MÚSICA
# ------------------------------------------------------------

def detect_real_end(words, audio_duration):
    """
    A letra pode ser maior que o que o cantor realmente cantou.

    Portanto o vídeo termina baseado na última palavra realmente
    reconhecida no áudio, e não na última linha da letra.
    """

    if not words:
        return audio_duration

    last_word = max(
        words,
        key=lambda x: x["end"]
    )

    end = float(last_word["end"]) + 0.65

    return min(
        audio_duration,
        end
    )


# ------------------------------------------------------------
# ESTILO
# ------------------------------------------------------------

SERIF_FONTS = [
    "Playfair Display",
    "DM Serif Display"
]

SANS_FONTS = [
    "Montserrat",
    "Oswald",
    "Anton",
    "Bebas Neue",
    "Archivo Black"
]


def choose_font(fonts, index, large_phrase=False):
    available_sans = [
        name
        for name in SANS_FONTS
        if name in fonts
    ]

    available_serif = [
        name
        for name in SERIF_FONTS
        if name in fonts
    ]

    if not available_sans:
        available_sans = list(fonts.keys())

    if not available_serif:
        available_serif = list(fonts.keys())

    # Fonte majoritária:
    # Sans forte e limpa.
    if not large_phrase:
        return available_sans[
            index % len(available_sans)
        ]

    # Frases grandes ganham mais variedade.
    if index % 3 == 0:
        return available_serif[
            index % len(available_serif)
        ]

    return available_sans[
        index % len(available_sans)
    ]


# ------------------------------------------------------------
# FUNDO MONOCROMÁTICO
# ------------------------------------------------------------

def background_color(scene_index):
    # Alternância deliberadamente simples:
    # preto -> branco -> preto -> branco.
    return (
        (8, 8, 8)
        if scene_index % 2 == 0
        else (248, 248, 248)
    )


def foreground_color(bg):
    if sum(bg) < 300:
        return (248, 248, 248)

    return (8, 8, 8)


# ------------------------------------------------------------
# ESCOLHA DO AZUL
# ------------------------------------------------------------

def should_be_blue(word_index, scene_index, total_words):
    """
    Azul é raro.
    Nunca transforma uma frase inteira em azul.
    """

    # aproximadamente 10-15% das palavras.
    value = (
        word_index * 37
        + scene_index * 17
        + total_words * 11
    ) % 19

    return value in (0, 1)


# ------------------------------------------------------------
# LAYOUT
# ------------------------------------------------------------

def calculate_layout(
    draw,
    words,
    font,
    max_width,
    stacked=False
):
    """
    Decide automaticamente como distribuir as palavras.

    Frases pequenas:
        UMA DUAS TRÊS

    Frases maiores:
        UMA DUAS
        TRÊS QUATRO
        CINCO SEIS

    Algumas frases grandes podem usar:
        UMA
        DUAS
        TRÊS
        QUATRO
    """

    if not words:
        return []

    space_width = text_size(
        draw,
        " ",
        font
    )[0]

    rows = []
    current = []
    current_width = 0

    for word in words:

        text = word["word"].upper()

        width = text_size(
            draw,
            text,
            font
        )[0]

        if not current:

            current = [word]
            current_width = width

        elif (
            current_width
            + space_width
            + width
            <= max_width
        ):

            current.append(word)

            current_width += (
                space_width + width
            )

        else:

            rows.append(current)

            current = [word]
            current_width = width

    if current:
        rows.append(current)

    # Para frases grandes:
    # ocasionalmente usar empilhamento.
    if len(words) >= 7 and len(rows) == 1:
        if len(words) >= 9:
            rows = [
                words[:len(words)//2],
                words[len(words)//2:]
            ]

    return rows


# ------------------------------------------------------------
# RENDER DE UMA CENA
# ------------------------------------------------------------

def render_scene(
    scene,
    width,
    height,
    fonts,
    local_time,
    previous_background,
    current_background
):

    background = Image.new(
        "RGB",
        (width, height),
        current_background
    )

    image = background.convert("RGBA")

    words = scene.get("words", [])

    if not words:
        return image.convert("RGB")

    draw = ImageDraw.Draw(image)

    foreground = foreground_color(
        current_background
    )

    scene_index = scene.get(
        "index",
        0
    )

    total_words = len(words)

    large_phrase = (
        total_words >= 7
    )

    # --------------------------------------------------------
    # TAMANHO DA FONTE
    # --------------------------------------------------------

    if total_words <= 3:
        font_size = int(
            height * 0.105
        )

    elif total_words <= 6:
        font_size = int(
            height * 0.090
        )

    elif total_words <= 9:
        font_size = int(
            height * 0.078
        )

    else:
        font_size = int(
            height * 0.068
        )

    # Nunca deixar pequeno demais.
    font_size = max(
        font_size,
        58
    )

    font_name = choose_font(
        fonts,
        scene_index,
        large_phrase
    )

    font_path = get_font_path(
        font_name,
        fonts
    )

    font = load_font(
        font_path,
        font_size
    )

    max_width = int(
        width * 0.86
    )

    rows = calculate_layout(
        draw,
        words,
        font,
        max_width
    )

    if not rows:
        return image.convert("RGB")

    line_height = int(
        font_size * 1.12
    )

    total_height = (
        len(rows)
        * line_height
    )

    start_y = (
        height / 2
        - total_height / 2
    )

    # --------------------------------------------------------
    # PALAVRAS APARECEM INDIVIDUALMENTE
    # --------------------------------------------------------

    spoken = []

    for index, word in enumerate(words):

        relative_start = (
            word["start"]
            - scene["start"]
        )

        if local_time >= relative_start:

            spoken.append(
                (
                    index,
                    word,
                    relative_start
                )
            )

    if not spoken:
        return image.convert("RGB")

    spoken_indices = {
        item[0]
        for item in spoken
    }

    # Só renderizar palavras já cantadas.
    visible_words = [
        word
        for index, word in enumerate(words)
        if index in spoken_indices
    ]

    rows = calculate_layout(
        draw,
        visible_words,
        font,
        max_width
    )

    if not rows:
        return image.convert("RGB")

    line_height = int(
        font_size * 1.12
    )

    total_height = (
        len(rows)
        * line_height
    )

    start_y = (
        height / 2
        - total_height / 2
    )

    visible_counter = 0

    # --------------------------------------------------------
    # TEXTO
    # --------------------------------------------------------

    for row_index, row in enumerate(rows):

        row_width = 0

        for word in row:

            row_width += text_size(
                draw,
                word["word"].upper(),
                font
            )[0]

        row_width += (
            (len(row) - 1)
            * text_size(
                draw,
                " ",
                font
            )[0]
        )

        x = (
            width / 2
            - row_width / 2
        )

        y = (
            start_y
            + row_index * line_height
        )

        for word in row:

            text = word["word"].upper()

            word_width, word_height = text_size(
                draw,
                text,
                font
            )

            # ----------------------------------------------
            # TEMPO DA PALAVRA
            # ----------------------------------------------

            relative_start = (
                word["start"]
                - scene["start"]
            )

            age = (
                local_time
                - relative_start
            )

            progress = clamp(
                age / 0.22
            )

            eased = ease_out(
                progress
            )

            alpha = int(
                255 * eased
            )

            offset_y = int(
                22 * (1 - eased)
            )

            # ----------------------------------------------
            # AZUL RARO
            # ----------------------------------------------

            current_index = visible_counter

            use_blue = should_be_blue(
                current_index,
                scene_index,
                total_words
            )

            color = (
                ROYAL_BLUE
                if use_blue
                else foreground
            )

            # ----------------------------------------------
            # SOMBRA MUITO SUAVE
            # ----------------------------------------------

            shadow_alpha = int(
                55 * eased
            )

            if current_background[0] < 100:
                shadow_color = (
                    0,
                    0,
                    0,
                    shadow_alpha
                )
            else:
                shadow_color = (
                    255,
                    255,
                    255,
                    shadow_alpha
                )

            draw.text(
                (
                    x + 2,
                    y + offset_y + 3
                ),
                text,
                font=font,
                fill=shadow_color
            )

            # ----------------------------------------------
            # PALAVRA
            # ----------------------------------------------

            draw.text(
                (
                    x,
                    y + offset_y
                ),
                text,
                font=font,
                fill=(
                    color[0],
                    color[1],
                    color[2],
                    alpha
                )
            )

            # ----------------------------------------------
            # PEQUENO IMPACTO NA PALAVRA NOVA
            # ----------------------------------------------

            if age >= 0 and age < 0.35:

                pulse = (
                    math.sin(
                        age * 18
                    )
                    * 0.5
                    + 0.5
                )

                if use_blue:

                    line_width = int(
                        word_width
                        * 0.45
                    )

                    line_x = (
                        x
                        + (
                            word_width
                            - line_width
                        ) / 2
                    )

                    line_y = (
                        y
                        + offset_y
                        + word_height
                        + 8
                    )

                    draw.rounded_rectangle(
                        (
                            line_x,
                            line_y,
                            line_x
                            + line_width,
                            line_y + 3
                        ),
                        radius=2,
                        fill=(
                            ROYAL_BLUE[0],
                            ROYAL_BLUE[1],
                            ROYAL_BLUE[2],
                            int(
                                70
                                * (1 - pulse)
                            )
                        )
                    )

            x += (
                word_width
                + text_size(
                    draw,
                    " ",
                    font
                )[0]
            )

            visible_counter += 1

    return image.convert("RGB")


# ------------------------------------------------------------
# TRANSIÇÃO ENTRE FRASES
# ------------------------------------------------------------

def render_frame(
    scene,
    previous_scene,
    time,
    width,
    height,
    fonts
):

    # --------------------------------------------------------
    # CENA ATUAL
    # --------------------------------------------------------

    current_bg = background_color(
        scene["index"]
    )

    previous_bg = current_bg

    if previous_scene is not None:
        previous_bg = background_color(
            previous_scene["index"]
        )

    local_time = (
        time
        - scene["start"]
    )

    current_image = render_scene(
        scene,
        width,
        height,
        fonts,
        local_time,
        previous_bg,
        current_bg
    )

    # --------------------------------------------------------
    # TRANSIÇÃO SUAVE
    # --------------------------------------------------------

    if previous_scene is None:
        return current_image

    transition_duration = 0.42

    transition_progress = (
        time
        - scene["start"]
    ) / transition_duration

    if not (
        0
        <= transition_progress
        < 1
    ):
        return current_image

    previous_duration = max(
        0.01,
        previous_scene["end"]
        - previous_scene["start"]
    )

    previous_image = render_scene(
        previous_scene,
        width,
        height,
        fonts,
        previous_duration,
        previous_bg,
        previous_bg
    )

    progress = ease_in_out(
        transition_progress
    )

    return Image.blend(
        previous_image,
        current_image,
        progress
    )


# ------------------------------------------------------------
# RENDER DO VÍDEO
# ------------------------------------------------------------

def render_video(
    audio_path,
    scenes,
    output_path,
    fonts,
    width,
    height,
    quality,
    progress_bar
):

    ffmpeg = get_ffmpeg()

    duration = get_duration(
        audio_path
    )

    if not scenes:
        raise RuntimeError(
            "Nenhuma frase foi criada."
        )

    # --------------------------------------------------------
    # FIM REAL
    # --------------------------------------------------------

    last_word_end = max(
        word["end"]
        for scene in scenes
        for word in scene["words"]
    )

    render_duration = min(
        duration,
        last_word_end + 0.70
    )

    if render_duration <= 0:
        render_duration = duration

    # --------------------------------------------------------
    # CODEC
    # --------------------------------------------------------

    if quality == "Alta qualidade":

        crf = "15"
        preset = "slow"

    else:

        crf = "18"
        preset = "medium"

    silent_video = Path(
        output_path
    ).with_name(
        "video_silent.mp4"
    )

    command = [
        ffmpeg,
        "-y",

        "-f",
        "rawvideo",

        "-vcodec",
        "rawvideo",

        "-pix_fmt",
        "rgb24",

        "-s",
        f"{width}x{height}",

        "-r",
        str(FPS),

        "-i",
        "-",

        "-t",
        str(render_duration),

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

        str(silent_video)
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
                render_duration
                * FPS
            )
        )
    )

    scene_index = 0

    try:

        for frame_number in range(
            total_frames
        ):

            current_time = (
                frame_number / FPS
            )

            # --------------------------------------------
            # LOCALIZA CENA
            # --------------------------------------------

            while (
                scene_index + 1
                < len(scenes)
                and current_time
                >= scenes[
                    scene_index
                ]["end"]
            ):
                scene_index += 1

            current_scene = scenes[
                min(
                    scene_index,
                    len(scenes) - 1
                )
            ]

            previous_scene = None

            if scene_index > 0:
                previous_scene = scenes[
                    scene_index - 1
                ]

            # --------------------------------------------
            # FRAME
            # --------------------------------------------

            frame = render_frame(
                current_scene,
                previous_scene,
                current_time,
                width,
                height,
                fonts
            )

            frame_array = np.asarray(
                frame,
                dtype=np.uint8
            )

            process.stdin.write(
                frame_array.tobytes()
            )

            if (
                progress_bar
                and frame_number % FPS == 0
            ):

                percentage = (
                    frame_number
                    / total_frames
                )

                progress_bar.progress(
                    min(
                        0.90,
                        percentage * 0.90
                    ),
                    text=(
                        "Renderizando vídeo... "
                        f"{int(percentage * 100)}%"
                    )
                )

        process.stdin.close()
        process.stdin = None

        error_output = (
            process.stderr
            .read()
            .decode(
                "utf-8",
                errors="replace"
            )
        )

        return_code = process.wait()

        if return_code != 0:

            raise RuntimeError(
                "FFmpeg falhou ao renderizar:\n"
                + error_output[-5000:]
            )

    except BrokenPipeError:

        try:
            process.stdin.close()
        except Exception:
            pass

        error_output = (
            process.stderr
            .read()
            .decode(
                "utf-8",
                errors="replace"
            )
        )

        process.wait()

        raise RuntimeError(
            "O FFmpeg encerrou durante a renderização:\n"
            + error_output[-5000:]
        )

    # --------------------------------------------------------
    # COLOCA ÁUDIO
    # --------------------------------------------------------

    mux_command = [
        ffmpeg,
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
        "256k",

        "-t",
        str(render_duration),

        "-movflags",
        "+faststart",

        str(output_path)
    ]

    run_command(
        mux_command,
        timeout=max(
            180,
            int(render_duration * 8)
        )
    )

    silent_video.unlink(
        missing_ok=True
    )

    if progress_bar:
        progress_bar.progress(
            1.0,
            text="Vídeo concluído!"
        )


# ============================================================
# INTERFACE STREAMLIT
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
        max-width: 950px;
        padding-top: 1.5rem;
    }

    h1 {
        letter-spacing: -0.04em;
    }

    .stButton button {
        border-radius: 12px;
        min-height: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("🎵 Lyric AI Studio")

st.caption(
    f"Final Kinetic Typography Engine · {APP_VERSION}"
)


st.markdown(
    """
    **O que esta versão prioriza:**

    • sincronização palavra por palavra  
    • letra manual alinhada ao áudio  
    • fim real da parte cantada  
    • frases permanecendo no mesmo enquadramento  
    • fontes grandes  
    • variação tipográfica  
    • fundo preto/branco monocromático  
    • letras sempre em contraste  
    • azul royal apenas em palavras pontuais  
    • fade-in nas palavras  
    • fade-out/transição suave entre frases  
    • vídeo vertical 9:16  
    """
)


# ------------------------------------------------------------
# UPLOADS
# ------------------------------------------------------------

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


lyrics = st.text_area(
    "2. Letra oficial — RECOMENDADA",
    height=180,
    placeholder=(
        "Cole a letra aqui.\n\n"
        "De preferência, coloque uma frase por linha."
    )
)


st.caption(
    "Exemplo:\n"
    "Eu quero te amar\n"
    "Até o amanhecer\n"
    "E nunca mais te esquecer"
)


# ------------------------------------------------------------
# CONFIGURAÇÕES
# ------------------------------------------------------------

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

    resolution = st.selectbox(
        "Resolução",
        [
            "1080×1920 — melhor qualidade",
            "720×1280 — mais rápido"
        ],
        index=0
    )


quality = st.selectbox(
    "Qualidade do vídeo",
    [
        "Alta qualidade",
        "Equilibrado"
    ],
    index=0
)


st.info(
    "💡 Para obter a melhor sincronização, "
    "cole a letra oficial com uma frase por linha. "
    "O texto da letra será usado como referência, "
    "mas os tempos continuarão vindo do áudio."
)


# ------------------------------------------------------------
# FONTES
# ------------------------------------------------------------

fonts = load_fonts()

st.caption(
    f"Fontes disponíveis: {len(fonts)}"
)


# ------------------------------------------------------------
# BOTÃO
# ------------------------------------------------------------

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

    temporary_directory = Path(
        tempfile.mkdtemp(
            prefix="lyric_ai_"
        )
    )

    try:

        # ----------------------------------------------------
        # SALVAR UPLOAD
        # ----------------------------------------------------

        input_path = (
            temporary_directory
            / audio_file.name
        )

        input_path.write_bytes(
            audio_file.getbuffer()
        )

        audio_path = (
            temporary_directory
            / "audio.wav"
        )

        output_path = (
            temporary_directory
            / "lyric_ai_final.mp4"
        )

        status = st.empty()

        progress = st.progress(
            0.0
        )

        # ----------------------------------------------------
        # ÁUDIO
        # ----------------------------------------------------

        status.write(
            "🎧 Preparando áudio..."
        )

        extract_audio(
            input_path,
            audio_path
        )

        duration = get_duration(
            audio_path
        )

        if duration <= 0:

            raise RuntimeError(
                "Não foi possível determinar "
                "a duração do áudio."
            )

        # ----------------------------------------------------
        # WHISPER
        # ----------------------------------------------------

        try:

            asr_words, language = transcribe(
                audio_path,
                model,
                status
            )

        except Exception as first_error:

            if model != "small":

                status.warning(
                    "O modelo escolhido falhou. "
                    "Tentando automaticamente com small..."
                )

                asr_words, language = transcribe(
                    audio_path,
                    "small",
                    status
                )

            else:

                raise first_error

        # ----------------------------------------------------
        # LETRA
        # ----------------------------------------------------

        if lyrics.strip():

            status.write(
                "📝 Alinhando a letra oficial "
                "com as palavras realmente cantadas..."
            )

            words = align_lyrics(
                lyrics,
                asr_words,
                duration
            )

        else:

            words = asr_words

        words = clean_words(
            words
        )

        if not words:

            raise RuntimeError(
                "Nenhuma palavra foi reconhecida. "
                "Tente novamente ou cole a letra oficial."
            )

        progress.progress(
            0.20,
            text="Letra sincronizada."
        )

        # ----------------------------------------------------
        # FIM REAL
        # ----------------------------------------------------

        real_end = detect_real_end(
            words,
            duration
        )

        # Não renderizar palavras além do canto real.
        words = [
            word
            for word in words
            if word["start"] < real_end
        ]

        # ----------------------------------------------------
        # FRASES
        # ----------------------------------------------------

        scenes = create_scenes(
            words
        )

        if not scenes:

            raise RuntimeError(
                "Não foi possível criar as frases."
            )

        # Limitar as cenas ao final real.
        valid_scenes = []

        for scene in scenes:

            if scene["start"] >= real_end:
                continue

            scene["end"] = min(
                scene["end"],
                real_end
            )

            valid_scenes.append(
                scene
            )

        scenes = valid_scenes

        progress.progress(
            0.30,
            text="Frases organizadas."
        )

        # ----------------------------------------------------
        # RESOLUÇÃO
        # ----------------------------------------------------

        if resolution.startswith("1080"):

            width = 1080
            height = 1920

        else:

            width = 720
            height = 1280

        # ----------------------------------------------------
        # RENDER
        # ----------------------------------------------------

        status.write(
            "🎨 Criando tipografia e transições..."
        )

        render_video(
            audio_path,
            scenes,
            output_path,
            fonts,
            width,
            height,
            quality,
            progress
        )

        # ----------------------------------------------------
        # RESULTADO
        # ----------------------------------------------------

        status.success(
            "✅ Vídeo criado com sucesso!"
        )

        st.video(
            str(output_path)
        )

        st.download_button(
            "⬇️ BAIXAR VÍDEO",
            data=output_path.read_bytes(),
            file_name="lyric_ai_final.mp4",
            mime="video/mp4",
            use_container_width=True
        )

        # ----------------------------------------------------
        # DIAGNÓSTICO
        # ----------------------------------------------------

        with st.expander(
            "🔎 Diagnóstico"
        ):

            average_probability = float(
                np.mean(
                    [
                        word["prob"]
                        for word in words
                    ]
                )
            )

            st.write(
                f"**Palavras utilizadas:** {len(words)}"
            )

            st.write(
                f"**Frases:** {len(scenes)}"
            )

            st.write(
                f"**Idioma detectado:** {language}"
            )

            st.write(
                f"**Duração do áudio:** "
                f"{duration:.2f}s"
            )

            st.write(
                f"**Fim real detectado:** "
                f"{real_end:.2f}s"
            )

            st.write(
                f"**Confiança média do reconhecimento:** "
                f"{average_probability:.2f}"
            )

            st.write(
                "**Texto utilizado:**"
            )

            st.code(
                " ".join(
                    word["word"]
                    for word in words
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
            "Se o erro estiver relacionado à memória, "
            "tente o modelo small e/ou resolução 720×1280."
        )