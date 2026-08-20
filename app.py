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
from PIL import Image, ImageDraw, ImageFont, ImageEnhance


# ============================================================
# LYRIC AI STUDIO
# STREAMLIT / PYTHON 3.12
# ============================================================

APP_VERSION = "7.0-FINAL-KINETIC"

FPS = 30
CACHE = Path(".lyric_cache")
FONT_DIR = CACHE / "fonts"
FONT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CORES
# ============================================================

BLACK = (7, 7, 9)
WHITE = (248, 248, 246)

# Azul royal usado somente em algumas palavras
ROYAL_BLUE = (42, 92, 255)


# ============================================================
# FONTES
# ============================================================

FONT_URLS = {
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

    "Cormorant Garamond":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/cormorantgaramond/CormorantGaramond%5Bwght%5D.ttf",
}


# ============================================================
# UTILIDADES
# ============================================================

def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_in_out(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def normalize_token(text):
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-zA-Z0-9]", "", text).lower()


def similarity(a, b):
    a = normalize_token(a)
    b = normalize_token(b)

    if not a or not b:
        return 0

    if a == b:
        return 1

    if a in b or b in a:
        return 0.88

    return difflib.SequenceMatcher(
        None, a, b, autojunk=False
    ).ratio()


# ============================================================
# FFMPEG
# ============================================================

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
        "FFmpeg não foi encontrado. Verifique o requirements.txt."
    )


def run_command(command, timeout=None):
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr[-5000:])

    return result.stdout


def media_duration(path):
    ffmpeg = get_ffmpeg()

    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-i",
            path,
            "-f",
            "null",
            "-"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    text = result.stderr

    match = re.search(
        r"Duration:\s*(\d+):(\d+):([\d.]+)",
        text
    )

    if not match:
        return 0

    h = int(match.group(1))
    m = int(match.group(2))
    s = float(match.group(3))

    return h * 3600 + m * 60 + s


# ============================================================
# FONTES
# ============================================================

@st.cache_resource(show_spinner=False)
def load_fonts():

    registry = {}

    for name, url in FONT_URLS.items():

        target = FONT_DIR / (
            name.replace(" ", "_") + ".ttf"
        )

        if not target.exists():

            try:

                request = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0"
                    }
                )

                with urllib.request.urlopen(
                    request,
                    timeout=20
                ) as response:

                    target.write_bytes(
                        response.read()
                    )

            except Exception:
                continue

        try:

            if target.exists() and target.stat().st_size > 10000:
                registry[name] = str(target)

        except Exception:
            pass

    fallback = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    if not registry:

        for path in fallback:

            if os.path.exists(path):
                registry["Fallback"] = path
                break

    return registry


def get_font(name, registry, size):

    path = registry.get(name)

    if not path:

        path = next(iter(registry.values()))

    return ImageFont.truetype(
        path,
        int(size)
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
            min(6, os.cpu_count() or 4)
        ),
        num_workers=1
    )


def transcribe(path, model_name, status):

    model = get_whisper(model_name)

    status.write(
        "🎧 Reconhecendo a música palavra por palavra..."
    )

    segments, info = model.transcribe(
        path,
        language="pt",
        task="transcribe",

        beam_size=5,
        best_of=5,

        temperature=0,

        condition_on_previous_text=True,

        vad_filter=False,

        word_timestamps=True,

        initial_prompt=(
            "Letra de música brasileira em português. "
            "Preserve palavras, repetições, gírias, "
            "nomes próprios e contrações."
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

            words.append({
                "word": text,
                "start": float(word.start),
                "end": float(word.end),
                "prob": float(
                    getattr(
                        word,
                        "probability",
                        0.0
                    ) or 0
                )
            })

    return words


# ============================================================
# ALINHAMENTO COM LETRA FORNECIDA
# ============================================================

def align_lyrics(lyrics, asr_words, duration):

    lines = [
        clean_text(x)
        for x in lyrics.splitlines()
        if clean_text(x)
    ]

    if not lines:
        return asr_words

    if not asr_words:
        return []

    result = []

    cursor = 0

    for phrase_id, line in enumerate(lines):

        tokens = re.findall(
            r"\S+",
            line
        )

        if not tokens:
            continue

        matches = []

        search = cursor

        for token in tokens:

            best_index = None
            best_score = 0

            upper = min(
                len(asr_words),
                search + 30
            )

            for i in range(search, upper):

                score = similarity(
                    token,
                    asr_words[i]["word"]
                )

                if score > best_score:

                    best_score = score
                    best_index = i

                if score >= 0.98:
                    break

            if (
                best_index is not None
                and best_score >= 0.55
            ):

                matches.append(
                    (
                        token,
                        best_index,
                        best_score
                    )
                )

                search = best_index + 1

        if not matches:

            continue

        cursor = matches[-1][1] + 1

        phrase_start = asr_words[
            matches[0][1]
        ]["start"]

        phrase_end = asr_words[
            matches[-1][1]
        ]["end"]

        mapped = {
            token: []
            for token in tokens
        }

        for token, index, score in matches:

            mapped.setdefault(
                token,
                []
            ).append(
                (
                    index,
                    score
                )
            )

        for token in tokens:

            if mapped.get(token):

                index, score = mapped[token].pop(0)

                start = asr_words[index]["start"]
                end = asr_words[index]["end"]

                result.append({
                    "word": token,
                    "start": start,
                    "end": max(
                        start + 0.055,
                        end
                    ),
                    "prob": max(
                        asr_words[index]["prob"],
                        score * 0.75
                    ),
                    "phrase_id": phrase_id,
                    "phrase_text": line
                })

            else:

                # Palavra que a transcrição não reconheceu:
                # colocamos dentro do intervalo da frase,
                # em vez de simplesmente descartá-la.
                index = len(result)

                span = max(
                    0.2,
                    phrase_end - phrase_start
                )

                position = tokens.index(token)

                start = (
                    phrase_start
                    + span
                    * position
                    / max(1, len(tokens))
                )

                end = min(
                    phrase_end,
                    start
                    + span
                    / max(1, len(tokens))
                )

                result.append({
                    "word": token,
                    "start": start,
                    "end": max(
                        start + 0.07,
                        end
                    ),
                    "prob": 0.45,
                    "phrase_id": phrase_id,
                    "phrase_text": line
                })

    return result


# ============================================================
# CRIAÇÃO DAS FRASES
# ============================================================

def create_phrases(words, duration):

    if not words:
        return []

    phrases = []

    current = []
    current_id = None

    for word in words:

        pid = word.get(
            "phrase_id",
            None
        )

        if not current:

            current = [word]
            current_id = pid
            continue

        gap = (
            word["start"]
            - current[-1]["end"]
        )

        # Quando temos letra fornecida:
        # cada linha vira uma composição.
        if (
            pid is not None
            and pid != current_id
        ):

            phrases.append(current)

            current = [word]
            current_id = pid

        # Sem letra fornecida:
        # pausa real determina troca.
        elif (
            pid is None
            and gap > 0.65
        ):

            phrases.append(current)
            current = [word]

        else:

            current.append(word)

    if current:
        phrases.append(current)

    scenes = []

    for i, phrase in enumerate(phrases):

        start = float(
            phrase[0]["start"]
        )

        end = float(
            phrase[-1]["end"]
        )

        scenes.append({
            "words": phrase,
            "start": max(
                0,
                start - 0.03
            ),
            "end": min(
                duration,
                end + 0.18
            )
        })

    # ========================================================
    # IMPORTANTE:
    # NUNCA criar vídeo depois do último momento real da música.
    # ========================================================

    for scene in scenes:

        scene["end"] = min(
            scene["end"],
            duration
        )

    return scenes


# ============================================================
# DIREÇÃO VISUAL
# ============================================================

MAIN_FONTS = [
    "Anton",
    "Bebas Neue",
    "Archivo Black",
    "Montserrat",
    "Oswald"
]

SECONDARY_FONTS = [
    "Montserrat",
    "Oswald",
    "DM Serif Display",
    "Playfair Display",
    "Cormorant Garamond",
    "Bebas Neue"
]


def choose_font(index, word_index, phrase_length, registry):

    available_main = [
        x for x in MAIN_FONTS
        if x in registry
    ]

    available_secondary = [
        x for x in SECONDARY_FONTS
        if x in registry
    ]

    if not available_main:
        return next(iter(registry))

    # Fonte grossa continua sendo dominante.
    if word_index == 0:
        return available_main[
            index % len(available_main)
        ]

    # Frases grandes recebem maior diversidade.
    if phrase_length >= 7:

        if word_index % 4 == 2 and available_secondary:
            return available_secondary[
                (index + word_index)
                % len(available_secondary)
            ]

        if word_index % 5 == 4 and available_secondary:
            return available_secondary[
                (index * 2 + word_index)
                % len(available_secondary)
            ]

    # Frases menores também podem ter uma troca.
    if (
        phrase_length >= 4
        and word_index == phrase_length // 2
        and available_secondary
    ):
        return available_secondary[
            index % len(available_secondary)
        ]

    return available_main[
        (index + word_index // 3)
        % len(available_main)
    ]


def should_be_blue(scene_index, word_index, phrase_length):

    # Aproximadamente uma frase sim / outra não.
    if scene_index % 2 != 0:
        return False

    # Não pintar muitas palavras.
    if phrase_length <= 2:
        return False

    # Algumas posições naturais.
    candidates = {
        1,
        phrase_length // 2,
        phrase_length - 1
    }

    if word_index not in candidates:
        return False

    # Evita sempre pintar a última palavra.
    if word_index == phrase_length - 1:
        return phrase_length >= 6

    return True


# ============================================================
# FUNDO MONOCROMÁTICO
# ============================================================

def create_background(
    width,
    height,
    scene_index,
    local_time,
    transition
):

    # Alternância preto / branco.
    dark = scene_index % 2 == 0

    if dark:

        current = np.full(
            (height, width, 3),
            BLACK,
            dtype=np.uint8
        )

        opposite = np.full(
            (height, width, 3),
            WHITE,
            dtype=np.uint8
        )

    else:

        current = np.full(
            (height, width, 3),
            WHITE,
            dtype=np.uint8
        )

        opposite = np.full(
            (height, width, 3),
            BLACK,
            dtype=np.uint8
        )

    # Transição muito suave entre os fundos.
    blend = ease_in_out(
        clamp(transition)
    )

    # Quase imperceptível para não ficar artificial.
    blend *= 0.85

    array = (
        current.astype(float)
        * (1 - blend)
        +
        opposite.astype(float)
        * blend
    )

    return Image.fromarray(
        np.uint8(array)
    )


# ============================================================
# TEXTO
# ============================================================

def draw_word(
    image,
    word,
    font,
    x,
    y,
    color,
    progress,
    important=False
):

    progress = clamp(progress)

    e = ease_out(progress)

    alpha = int(
        255 * e
    )

    # Entrada suave.
    offset_y = int(
        (1 - e) * 24
    )

    # Pequeno zoom inicial.
    scale = (
        0.94
        + 0.06 * e
    )

    bbox = font.getbbox(word)

    width = max(
        1,
        bbox[2] - bbox[0]
    )

    height = max(
        1,
        bbox[3] - bbox[1]
    )

    pad = 35

    layer = Image.new(
        "RGBA",
        (
            width + pad * 2,
            height + pad * 2
        ),
        (0, 0, 0, 0)
    )

    draw = ImageDraw.Draw(layer)

    fill = color + (
        alpha,
    )

    # Sombra extremamente discreta.
    if color == WHITE:

        shadow = (
            0,
            0,
            0,
            int(alpha * 0.30)
        )

    else:

        shadow = (
            255,
            255,
            255,
            int(alpha * 0.30)
        )

    draw.text(
        (
            pad + 2,
            pad + 3
        ),
        word,
        font=font,
        fill=shadow
    )

    draw.text(
        (
            pad,
            pad
        ),
        word,
        font=font,
        fill=fill
    )

    if scale != 1:

        layer = layer.resize(
            (
                int(layer.width * scale),
                int(layer.height * scale)
            ),
            Image.Resampling.LANCZOS
        )

    image.alpha_composite(
        layer,
        (
            int(x - (layer.width - width) / 2),
            int(y + offset_y)
        )
    )


# ============================================================
# RENDER DE UMA FRASE
# ============================================================

def render_scene(
    scene,
    scene_index,
    width,
    height,
    local_time,
    registry
):

    # ========================================================
    # FUNDO
    # ========================================================

    duration = max(
        0.1,
        scene["end"] - scene["start"]
    )

    # Fade final da frase.
    fade_out = clamp(
        (duration - local_time) / 0.20
    )

    background = create_background(
        width,
        height,
        scene_index,
        local_time,
        1 - fade_out
    )

    image = background.convert(
        "RGBA"
    )

    draw = ImageDraw.Draw(
        image
    )

    words = scene["words"]

    if not words:
        return image.convert("RGB")

    # ========================================================
    # PALAVRAS QUE JÁ FORAM CANTADAS
    # ========================================================

    visible = []

    for index, word in enumerate(words):

        relative = (
            local_time
            - (
                word["start"]
                - scene["start"]
            )
        )

        if relative >= 0:

            visible.append(
                (
                    index,
                    word,
                    relative
                )
            )

    if not visible:
        return image.convert("RGB")

    # ========================================================
    # CONTRASTE
    # ========================================================

    dark_background = (
        scene_index % 2 == 0
    )

    normal_color = (
        WHITE
        if dark_background
        else BLACK
    )

    # ========================================================
    # TAMANHO
    # ========================================================

    phrase_length = len(words)

    if phrase_length <= 3:

        font_size = int(
            height * 0.095
        )

    elif phrase_length <= 6:

        font_size = int(
            height * 0.080
        )

    else:

        font_size = int(
            height * 0.070
        )

    # Letras maiores que na versão anterior.
    font_size = max(
        font_size,
        92
    )

    # ========================================================
    # LAYOUT
    # ========================================================

    # Algumas frases grandes ficam empilhadas.
    stacked = (
        phrase_length >= 7
        and scene_index % 3 == 1
    )

    max_width = int(
        width * 0.84
    )

    # ========================================================
    # MONTAGEM DAS PALAVRAS
    # ========================================================

    if stacked:

        # Uma palavra por linha.
        rows = [
            [item]
            for item in visible
        ]

    else:

        rows = []

        current = []
        current_width = 0

        for item in visible:

            index, word, relative = item

            font_name = choose_font(
                scene_index,
                index,
                phrase_length,
                registry
            )

            font = get_font(
                font_name,
                registry,
                font_size
            )

            word_text = word[
                "word"
            ].upper()

            word_width, _ = text_size(
                draw,
                word_text,
                font
            )

            space_font = get_font(
                font_name,
                registry,
                font_size
            )

            space_width, _ = text_size(
                draw,
                " ",
                space_font
            )

            if (
                current
                and
                current_width
                + word_width
                + space_width
                > max_width
            ):

                rows.append(
                    current
                )

                current = []
                current_width = 0

            current.append(item)

            current_width += (
                word_width
                + (
                    space_width
                    if len(current) > 1
                    else 0
                )
            )

        if current:
            rows.append(
                current
            )

    # ========================================================
    # POSIÇÃO VERTICAL
    # ========================================================

    line_height = int(
        font_size * 1.08
    )

    total_height = (
        len(rows)
        * line_height
    )

    start_y = (
        height / 2
        - total_height / 2
    )

    # ========================================================
    # DESENHAR
    # ========================================================

    for row_index, row in enumerate(rows):

        # Calcular largura da linha.
        widths = []

        for item in row:

            index, word, relative = item

            font_name = choose_font(
                scene_index,
                index,
                phrase_length,
                registry
            )

            font = get_font(
                font_name,
                registry,
                font_size
            )

            text = word[
                "word"
            ].upper()

            w, _ = text_size(
                draw,
                text,
                font
            )

            widths.append(
                (
                    item,
                    font,
                    w
                )
            )

        spacing = int(
            font_size * 0.12
        )

        line_width = (
            sum(x[2] for x in widths)
            +
            spacing
            * max(
                0,
                len(widths) - 1
            )
        )

        cursor_x = (
            width - line_width
        ) / 2

        y = (
            start_y
            + row_index * line_height
        )

        for item, font, word_width in widths:

            index, word, relative = item

            # =================================================
            # AZUL ROYAL
            # =================================================

            blue = should_be_blue(
                scene_index,
                index,
                phrase_length
            )

            color = (
                ROYAL_BLUE
                if blue
                else normal_color
            )

            # =================================================
            # ENTRADA
            # =================================================

            progress = clamp(
                relative / 0.22
            )

            draw_word(
                image,
                word[
                    "word"
                ].upper(),
                font,
                cursor_x,
                y,
                color,
                progress,
                important=blue
            )

            cursor_x += (
                word_width
                + spacing
            )

    # ========================================================
    # FADE OUT DA FRASE
    # ========================================================

    if fade_out < 1:

        overlay = Image.new(
            "RGBA",
            (width, height),
            (
                0,
                0,
                0,
                int(
                    255
                    * (1 - fade_out)
                    * 0.08
                )
            )
        )

        image = Image.alpha_composite(
            image,
            overlay
        )

    return image.convert(
        "RGB"
    )


# ============================================================
# RENDERIZAÇÃO DO VÍDEO
# ============================================================

def render_video(
    audio_path,
    scenes,
    registry,
    output_path,
    resolution,
    quality,
    progress_bar
):

    width, height = resolution

    duration = media_duration(
        audio_path
    )

    if duration <= 0:
        raise RuntimeError(
            "Não foi possível identificar a duração da música."
        )

    ffmpeg = get_ffmpeg()

    crf = (
        "14"
        if quality == "Alta"
        else "17"
    )

    preset = (
        "slow"
        if quality == "Alta"
        else "medium"
    )

    silent = Path(
        output_path
    ).with_name(
        "video_silent.mp4"
    )

    command = [
        ffmpeg,
        "-y",

        "-f",
        "rawvideo",

        "-pix_fmt",
        "rgb24",

        "-s",
        f"{width}x{height}",

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

    total_frames = int(
        math.ceil(
            duration * FPS
        )
    )

    scene_index = 0

    try:

        for frame_number in range(
            total_frames
        ):

            time_position = (
                frame_number / FPS
            )

            # Encontrar frase atual.
            while (
                scene_index + 1
                < len(scenes)
                and
                time_position
                >= scenes[
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
                    "words": [],
                    "start": 0,
                    "end": duration
                }

            local_time = (
                time_position
                - scene["start"]
            )

            frame = render_scene(
                scene,
                scene_index,
                width,
                height,
                local_time,
                registry
            )

            process.stdin.write(
                np.asarray(
                    frame,
                    dtype=np.uint8
                ).tobytes()
            )

            if (
                progress_bar
                and frame_number % FPS == 0
            ):

                progress_bar.progress(
                    min(
                        0.94,
                        frame_number
                        / total_frames
                        * 0.94
                    ),
                    text=(
                        f"Renderizando "
                        f"{int(frame_number / total_frames * 100)}%"
                    )
                )

        process.stdin.close()

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
                "FFmpeg falhou:\n"
                + stderr[-5000:]
            )

    except Exception:

        try:
            process.stdin.close()
        except Exception:
            pass

        process.kill()

        raise

    # ========================================================
    # INSERIR ÁUDIO ORIGINAL
    # ========================================================

    final_command = [
        ffmpeg,
        "-y",

        "-i",
        str(silent),

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

        "-shortest",

        "-movflags",
        "+faststart",

        str(output_path)
    ]

    run_command(
        final_command,
        timeout=max(
            180,
            int(duration * 8)
        )
    )

    silent.unlink(
        missing_ok=True
    )

    if progress_bar:

        progress_bar.progress(
            1.0,
            text="Vídeo concluído."
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
    </style>
    """,
    unsafe_allow_html=True
)


st.title(
    "🎵 Lyric AI Studio"
)

st.caption(
    f"Final Kinetic Engine · {APP_VERSION}"
)


audio_file = st.file_uploader(
    "1. Música ou vídeo",
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
    "2. Letra oficial (RECOMENDADA)",
    height=180,
    placeholder=(
        "Cole cada frase em uma linha.\n\n"
        "Exemplo:\n"
        "Eu sei que vou te amar\n"
        "Por toda a minha vida\n"
        "Eu vou te amar"
    )
)


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
        "Qualidade",
        [
            "Equilibrado",
            "Alta"
        ]
    )


resolution = st.selectbox(
    "Resolução",
    [
        "1080 × 1920",
        "720 × 1280"
    ]
)


st.info(
    "A letra fornecida é usada como texto oficial, "
    "enquanto os timestamps do áudio determinam "
    "quando cada palavra aparece."
)


registry = load_fonts()


st.caption(
    f"{len(registry)} fontes disponíveis."
)


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

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="lyric_ai_"
        )
    )

    try:

        audio_path = (
            temp_dir
            / safe_name(
                audio_file.name
            )
        )

        audio_path.write_bytes(
            audio_file.getbuffer()
        )

        status = st.empty()

        progress = st.progress(
            0,
            text="Preparando..."
        )

        # ====================================================
        # DURAÇÃO REAL
        # ====================================================

        status.write(
            "⏱️ Detectando a duração real da música..."
        )

        duration = media_duration(
            str(audio_path)
        )

        if duration <= 0:

            raise RuntimeError(
                "Não foi possível determinar a duração do áudio."
            )

        # ====================================================
        # TRANSCRIÇÃO
        # ====================================================

        try:

            asr_words = transcribe(
                str(audio_path),
                model,
                status
            )

        except Exception:

            if model != "small":

                status.warning(
                    "Modelo pesado demais para este ambiente. "
                    "Tentando automaticamente o modelo small..."
                )

                asr_words = transcribe(
                    str(audio_path),
                    "small",
                    status
                )

            else:

                raise

        # ====================================================
        # LETRA OFICIAL
        # ====================================================

        if lyrics.strip():

            status.write(
                "🧠 Alinhando a letra oficial "
                "com o canto real..."
            )

            words = align_lyrics(
                lyrics,
                asr_words,
                duration
            )

        else:

            words = asr_words

        if not words:

            raise RuntimeError(
                "Nenhuma palavra foi reconhecida. "
                "Cole a letra oficial e tente novamente."
            )

        progress.progress(
            0.20,
            text="Sincronização concluída."
        )

        # ====================================================
        # FRASES
        # ====================================================

        status.write(
            "✍️ Montando frases e sincronização..."
        )

        scenes = create_phrases(
            words,
            duration
        )

        if not scenes:

            raise RuntimeError(
                "Não foi possível criar as frases."
            )

        progress.progress(
            0.30,
            text="Direção visual preparada."
        )

        # ====================================================
        # RESOLUÇÃO
        # ====================================================

        if resolution.startswith("1080"):

            size = (
                1080,
                1920
            )

        else:

            size = (
                720,
                1280
            )

        # ====================================================
        # RENDER
        # ====================================================

        output = (
            temp_dir
            / "lyric_ai_final.mp4"
        )

        status.write(
            "🎬 Renderizando vídeo..."
        )

        render_video(
            str(audio_path),
            scenes,
            registry,
            str(output),
            size,
            quality,
            progress
        )

        status.success(
            "✅ Vídeo criado com sucesso."
        )

        # ====================================================
        # RESULTADO
        # ====================================================

        st.video(
            str(output)
        )

        st.download_button(
            "⬇️ BAIXAR VÍDEO",
            data=output.read_bytes(),
            file_name="lyric_ai_final.mp4",
            mime="video/mp4",
            use_container_width=True
        )

        # ====================================================
        # DIAGNÓSTICO
        # ====================================================

        with st.expander(
            "Diagnóstico"
        ):

            st.write(
                f"**Duração detectada:** "
                f"{duration:.2f}s"
            )

            st.write(
                f"**Palavras sincronizadas:** "
                f"{len(words)}"
            )

            st.write(
                f"**Frases:** "
                f"{len(scenes)}"
            )

            st.write(
                "**Reconhecimento:** "
                "timestamps por palavra"
            )

            st.write(
                "**Fundo:** "
                "monocromático preto/branco"
            )

            st.write(
                "**Azul:** "
                "Royal Blue aplicado seletivamente"
            )

            st.write(
                "**Tipografia:** "
                "fonte principal grossa + "
                "variações internas"
            )

    except Exception as error:

        st.error(
            "❌ A geração falhou."
        )

        st.code(
            str(error)
        )

        st.info(
            "Se o erro estiver relacionado ao modelo, "
            "selecione 'small'."
        )

    finally:

        # Os arquivos permanecem disponíveis durante
        # a execução atual do Streamlit.
        pass