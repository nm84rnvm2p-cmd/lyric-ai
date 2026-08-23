# -*- coding: utf-8 -*-

import os
import re
import math
import shutil
import subprocess
import tempfile
import unicodedata
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageChops


APP_VERSION = "16.0-STABLE"

W = 720
H = 1280
FPS = 20
FADE = 0.20

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
ROYAL = (45, 92, 255)


# ============================================================
# FFmpeg
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
            result.stderr[-6000:]
            or "O FFmpeg terminou com erro."
        )

    return result.stdout, result.stderr


def get_media_info(path):
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
            "Não foi possível determinar a duração do arquivo."
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
# UTF-8 / texto
# ============================================================

def clean_text(value):
    if not value:
        return ""

    text = unicodedata.normalize(
        "NFC",
        str(value),
    )

    text = "".join(
        ch
        for ch in text
        if ch == "\n"
        or unicodedata.category(ch)
        not in ("Cc", "Cf")
    )

    text = text.replace("\ufffd", "")
    text = text.replace("□", "")

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def parse_time(value):
    value = value.strip().replace(",", ".")

    if ":" in value:
        minutes, seconds = value.split(":", 1)
        return int(minutes) * 60 + float(seconds)

    return float(value)


TIME_LINE = re.compile(
    r"^\s*"
    r"(\d{1,2}:\d{2}(?:[.,]\d{1,3})?|\d+(?:[.,]\d+)?)"
    r"\s*[-–—]\s*"
    r"(\d{1,2}:\d{2}(?:[.,]\d{1,3})?|\d+(?:[.,]\d+)?)"
    r"\s*(?:\|\s*)?(.*)$"
)


def parse_timed_lyrics(text):
    lines = (
        text or ""
    ).replace("\r", "").split("\n")

    result = []
    i = 0

    while i < len(lines):

        line = lines[i].strip()

        if not line:
            i += 1
            continue

        match = TIME_LINE.match(line)

        if match:

            start = parse_time(
                match.group(1)
            )

            end = parse_time(
                match.group(2)
            )

            lyric = clean_text(
                match.group(3)
            )

            if (
                not lyric
                and i + 1 < len(lines)
            ):
                lyric = clean_text(
                    lines[i + 1]
                )
                i += 1

            if lyric and end > start:

                result.append(
                    {
                        "start": start,
                        "end": end,
                        "text": lyric,
                    }
                )

        i += 1

    return sorted(
        result,
        key=lambda item: item["start"],
    )


# ============================================================
# Whisper
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model(model_name):

    from faster_whisper import WhisperModel

    return WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
        cpu_threads=2,
        num_workers=1,
    )


def transcribe_audio(
    path,
    model_name,
    status,
):

    status.write(
        f"🎙️ Transcrevendo com **{model_name}**..."
    )

    model = load_model(
        model_name
    )

    segments_iterator, info = model.transcribe(
        str(path),
        language="pt",
        word_timestamps=True,
        beam_size=1,
        best_of=1,
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=False,
        initial_prompt=(
            "Letra de música brasileira em português. "
            "Reconheça todas as palavras, repetições, "
            "gírias e acentos. Não traduza."
        ),
    )

    segments = []
    words = []

    for segment in segments_iterator:

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
        ),
    )


# ============================================================
# Fallback de palavras
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

    duration = max(
        0.10,
        end - start,
    )

    step = duration / len(tokens)

    result = []

    for index, token in enumerate(tokens):

        word_start = (
            start + index * step
        )

        if index == len(tokens) - 1:
            word_end = end
        else:
            word_end = (
                start
                + (index + 1) * step
            )

        result.append(
            {
                "word": clean_text(token),
                "start": word_start,
                "end": max(
                    word_start + 0.06,
                    word_end,
                ),
            }
        )

    return result


def overlap_words(
    words,
    start,
    end,
):

    return [
        dict(word)
        for word in words
        if word["end"] > start
        and word["start"] < end
    ]


# ============================================================
# Construção das legendas
# ============================================================

def build_scenes(
    timed_lyrics,
    segments,
    words,
    duration,
):

    scenes = []

    # --------------------------------------------------------
    # Letra com timestamps fornecidos pelo usuário
    # --------------------------------------------------------

    if timed_lyrics:

        for phrase in timed_lyrics:

            start = max(
                0.0,
                min(
                    duration,
                    float(
                        phrase["start"]
                    ),
                ),
            )

            end = max(
                start + 0.08,
                min(
                    duration,
                    float(
                        phrase["end"]
                    ),
                ),
            )

            if start >= duration:
                continue

            official_tokens = re.findall(
                r"\S+",
                clean_text(
                    phrase["text"]
                ),
            )

            asr_words = overlap_words(
                words,
                start,
                end,
            )

            if asr_words:

                if len(asr_words) >= len(
                    official_tokens
                ):

                    timed_words = asr_words[
                        :len(official_tokens)
                    ]

                    for index, token in enumerate(
                        official_tokens
                    ):
                        timed_words[index][
                            "word"
                        ] = clean_text(token)

                else:

                    timed_words = distribute_words(
                        phrase["text"],
                        start,
                        end,
                    )

            else:

                timed_words = distribute_words(
                    phrase["text"],
                    start,
                    end,
                )

            if timed_words:

                scenes.append(
                    {
                        "start": start,
                        "end": end,
                        "text": clean_text(
                            phrase["text"]
                        ),
                        "words": timed_words,
                    }
                )

        return scenes

    # --------------------------------------------------------
    # Letra sem timestamps
    # --------------------------------------------------------

    for segment in segments:

        start = max(
            0.0,
            min(
                duration,
                float(segment["start"]),
            ),
        )

        end = min(
            duration,
            max(
                start + 0.08,
                float(segment["end"]),
            ),
        )

        if start >= duration:
            continue

        segment_words = overlap_words(
            words,
            start,
            end,
        )

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

    # --------------------------------------------------------
    # Último fallback
    # --------------------------------------------------------

    if not scenes and words:

        for index in range(
            0,
            len(words),
            8,
        ):

            group = words[
                index:index + 8
            ]

            if not group:
                continue

            start = max(
                0.0,
                group[0]["start"],
            )

            end = min(
                duration,
                max(
                    start + 0.10,
                    group[-1]["end"]
                    + 0.18,
                ),
            )

            scenes.append(
                {
                    "start": start,
                    "end": end,
                    "text": " ".join(
                        clean_text(
                            item["word"]
                        )
                        for item in group
                    ),
                    "words": [
                        dict(item)
                        for item in group
                    ],
                }
            )

    return scenes


# ============================================================
# Fontes locais
# ============================================================

def find_local_fonts():

    candidates = [

        "/usr/share/fonts/truetype/dejavu/"
        "DejaVuSansCondensed-Bold.ttf",

        "/usr/share/fonts/truetype/dejavu/"
        "DejaVuSans-Bold.ttf",

        "/usr/share/fonts/truetype/dejavu/"
        "DejaVuSerif-Bold.ttf",

        "/usr/share/fonts/truetype/liberation2/"
        "LiberationSans-Bold.ttf",

        "/usr/share/fonts/truetype/liberation/"
        "LiberationSans-Bold.ttf",

        "/usr/share/fonts/truetype/lato/"
        "Lato-Bold.ttf",
    ]

    found = []

    for path in candidates:

        if (
            os.path.isfile(path)
            and path not in found
        ):
            found.append(path)

    if not found:

        raise RuntimeError(
            "Nenhuma fonte local compatível "
            "foi encontrada no Streamlit Cloud."
        )

    return found


FONT_PATHS = find_local_fonts()


def get_font(
    size,
    variant=0,
):

    path = FONT_PATHS[
        variant % len(FONT_PATHS)
    ]

    return ImageFont.truetype(
        path,
        max(
            20,
            int(size),
        ),
    )


# ============================================================
# Layout seguro
# ============================================================

def make_layout(
    words,
    scene_index,
):

    if not words:
        return []

    safe_width = int(
        W * 0.86
    )

    safe_height = int(
        H * 0.60
    )

    if len(words) <= 4:
        initial_size = 116

    elif len(words) <= 7:
        initial_size = 96

    elif len(words) <= 10:
        initial_size = 80

    else:
        initial_size = 68

    for size in range(
        initial_size,
        31,
        -4,
    ):

        rows = []
        current_row = []
        current_width = 0

        gap = max(
            10,
            size // 8,
        )

        for index, item in enumerate(
            words
        ):

            text = clean_text(
                item["word"]
            ).upper()

            # Fonte grossa é a principal.
            variant = 0

            if (
                index % 5 == 3
                and len(FONT_PATHS) > 1
            ):
                variant = 1

            elif (
                index % 9 == 6
                and len(FONT_PATHS) > 2
            ):
                variant = 2

            font = get_font(
                size,
                variant,
            )

            box = font.getbbox(
                text
            )

            word_width = max(
                1,
                box[2] - box[0],
            )

            word_height = max(
                1,
                box[3] - box[1],
            )

            # Palavra individual nunca ultrapassa
            # a área segura.
            if word_width > safe_width:

                test_size = size

                while (
                    word_width > safe_width
                    and test_size > 22
                ):

                    test_size -= 2

                    font = get_font(
                        test_size,
                        variant,
                    )

                    box = font.getbbox(
                        text
                    )

                    word_width = max(
                        1,
                        box[2] - box[0],
                    )

                    word_height = max(
                        1,
                        box[3] - box[1],
                    )

            required_width = (
                word_width
                if not current_row
                else current_width
                + gap
                + word_width
            )

            if (
                current_row
                and required_width
                > safe_width
            ):

                rows.append(
                    current_row
                )

                current_row = []
                current_width = 0

            current_row.append(
                (
                    index,
                    text,
                    font,
                    word_width,
                    word_height,
                )
            )

            if len(current_row) == 1:
                current_width = word_width
            else:
                current_width += (
                    gap + word_width
                )

        if current_row:
            rows.append(
                current_row
            )

        vertical_gap = max(
            14,
            size // 5,
        )

        total_height = (
            sum(
                max(
                    item[4]
                    for item in row
                )
                for row in rows
            )
            + vertical_gap
            * max(
                0,
                len(rows) - 1,
            )
        )

        if (
            len(rows) <= 4
            and total_height <= safe_height
        ):
            return rows

    # Fallback seguro:
    # uma palavra por linha.
    rows = []

    for index, item in enumerate(
        words
    ):

        text = clean_text(
            item["word"]
        ).upper()

        size = 48

        font = get_font(
            size,
            0,
        )

        box = font.getbbox(
            text
        )

        width = (
            box[2] - box[0]
        )

        while (
            width > safe_width
            and size > 20
        ):

            size -= 2

            font = get_font(
                size,
                0,
            )

            box = font.getbbox(
                text
            )

            width = (
                box[2] - box[0]
            )

        rows.append(
            [
                (
                    index,
                    text,
                    font,
                    max(
                        1,
                        box[2] - box[0],
                    ),
                    max(
                        1,
                        box[3] - box[1],
                    ),
                )
            ]
        )

    return rows


# ============================================================
# Frame
# ============================================================

def background_for_scene(
    scene_index,
):

    return (
        BLACK
        if scene_index % 2 == 0
        else WHITE
    )


def blended_background(
    previous_bg,
    current_bg,
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
            previous_bg[i]
            * (1.0 - amount)
            + current_bg[i]
            * amount
        )
        for i in range(3)
    )


def draw_frame(
    time_position,
    scenes,
    scene_index=None,
):

    if scene_index is None:

        scene_index = -1

        for index, scene in enumerate(
            scenes
        ):

            if (
                scene["start"]
                <= time_position
                < scene["end"]
            ):
                scene_index = index
                break

    if scene_index < 0:

        if (
            scenes
            and time_position
            < scenes[0]["start"]
        ):
            bg = background_for_scene(0)

        else:
            bg = background_for_scene(
                max(
                    0,
                    len(scenes) - 1,
                )
            )

        return (
            Image.new(
                "RGB",
                (W, H),
                bg,
            ),
            False,
        )

    scene = scenes[
        scene_index
    ]

    current_bg = background_for_scene(
        scene_index
    )

    bg = current_bg

    # Fade simples de 0,20s.
    if scene_index > 0:

        elapsed = (
            time_position
            - scene["start"]
        )

        if (
            0 <= elapsed < FADE
        ):

            previous_bg = (
                background_for_scene(
                    scene_index - 1
                )
            )

            bg = blended_background(
                previous_bg,
                current_bg,
                elapsed / FADE,
            )

    image = Image.new(
        "RGB",
        (W, H),
        bg,
    )

    draw = ImageDraw.Draw(
        image
    )

    active_indexes = {
        index
        for index, word in enumerate(
            scene["words"]
        )
        if (
            time_position
            >= float(word["start"])
            and
            time_position
            < float(word["end"]) + 0.08
        )
    }

    if not active_indexes:

        return image, False

    rows = scene["_layout"]

    brightness = (
        sum(bg) / 3.0
    )

    foreground = (
        BLACK
        if brightness > 128
        else WHITE
    )

    vertical_gap = 18

    total_height = (
        sum(
            max(
                item[4]
                for item in row
            )
            for row in rows
        )
        + vertical_gap
        * max(
            0,
            len(rows) - 1,
        )
    )

    y = max(
        80,
        (H - total_height) // 2,
    )

    for row in rows:

        gap = max(
            10,
            row[0][2].size // 8,
        )

        row_width = (
            sum(
                item[3]
                for item in row
            )
            + gap
            * max(
                0,
                len(row) - 1,
            )
        )

        x = max(
            24,
            (W - row_width) // 2,
        )

        row_height = max(
            item[4]
            for item in row
        )

        for local_index, item in enumerate(
            row
        ):

            (
                word_index,
                text,
                font,
                word_width,
                _,
            ) = item

            if (
                word_index
                not in active_indexes
            ):
                x += (
                    word_width
                    + gap
                )
                continue

            blue = (
                scene_index % 2 == 0
                and local_index == 1
                and len(row) >= 3
            )

            color = (
                ROYAL
                if blue
                else foreground
            )

            shadow = (
                (90, 90, 90)
                if foreground == WHITE
                else (220, 220, 220)
            )

            draw.text(
                (
                    x + 2,
                    y + 2,
                ),
                text,
                font=font,
                fill=shadow,
            )

            draw.text(
                (
                    x,
                    y,
                ),
                text,
                font=font,
                fill=color,
            )

            x += (
                word_width
                + gap
            )

        y += (
            row_height
            + vertical_gap
        )

    return image, True


# ============================================================
# Renderização
# ============================================================

def render_video(
    audio_path,
    scenes,
    output_path,
    status,
    progress,
):

    duration, has_audio = get_media_info(
        audio_path
    )

    if not has_audio:

        raise RuntimeError(
            "O arquivo de entrada não possui áudio."
        )

    if not scenes:

        raise RuntimeError(
            "Nenhuma legenda chegou à renderização."
        )

    # Calcula fontes e posições uma única vez.
    # Isso é importante para o Streamlit Cloud.
    for index, scene in enumerate(
        scenes
    ):

        scene["_layout"] = make_layout(
            scene["words"],
            index,
        )

    with tempfile.TemporaryDirectory(
        prefix="lyric_render_"
    ) as temp_dir:

        temp_dir = Path(
            temp_dir
        )

        silent_video = (
            temp_dir
            / "silent.mp4"
        )

        ffmpeg_command = [
            get_ffmpeg(),
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
            "veryfast",

            "-crf",
            "23",

            "-pix_fmt",
            "yuv420p",

            "-movflags",
            "+faststart",

            str(silent_video),
        ]

        process = subprocess.Popen(
            ffmpeg_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        frame_count = max(
            1,
            int(
                math.ceil(
                    duration * FPS
                )
            ),
        )

        text_frame_found = False
        scene_index = 0

        try:

            for frame_number in range(
                frame_count
            ):

                time_position = (
                    frame_number / FPS
                )

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

                image, has_text = draw_frame(
                    time_position,
                    scenes,
                    scene_index,
                )

                if has_text:

                    expected_bg = (
                        background_for_scene(
                            scene_index
                        )
                    )

                    uniform_bg = Image.new(
                        "RGB",
                        (W, H),
                        expected_bg,
                    )

                    difference = (
                        ImageChops.difference(
                            image,
                            uniform_bg,
                        )
                    )

                    if difference.getbbox():
                        text_frame_found = True

                process.stdin.write(
                    image.tobytes()
                )

                if (
                    progress
                    and frame_number % 20 == 0
                ):

                    percent = (
                        (frame_number + 1)
                        / frame_count
                        * 0.86
                    )

                    progress.progress(
                        min(
                            0.86,
                            percent,
                        ),
                        text=(
                            "Renderizando letras... "
                            f"{frame_number + 1}/"
                            f"{frame_count}"
                        ),
                    )

            process.stdin.close()

            stderr = (
                process.stderr.read()
                .decode(
                    "utf-8",
                    "replace",
                )
            )

            return_code = (
                process.wait()
            )

            if return_code != 0:

                raise RuntimeError(
                    stderr[-6000:]
                    or
                    "FFmpeg falhou ao codificar."
                )

        except Exception:

            try:
                process.stdin.close()
            except Exception:
                pass

            try:
                process.kill()
            except Exception:
                pass

            raise

        if not text_frame_found:

            raise RuntimeError(
                "VALIDAÇÃO DE TEXTO FALHOU: "
                "nenhum frame apresentou "
                "pixels de legenda. "
                "O MP4 não será entregue."
            )

        if (
            not silent_video.exists()
            or silent_video.stat().st_size
            < 10000
        ):

            raise RuntimeError(
                "O vídeo intermediário é inválido."
            )

        status.write(
            "🎬 Texto renderizado. "
            "Adicionando áudio..."
        )

        final_command = [
            get_ffmpeg(),
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
            "160k",

            "-t",
            f"{duration:.3f}",

            "-movflags",
            "+faststart",

            str(output_path),
        ]

        run_cmd(
            final_command,
            timeout=max(
                180,
                int(duration * 8),
            ),
        )

        if progress:

            progress.progress(
                1.0,
                text="Vídeo concluído.",
            )


# ============================================================
# Interface
# ============================================================

def main():

    st.set_page_config(
        page_title="Lyric AI Studio",
        page_icon="🎵",
        layout="centered",
    )

    st.title(
        "🎵 Lyric AI Studio"
    )

    st.caption(
        f"Stable Lite · {APP_VERSION} · "
        f"{W}×{H} · {FPS} FPS"
    )

    audio = st.file_uploader(
        "1. Música ou vídeo",
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
        "2. Letra oficial",
        height=220,
        placeholder=(
            "Recomendado:\n"
            "00:02.3 - 00:06.8\n"
            "Não tenho vergonha de dizer que sou maluco por você\n\n"
            "00:06.8 - 00:10.5\n"
            "É, sou maluco, deixa o mundo perceber"
        ),
    )

    model = st.selectbox(
        "Reconhecimento",
        [
            "small",
            "base",
            "tiny",
            "large-v3-turbo",
        ],
        index=0,
        help=(
            "small é o recomendado para "
            "o Streamlit Cloud gratuito. "
            "large-v3-turbo pode usar mais memória."
        ),
    )

    st.caption(
        "720 × 1280 · fontes locais · "
        "renderização sequencial."
    )

    if not st.button(
        "🚀 CRIAR LYRIC VIDEO",
        type="primary",
        use_container_width=True,
    ):
        return

    if not audio:

        st.error(
            "Envie a música primeiro."
        )

        return

    status = st.empty()

    progress = st.progress(
        0,
        text="Preparando...",
    )

    with tempfile.TemporaryDirectory(
        prefix="lyric_ai_"
    ) as temp_dir:

        temp_dir = Path(
            temp_dir
        )

        suffix = (
            Path(audio.name)
            .suffix
            .lower()
            or ".mp3"
        )

        input_path = (
            temp_dir
            / ("input" + suffix)
        )

        output_path = (
            temp_dir
            / "lyric_ai_final.mp4"
        )

        input_path.write_bytes(
            audio.getbuffer()
        )

        try:

            # ------------------------------------------------
            # Duração
            # ------------------------------------------------

            duration, has_audio = (
                get_media_info(
                    input_path
                )
            )

            if duration <= 0:

                raise RuntimeError(
                    "Duração inválida."
                )

            if not has_audio:

                raise RuntimeError(
                    "Nenhum áudio foi encontrado."
                )

            status.write(
                f"⏱️ Duração: "
                f"**{duration:.2f}s**"
            )

            progress.progress(
                0.03,
                text="Reconhecendo a letra...",
            )

            # ------------------------------------------------
            # Whisper
            # ------------------------------------------------

            (
                segments,
                words,
                language,
            ) = transcribe_audio(
                input_path,
                model,
                status,
            )

            # ------------------------------------------------
            # Letra
            # ------------------------------------------------

            timed_lyrics = (
                parse_timed_lyrics(
                    lyrics
                )
            )

            scenes = build_scenes(
                timed_lyrics,
                segments,
                words,
                duration,
            )

            if not scenes:

                raise RuntimeError(
                    "Nenhuma legenda válida foi criada. "
                    "O vídeo não será gerado sem texto."
                )

            scenes = [
                scene
                for scene in scenes
                if (
                    scene["words"]
                    and
                    scene["end"]
                    > scene["start"]
                )
            ]

            if not scenes:

                raise RuntimeError(
                    "As legendas ficaram sem palavras."
                )

            first_caption = (
                scenes[0]["start"]
            )

            last_caption = (
                scenes[-1]["end"]
            )

            total_caption_words = sum(
                len(scene["words"])
                for scene in scenes
            )

            # ------------------------------------------------
            # Diagnóstico
            # ------------------------------------------------

            with st.expander(
                "🔎 Diagnóstico",
                expanded=True,
            ):

                st.write(
                    f"Duração do áudio: "
                    f"**{duration:.2f}s**"
                )

                if segments:

                    st.write(
                        f"Fim da transcrição: "
                        f"**{segments[-1]['end']:.2f}s**"
                    )

                else:

                    st.write(
                        "Fim da transcrição: **0.00s**"
                    )

                st.write(
                    f"Quantidade de segmentos: "
                    f"**{len(segments)}**"
                )

                st.write(
                    f"Quantidade de palavras: "
                    f"**{len(words)}**"
                )

                st.write(
                    f"Primeira legenda: "
                    f"**{first_caption:.2f}s**"
                )

                st.write(
                    f"Última legenda: "
                    f"**{last_caption:.2f}s**"
                )

                st.write(
                    f"Resolução: "
                    f"**{W} × {H}**"
                )

                st.write(
                    f"Modelo utilizado: "
                    f"**{model}**"
                )

                st.write(
                    f"Idioma: "
                    f"**{language}**"
                )

                st.write(
                    "Legendas renderizadas: "
                    "**aguardando renderização**"
                )

                st.code(
                    "\n".join(
                        (
                            f"{scene['start']:.2f} - "
                            f"{scene['end']:.2f} | "
                            f"{scene['text']}"
                        )
                        for scene in scenes
                    )
                )

            progress.progress(
                0.08,
                text="Iniciando renderização...",
            )

            # ------------------------------------------------
            # Renderização
            # ------------------------------------------------

            render_video(
                input_path,
                scenes,
                output_path,
                status,
                progress,
            )

            # ------------------------------------------------
            # Validação final
            # ------------------------------------------------

            if not output_path.exists():

                raise RuntimeError(
                    "O MP4 final não existe."
                )

            file_size = (
                output_path.stat().st_size
            )

            if file_size < 10000:

                raise RuntimeError(
                    "O MP4 final é inválido."
                )

            (
                final_duration,
                final_has_audio,
            ) = get_media_info(
                output_path
            )

            if not final_has_audio:

                raise RuntimeError(
                    "O MP4 final não contém áudio."
                )

            if (
                final_duration
                < duration - 0.5
            ):

                raise RuntimeError(
                    f"A duração final "
                    f"({final_duration:.2f}s) "
                    f"ficou muito menor que "
                    f"a original "
                    f"({duration:.2f}s)."
                )

            if (
                total_caption_words <= 0
            ):

                raise RuntimeError(
                    "Validação final: "
                    "zero palavras renderizadas."
                )

            with st.expander(
                "✅ Validação final",
                expanded=True,
            ):

                st.write(
                    f"Legendas renderizadas: "
                    f"**{total_caption_words} palavras**"
                )

                st.write(
                    f"Frases renderizadas: "
                    f"**{len(scenes)}**"
                )

                st.write(
                    f"Duração final: "
                    f"**{final_duration:.2f}s**"
                )

                st.write(
                    "Áudio no MP4: **sim**"
                )

                st.write(
                    f"Tamanho: "
                    f"**{file_size / 1024 / 1024:.1f} MB**"
                )

            final_bytes = (
                output_path.read_bytes()
            )

            st.success(
                "✅ Vídeo criado com texto "
                "visível, áudio e duração validada."
            )

            st.video(
                final_bytes
            )

            st.download_button(
                "⬇️ BAIXAR MP4",
                final_bytes,
                "lyric_ai_final.mp4",
                "video/mp4",
                use_container_width=True,
            )

        except Exception as error:

            st.error(
                "❌ A geração falhou. "
                "Nenhum vídeo sem texto será entregue."
            )

            st.exception(error)


if __name__ == "__main__":
    main()