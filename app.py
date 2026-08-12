import os
import re
import math
import tempfile
import subprocess
from pathlib import Path

import streamlit as st
import imageio_ffmpeg
import numpy as np

from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager
import matplotlib


# ============================================================
# CONFIGURAÇÃO
# ============================================================

APP_NAME = "LYRIC AI"

WIDTH = 1080
HEIGHT = 1920
FPS = 30

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

TEMP_ROOT = Path(tempfile.gettempdir()) / "lyric_ai"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURAÇÃO STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Lyric AI",
    page_icon="🎵",
    layout="centered"
)

st.markdown(
    """
    <style>
        .stApp {
            background: #080808;
        }

        .title {
            text-align: center;
            font-size: 42px;
            font-weight: 900;
            margin-top: 15px;
        }

        .subtitle {
            text-align: center;
            opacity: .7;
            margin-bottom: 30px;
        }

        div.stButton > button {
            width: 100%;
            height: 55px;
            border-radius: 14px;
            font-size: 18px;
            font-weight: 800;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="title">🎵 LYRIC AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Crie seu lyric video automaticamente</div>',
    unsafe_allow_html=True
)


# ============================================================
# UTILIDADES
# ============================================================

def run_command(command, timeout=900):

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout
    )

    if result.returncode != 0:

        error = result.stderr.decode(
            "utf-8",
            errors="replace"
        )

        raise RuntimeError(
            error[-8000:]
        )

    return result.stdout


def ffnum(value):
    """
    Converte números para um formato seguro para o FFmpeg.
    """

    return f"{float(value):.3f}"


def safe_name(name):

    name = Path(name).stem

    name = re.sub(
        r"[^a-zA-Z0-9À-ÿ _-]",
        "",
        name
    )

    return name.strip() or "lyric_video"


def save_upload(uploaded, folder, name):

    folder = Path(folder)
    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    path = folder / name

    with open(path, "wb") as f:

        f.write(
            uploaded.getbuffer()
        )

    return path


# ============================================================
# DURAÇÃO
# ============================================================

def get_duration(path):

    command = [
        FFMPEG,
        "-i",
        str(path)
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    text = result.stderr.decode(
        "utf-8",
        errors="replace"
    )

    match = re.search(
        r"Duration:\s*(\d+):(\d+):([\d.]+)",
        text
    )

    if not match:

        raise RuntimeError(
            "Não foi possível determinar a duração do arquivo."
        )

    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))

    return (
        hours * 3600
        + minutes * 60
        + seconds
    )


# ============================================================
# ÁUDIO
# ============================================================

def convert_audio(input_path, output_path):

    command = [
        FFMPEG,
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
    ]

    run_command(command)


def extract_audio(video_path, output_path):

    command = [
        FFMPEG,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path)
    ]

    run_command(command)


# ============================================================
# ENERGIA DO ÁUDIO
# ============================================================

def calculate_energy(audio_path):

    try:

        import wave

        with wave.open(
            str(audio_path),
            "rb"
        ) as wav:

            rate = wav.getframerate()
            frames = wav.readframes(
                wav.getnframes()
            )

        audio = np.frombuffer(
            frames,
            dtype=np.int16
        ).astype(np.float32)

        if len(audio) == 0:

            return []

        audio /= 32768.0

        window = int(
            rate * 0.25
        )

        values = []

        for i in range(
            0,
            len(audio),
            window
        ):

            chunk = audio[
                i:i + window
            ]

            if len(chunk) == 0:
                continue

            rms = float(
                np.sqrt(
                    np.mean(
                        chunk ** 2
                    ) + 1e-9
                )
            )

            values.append(rms)

        if not values:

            return []

        values = np.array(
            values
        )

        low = np.percentile(
            values,
            10
        )

        high = np.percentile(
            values,
            90
        )

        if high <= low:

            return [
                0.5
                for _ in values
            ]

        normalized = np.clip(
            (
                values - low
            ) / (
                high - low
            ),
            0,
            1
        )

        return normalized.tolist()

    except Exception:

        return []


def energy_at(
    energy,
    time
):

    if not energy:

        return 0.5

    index = int(
        time / 0.25
    )

    index = max(
        0,
        min(
            index,
            len(energy) - 1
        )
    )

    return float(
        energy[index]
    )


# ============================================================
# FONTES
# ============================================================

@st.cache_resource
def get_font_library():

    paths = []

    try:

        paths.extend(
            font_manager.findSystemFonts(
                fontext="ttf"
            )
        )

    except Exception:
        pass

    try:

        paths.extend(
            font_manager.findSystemFonts(
                fontext="otf"
            )
        )

    except Exception:
        pass

    preferred = [
        "Inter",
        "Lato",
        "LiberationSans",
        "LiberationSerif",
        "DejaVuSansCondensed",
        "DejaVuSans",
        "DejaVuSerif",
        "DejaVuSansMono"
    ]

    selected = []

    for keyword in preferred:

        matches = [
            p
            for p in paths
            if keyword.lower()
            in Path(p).name.lower()
        ]

        matches.sort(
            key=lambda x: (
                "Bold" not in Path(x).name,
                len(Path(x).name)
            )
        )

        if matches:

            selected.append(
                matches[0]
            )

    # Fallback do próprio matplotlib.
    matplotlib_fonts = (
        Path(
            matplotlib.get_data_path()
        )
        / "fonts"
        / "ttf"
    )

    fallbacks = [
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
        "DejaVuSansCondensed-Bold.ttf",
        "DejaVuSerif-Bold.ttf",
        "DejaVuSerif.ttf",
        "DejaVuSansMono-Bold.ttf"
    ]

    for name in fallbacks:

        path = (
            matplotlib_fonts / name
        )

        if path.exists():

            selected.append(
                str(path)
            )

    unique = []

    for path in selected:

        if (
            os.path.exists(path)
            and path not in unique
        ):

            unique.append(path)

    if not unique:

        raise RuntimeError(
            "Nenhuma fonte compatível foi encontrada."
        )

    return unique[:10]


def get_font(index):

    fonts = get_font_library()

    return fonts[
        index % len(fonts)
    ]


# ============================================================
# WHISPER
# ============================================================

@st.cache_resource
def load_model(model_name):

    from faster_whisper import WhisperModel

    return WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
        num_workers=1
    )


def transcribe(
    audio_path,
    model_name
):

    model = load_model(
        model_name
    )

    segments, info = model.transcribe(
        str(audio_path),
        language="pt",
        task="transcribe",
        beam_size=5,
        word_timestamps=True,
        vad_filter=True
    )

    words = []

    for segment in segments:

        if segment.words:

            for word in segment.words:

                text = (
                    word.word
                    or ""
                ).strip()

                if not text:
                    continue

                start = float(
                    word.start
                )

                end = float(
                    word.end
                )

                if end <= start:

                    end = (
                        start + 0.15
                    )

                words.append(
                    {
                        "text": text,
                        "start": start,
                        "end": end
                    }
                )

        else:

            text = (
                segment.text
                or ""
            ).strip()

            if text:

                words.append(
                    {
                        "text": text,
                        "start": float(
                            segment.start
                        ),
                        "end": float(
                            segment.end
                        )
                    }
                )

    return words


# ============================================================
# FRASES
# ============================================================

def clean_text(text):

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def create_phrases(
    words,
    energy
):

    if not words:

        return []

    phrases = []

    current = []

    start = None
    previous_end = None

    for word in words:

        if start is None:

            start = word["start"]

        pause = 0

        if previous_end is not None:

            pause = (
                word["start"]
                - previous_end
            )

        duration = (
            word["end"]
            - start
        )

        should_break = False

        if current:

            if pause >= 0.42:

                should_break = True

            if len(current) >= 7:

                should_break = True

            if duration >= 3.2:

                should_break = True

        if should_break:

            text = clean_text(
                " ".join(
                    w["text"]
                    for w in current
                )
            )

            if text:

                p_start = (
                    current[0]["start"]
                )

                p_end = (
                    current[-1]["end"]
                )

                phrases.append(
                    {
                        "text": text,
                        "start": p_start,
                        "end": p_end,
                        "intensity":
                            energy_at(
                                energy,
                                (
                                    p_start
                                    + p_end
                                ) / 2
                            )
                    }
                )

            current = []

            start = word["start"]

        current.append(word)

        previous_end = word["end"]

    if current:

        text = clean_text(
            " ".join(
                w["text"]
                for w in current
            )
        )

        if text:

            p_start = (
                current[0]["start"]
            )

            p_end = (
                current[-1]["end"]
            )

            phrases.append(
                {
                    "text": text,
                    "start": p_start,
                    "end": p_end,
                    "intensity":
                        energy_at(
                            energy,
                            (
                                p_start
                                + p_end
                            ) / 2
                        )
                }
            )

    return phrases


def manual_phrases(
    lyrics,
    duration,
    energy
):

    lines = [
        clean_text(x)
        for x in lyrics.splitlines()
        if clean_text(x)
    ]

    if not lines:

        return []

    weights = [
        max(
            1,
            len(line)
        )
        for line in lines
    ]

    total = sum(weights)

    result = []

    cursor = 0

    for i, line in enumerate(lines):

        length = (
            duration
            * weights[i]
            / total
        )

        start = cursor
        end = min(
            duration,
            cursor + length
        )

        result.append(
            {
                "text": line,
                "start": start,
                "end": end,
                "intensity":
                    energy_at(
                        energy,
                        (
                            start
                            + end
                        ) / 2
                    )
            }
        )

        cursor = end

    return result


# ============================================================
# TEXTO PNG
# ============================================================

def wrap_text(
    draw,
    words,
    font,
    max_width
):

    lines = []
    current = []

    for word in words:

        candidate = (
            " ".join(
                current + [word]
            )
        )

        box = draw.textbbox(
            (0, 0),
            candidate,
            font=font
        )

        width = (
            box[2]
            - box[0]
        )

        if (
            width <= max_width
            or not current
        ):

            current.append(
                word
            )

        else:

            lines.append(
                " ".join(current)
            )

            current = [word]

    if current:

        lines.append(
            " ".join(current)
        )

    return lines


def create_text_image(
    text,
    intensity,
    font_index,
    output
):

    font_path = get_font(
        font_index
    )

    if not font_path:

        raise RuntimeError(
            "Fonte inválida."
        )

    size = 76

    if intensity >= 0.8:

        size = 94

    elif intensity >= 0.6:

        size = 86

    if len(text) <= 18:

        size += 8

    font = ImageFont.truetype(
        font_path,
        size
    )

    image = Image.new(
        "RGBA",
        (1000, 500),
        (0, 0, 0, 0)
    )

    draw = ImageDraw.Draw(
        image
    )

    words = text.split()

    lines = wrap_text(
        draw,
        words,
        font,
        900
    )

    line_height = int(
        size * 1.18
    )

    total_height = (
        len(lines)
        * line_height
    )

    y = (
        500
        - total_height
    ) // 2

    # Palavra de destaque.
    useful = [
        w
        for w in words
        if len(
            re.sub(
                r"\W",
                "",
                w
            )
        ) >= 5
    ]

    if useful:

        highlight = max(
            useful,
            key=len
        )

    else:

        highlight = (
            words[-1]
            if words
            else ""
        )

    accent = [
        (255, 255, 255, 255),
        (245, 215, 150, 255),
        (255, 230, 190, 255),
        (220, 230, 255, 255)
    ][
        font_index % 4
    ]

    for line in lines:

        box = draw.textbbox(
            (0, 0),
            line,
            font=font
        )

        line_width = (
            box[2]
            - box[0]
        )

        x = (
            1000
            - line_width
        ) / 2

        for word in line.split():

            word_box = draw.textbbox(
                (0, 0),
                word,
                font=font
            )

            word_width = (
                word_box[2]
                - word_box[0]
            )

            is_highlight = (
                word.strip(
                    ".,!?;:"
                ).lower()
                ==
                highlight.strip(
                    ".,!?;:"
                ).lower()
            )

            # Sombra.
            draw.text(
                (
                    x + 5,
                    y + 6
                ),
                word,
                font=font,
                fill=(0, 0, 0, 220),
                stroke_width=7,
                stroke_fill=(0, 0, 0, 220)
            )

            # Texto.
            draw.text(
                (
                    x,
                    y
                ),
                word,
                font=font,
                fill=(
                    accent
                    if is_highlight
                    else (255, 255, 255, 255)
                ),
                stroke_width=3,
                stroke_fill=(0, 0, 0, 200)
            )

            space = draw.textbbox(
                (0, 0),
                " ",
                font=font
            )[2]

            x += (
                word_width
                + space
            )

        y += line_height

    bbox = image.getbbox()

    if bbox:

        image = image.crop(
            (
                max(
                    0,
                    bbox[0] - 30
                ),
                max(
                    0,
                    bbox[1] - 30
                ),
                min(
                    1000,
                    bbox[2] + 30
                ),
                min(
                    500,
                    bbox[3] + 30
                )
            )
        )

    image.save(
        output,
        "PNG"
    )


# ============================================================
# RENDERIZAÇÃO
# ============================================================

def render_video(
    audio_path,
    background_path,
    phrases,
    workdir,
    output
):

    workdir = Path(
        workdir
    )

    output = Path(
        output
    )

    duration = get_duration(
        audio_path
    )

    fonts = get_font_library()

    # --------------------------------------------------------
    # Gera imagens das frases
    # --------------------------------------------------------

    text_files = []

    for i, phrase in enumerate(
        phrases
    ):

        path = (
            workdir
            / f"text_{i:04d}.png"
        )

        create_text_image(
            phrase["text"],
            phrase["intensity"],
            i % len(fonts),
            path
        )

        text_files.append(
            path
        )

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    command = [
        FFMPEG,
        "-y"
    ]

    if background_path:

        command += [
            "-stream_loop",
            "-1",
            "-i",
            str(background_path)
        ]

        audio_index = 1
        first_text_index = 2

    else:

        # Fundo sólido.
        #
        # IMPORTANTE:
        # Não usamos GEQ.
        # Isso elimina o erro que apareceu no seu teste.
        command += [
            "-f",
            "lavfi",
            "-i",
            f"color=c=0x101018:s={WIDTH}x{HEIGHT}:r={FPS}"
        ]

        audio_index = 1
        first_text_index = 2

    command += [
        "-i",
        str(audio_path)
    ]

    # Inputs PNG.
    for i, text_path in enumerate(
        text_files
    ):

        phrase_duration = max(
            0.25,
            phrase["end"]
            - phrase["start"]
        )

        command += [
            "-loop",
            "1",
            "-t",
            ffnum(
                phrase_duration
                + 0.4
            ),
            "-i",
            str(text_path)
        ]

    # --------------------------------------------------------
    # FILTER COMPLEX
    # --------------------------------------------------------

    filters = []

    if background_path:

        filters.append(
            "[0:v]"
            f"scale={WIDTH}:{HEIGHT}:"
            "force_original_aspect_ratio=increase,"
            f"crop={WIDTH}:{HEIGHT},"
            "setsar=1,"
            "eq=brightness=-0.10:saturation=0.82"
            "[bg]"
        )

    else:

        filters.append(
            "[0:v]"
            "format=yuv420p"
            "[bg]"
        )

    current = "bg"

    for i, phrase in enumerate(
        phrases
    ):

        input_index = (
            first_text_index + i
        )

        duration_phrase = max(
            0.25,
            phrase["end"]
            - phrase["start"]
        )

        fade = min(
            0.16,
            duration_phrase / 4
        )

        fade_out_start = max(
            0,
            duration_phrase - fade
        )

        # A escala agora é FIXA.
        #
        # Isso é intencional:
        # removemos outra fonte potencial
        # de incompatibilidade do FFmpeg.
        filters.append(
            f"[{input_index}:v]"
            "format=rgba,"
            f"fade=t=in:st=0:d={ffnum(fade)}:alpha=1,"
            f"fade=t=out:"
            f"st={ffnum(fade_out_start)}:"
            f"d={ffnum(fade)}:alpha=1,"
            "setpts=PTS-STARTPTS"
            f"[txt{i}]"
        )

        # Posições alternadas.
        position = i % 3

        if position == 0:

            y = "(H-h)/2-130"

        elif position == 1:

            y = "(H-h)/2"

        else:

            y = "(H-h)/2+110"

        # Pequeno movimento horizontal.
        x = (
            "(W-w)/2"
            "+12*sin(2*PI*t/1.7)"
        )

        output_label = (
            f"v{i}"
        )

        # IMPORTANTE:
        # A expressão between é mantida
        # entre aspas simples.
        filters.append(
            f"[{current}]"
            f"[txt{i}]"
            f"overlay="
            f"x='{x}':"
            f"y='{y}':"
            "eval=frame:"
            f"enable='between(t,"
            f"{ffnum(phrase['start'])},"
            f"{ffnum(phrase['end'])})'"
            f"[{output_label}]"
        )

        current = output_label

    filter_complex = ";".join(
        filters
    )

    # --------------------------------------------------------
    # COMANDO FINAL
    # --------------------------------------------------------

    command += [
        "-filter_complex",
        filter_complex,

        "-map",
        f"[{current}]",

        "-map",
        f"{audio_index}:a:0",

        "-t",
        ffnum(duration),

        "-r",
        str(FPS),

        "-c:v",
        "libx264",

        "-preset",
        "veryfast",

        "-crf",
        "21",

        "-pix_fmt",
        "yuv420p",

        "-c:a",
        "aac",

        "-b:a",
        "192k",

        "-movflags",
        "+faststart",

        "-shortest",

        str(output)
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode != 0:

        error = result.stderr.decode(
            "utf-8",
            errors="replace"
        )

        raise RuntimeError(
            "FFmpeg falhou durante a renderização.\n\n"
            + error[-10000:]
        )

    if (
        not output.exists()
        or output.stat().st_size < 10000
    ):

        raise RuntimeError(
            "O FFmpeg terminou, mas o MP4 não foi criado corretamente."
        )

    return output


# ============================================================
# INTERFACE
# ============================================================

st.subheader("1. Música")

audio_upload = st.file_uploader(
    "Envie sua música",
    type=[
        "mp3",
        "wav",
        "m4a",
        "aac",
        "flac",
        "ogg"
    ]
)

st.subheader("2. Vídeo de fundo — opcional")

background_upload = st.file_uploader(
    "Envie um vídeo de show ou outro fundo",
    type=[
        "mp4",
        "mov",
        "m4v",
        "webm"
    ]
)

rights_confirmed = False

if background_upload:

    rights_confirmed = st.checkbox(
        "Confirmo que tenho autorização para utilizar este vídeo."
    )

st.subheader("3. Letras")

lyrics_mode = st.radio(
    "Como obter a letra?",
    [
        "IA transcreve automaticamente",
        "Eu vou fornecer a letra"
    ]
)

manual_lyrics = ""

if lyrics_mode == "Eu vou fornecer a letra":

    manual_lyrics = st.text_area(
        "Cole a letra abaixo",
        height=220
    )

st.subheader("4. Modelo")

model = st.selectbox(
    "Modelo de transcrição",
    [
        "tiny",
        "base"
    ]
)

st.info(
    "A fonte, tamanho, posição e destaque das letras "
    "são escolhidos automaticamente."
)

generate = st.button(
    "🚀 CRIAR LYRIC VIDEO",
    type="primary"
)


# ============================================================
# EXECUÇÃO
# ============================================================

if generate:

    if audio_upload is None and background_upload is None:

        st.error(
            "Envie uma música ou um vídeo."
        )

        st.stop()

    if (
        background_upload
        and not rights_confirmed
    ):

        st.error(
            "Confirme que possui autorização para utilizar o vídeo."
        )

        st.stop()

    session = (
        TEMP_ROOT
        / next(
            tempfile._get_candidate_names()
        )
    )

    session.mkdir(
        parents=True,
        exist_ok=True
    )

    try:

        with st.status(
            "Criando seu vídeo...",
            expanded=True
        ) as status:

            # ------------------------------------------------
            # Arquivos
            # ------------------------------------------------

            st.write(
                "📦 Preparando arquivos..."
            )

            audio_path = None
            background_path = None

            if audio_upload:

                original_audio = save_upload(
                    audio_upload,
                    session,
                    "audio_original"
                    + Path(
                        audio_upload.name
                    ).suffix
                )

                audio_path = (
                    session
                    / "audio.wav"
                )

                convert_audio(
                    original_audio,
                    audio_path
                )

            if background_upload:

                background_path = save_upload(
                    background_upload,
                    session,
                    "background"
                    + Path(
                        background_upload.name
                    ).suffix
                )

            # Se o usuário mandou apenas um vídeo.
            if (
                audio_path is None
                and background_path is not None
            ):

                st.write(
                    "🎧 Extraindo áudio do vídeo..."
                )

                audio_path = (
                    session
                    / "audio.wav"
                )

                extract_audio(
                    background_path,
                    audio_path
                )

            duration = get_duration(
                audio_path
            )

            st.write(
                "⏱️ Duração: "
                + f"{duration:.1f}s"
            )

            # ------------------------------------------------
            # Áudio
            # ------------------------------------------------

            st.write(
                "🎚️ Analisando intensidade..."
            )

            energy = calculate_energy(
                audio_path
            )

            # ------------------------------------------------
            # Transcrição
            # ------------------------------------------------

            if (
                lyrics_mode
                == "Eu vou fornecer a letra"
            ):

                if not manual_lyrics.strip():

                    raise RuntimeError(
                        "Você não forneceu nenhuma letra."
                    )

                st.write(
                    "📝 Organizando a letra..."
                )

                phrases = manual_phrases(
                    manual_lyrics,
                    duration,
                    energy
                )

            else:

                st.write(
                    "🤖 Transcrevendo a música..."
                )

                words = transcribe(
                    audio_path,
                    model
                )

                if not words:

                    raise RuntimeError(
                        "Nenhuma letra foi detectada."
                    )

                phrases = create_phrases(
                    words,
                    energy
                )

            if not phrases:

                raise RuntimeError(
                    "Não foi possível criar as frases."
                )

            st.write(
                f"📝 {len(phrases)} frases encontradas."
            )

            # ------------------------------------------------
            # Renderização
            # ------------------------------------------------

            st.write(
                "🎨 Criando visual..."
            )

            output = (
                session
                / "lyric_video.mp4"
            )

            st.write(
                "🎬 Renderizando MP4..."
            )

            render_video(
                audio_path,
                background_path,
                phrases,
                session,
                output
            )

            status.update(
                label="Vídeo pronto!",
                state="complete",
                expanded=False
            )

        # ----------------------------------------------------
        # RESULTADO
        # ----------------------------------------------------

        st.success(
            "🎉 Seu lyric video foi criado!"
        )

        st.subheader(
            "▶️ Resultado"
        )

        st.video(
            str(output)
        )

        with open(
            output,
            "rb"
        ) as f:

            video_bytes = f.read()

        filename = (
            safe_name(
                audio_upload.name
                if audio_upload
                else background_upload.name
            )
            + "_LYRIC_AI.mp4"
        )

        st.download_button(
            "⬇️ BAIXAR VÍDEO",
            video_bytes,
            file_name=filename,
            mime="video/mp4",
            type="primary"
        )

        st.divider()

        st.write(
            "### 🧠 Análise automática"
        )

        st.write(
            f"• Frases: **{len(phrases)}**"
        )

        st.write(
            "• Resolução: **1080 × 1920**"
        )

        st.write(
            "• Formato: **9:16**"
        )

        st.write(
            "• Fonte: **automática**"
        )

        st.write(
            "• Destaques: **automáticos**"
        )

        st.write(
            "• Entrada/saída: **fade automático**"
        )

        st.write(
            "• Fundo: "
            + (
                "**vídeo enviado por você**"
                if background_path
                else "**fundo gerado pelo programa**"
            )
        )

    except Exception as error:

        st.error(
            "❌ A renderização falhou."
        )

        with st.expander(
            "Detalhes técnicos"
        ):

            st.code(
                str(error),
                language="text"
            )