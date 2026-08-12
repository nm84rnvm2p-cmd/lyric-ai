import os
import re
import math
import wave
import shutil
import tempfile
import subprocess
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import streamlit as st
import imageio_ffmpeg

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

TEMP_ROOT = Path(tempfile.gettempdir()) / "lyric_ai"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


# ============================================================
# MODELOS
# ============================================================

@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Phrase:
    text: str
    start: float
    end: float
    intensity: float = 0.5
    style_index: int = 0


# ============================================================
# INTERFACE
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

        .main-title {
            font-size: 42px;
            font-weight: 900;
            text-align: center;
            margin-top: 10px;
            margin-bottom: 0;
        }

        .subtitle {
            text-align: center;
            opacity: 0.72;
            margin-bottom: 30px;
        }

        div.stButton > button {
            width: 100%;
            border-radius: 14px;
            height: 52px;
            font-size: 18px;
            font-weight: 800;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">🎵 LYRIC AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Criação automática de lyric videos verticais</div>',
    unsafe_allow_html=True
)


# ============================================================
# UTILIDADES
# ============================================================

def run_cmd(command, timeout=None):
    """
    Executa um comando e devolve stdout.
    Se falhar, lança erro contendo stderr.
    """
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout
    )

    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(error[-6000:])

    return result.stdout


def safe_filename(name):
    name = Path(name).stem
    name = re.sub(r"[^a-zA-Z0-9À-ÿ _-]", "", name)
    name = name.strip()
    return name or "lyric_video"


def save_uploaded_file(uploaded, folder, filename=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = uploaded.name

    path = folder / filename

    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())

    return path


def ffprobe_duration(path):
    """
    Descobre a duração de um arquivo usando o ffprobe
    correspondente ao FFmpeg disponibilizado pelo imageio-ffmpeg.
    """
    ffprobe = FFMPEG.replace("ffmpeg", "ffprobe")

    if not Path(ffprobe).exists():
        # fallback: usa ffmpeg
        command = [
            FFMPEG,
            "-i",
            str(path)
        ]

        try:
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

            if match:
                h = int(match.group(1))
                m = int(match.group(2))
                s = float(match.group(3))
                return h * 3600 + m * 60 + s

        except Exception:
            pass

    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        value = result.stdout.decode().strip()

        if value:
            return float(value)

    except Exception:
        pass

    raise RuntimeError(
        "Não consegui descobrir a duração do arquivo."
    )


# ============================================================
# ÁUDIO
# ============================================================

def extract_audio_from_video(video_path, output_path):
    """
    Extrai o áudio de um vídeo para WAV PCM.
    """
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

    run_cmd(command, timeout=600)


def normalize_audio(input_path, output_path):
    """
    Converte qualquer áudio para WAV PCM mono 16 kHz.
    """
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

    run_cmd(command, timeout=600)


def calculate_audio_energy(audio_path):
    """
    Calcula energia média do áudio em pequenas janelas.
    Serve para o sistema decidir onde aumentar o impacto visual.
    """
    try:
        with wave.open(str(audio_path), "rb") as wav:
            rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())

        audio = np.frombuffer(
            frames,
            dtype=np.int16
        ).astype(np.float32)

        if len(audio) == 0:
            return []

        audio /= 32768.0

        window = max(1, int(rate * 0.25))

        values = []

        for i in range(0, len(audio), window):
            chunk = audio[i:i + window]

            if len(chunk) == 0:
                continue

            rms = float(
                np.sqrt(
                    np.mean(
                        np.square(chunk)
                    ) + 1e-9
                )
            )

            values.append(rms)

        if not values:
            return []

        arr = np.array(values)

        low = np.percentile(arr, 10)
        high = np.percentile(arr, 90)

        if high <= low:
            return [0.5] * len(arr)

        normalized = np.clip(
            (arr - low) / (high - low),
            0,
            1
        )

        return normalized.tolist()

    except Exception:
        return []


def energy_at_time(energy, time_sec):
    if not energy:
        return 0.5

    index = int(time_sec / 0.25)

    index = max(
        0,
        min(index, len(energy) - 1)
    )

    return float(energy[index])


# ============================================================
# FONTES
# ============================================================

@st.cache_resource
def build_font_library():
    """
    Constrói uma biblioteca local de fontes.

    A lógica NÃO depende de uma única fonte.
    Se algumas não existirem, outras são utilizadas.

    O matplotlib também possui fontes próprias, garantindo
    um fallback mesmo em ambientes muito restritos.
    """

    candidates = []

    system_fonts = []

    try:
        system_fonts = font_manager.findSystemFonts(
            fontpaths=None,
            fontext="ttf"
        )
    except Exception:
        system_fonts = []

    try:
        system_fonts += font_manager.findSystemFonts(
            fontpaths=None,
            fontext="otf"
        )
    except Exception:
        pass

    # Fontes procuradas primeiro.
    preferred_keywords = [
        "Inter",
        "Lato",
        "LiberationSans",
        "LiberationSerif",
        "DejaVuSansCondensed",
        "DejaVuSans",
        "DejaVuSerif",
        "DejaVuSansMono",
        "STIXGeneral"
    ]

    for keyword in preferred_keywords:
        matches = [
            p for p in system_fonts
            if keyword.lower() in Path(p).name.lower()
        ]

        # Dá preferência às versões fortes.
        matches.sort(
            key=lambda p: (
                "Bold" not in Path(p).name,
                "Medium" not in Path(p).name,
                len(Path(p).name)
            )
        )

        if matches:
            candidates.append(matches[0])

    # Fallback garantido usando as fontes do matplotlib.
    mpl_font_dir = (
        Path(matplotlib.get_data_path())
        / "fonts"
        / "ttf"
    )

    fallback_names = [
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
        "DejaVuSansCondensed-Bold.ttf",
        "DejaVuSerif-Bold.ttf",
        "DejaVuSerif.ttf",
        "DejaVuSansMono-Bold.ttf",
        "STIXGeneral-Bold.ttf",
        "STIXGeneral-Italic.ttf",
    ]

    for name in fallback_names:
        p = mpl_font_dir / name

        if p.exists():
            candidates.append(str(p))

    # Remove duplicatas.
    unique = []

    for p in candidates:
        p = str(p)

        if os.path.exists(p) and p not in unique:
            unique.append(p)

    # Caso extremo: uma única fonte fallback.
    if not unique:
        raise RuntimeError(
            "Não foi possível localizar nenhuma fonte compatível."
        )

    # Mantém no máximo 10.
    return unique[:10]


def get_font(font_index):
    fonts = build_font_library()

    if not fonts:
        raise RuntimeError(
            "Biblioteca de fontes vazia."
        )

    index = int(font_index) % len(fonts)

    path = fonts[index]

    if not os.path.exists(path):
        path = fonts[0]

    return path


# ============================================================
# TRANSCRIÇÃO
# ============================================================

@st.cache_resource(show_spinner=False)
def load_whisper(model_size):
    from faster_whisper import WhisperModel

    return WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
        num_workers=1
    )


def transcribe_audio(audio_path, model_size="tiny"):
    """
    Transcrição com timestamps por palavra.
    """

    model = load_whisper(model_size)

    segments, info = model.transcribe(
        str(audio_path),
        language="pt",
        task="transcribe",
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=True
    )

    words = []

    for segment in segments:

        if segment.words:

            for word in segment.words:

                text = (word.word or "").strip()

                if not text:
                    continue

                start = float(word.start)
                end = float(word.end)

                if end <= start:
                    end = start + 0.15

                words.append(
                    Word(
                        text=text,
                        start=start,
                        end=end
                    )
                )

        else:

            text = (segment.text or "").strip()

            if text:
                words.append(
                    Word(
                        text=text,
                        start=float(segment.start),
                        end=float(segment.end)
                    )
                )

    return words, info


# ============================================================
# FRASES
# ============================================================

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def create_phrases(words, energy):
    """
    Agrupa palavras em frases visualmente adequadas.

    A IA tenta evitar:
    - frases enormes
    - frases rápidas demais
    - blocos estáticos por muito tempo

    Também considera pausas naturais.
    """

    if not words:
        return []

    phrases = []

    current = []
    current_start = None
    previous_end = None

    for word in words:

        if current_start is None:
            current_start = word.start

        pause = 0

        if previous_end is not None:
            pause = word.start - previous_end

        current_duration = word.end - current_start

        should_break = False

        if current:
            if pause >= 0.42:
                should_break = True

            if len(current) >= 7:
                should_break = True

            if current_duration >= 3.25:
                should_break = True

            if current[-1].text.endswith(
                (".", "!", "?", ",", ";", ":")
            ):
                if current_duration >= 1.0:
                    should_break = True

        if should_break:

            text = clean_text(
                " ".join(w.text for w in current)
            )

            if text:
                start = current[0].start
                end = current[-1].end

                intensity = (
                    energy_at_time(
                        energy,
                        (start + end) / 2
                    )
                )

                phrases.append(
                    Phrase(
                        text=text,
                        start=start,
                        end=end,
                        intensity=intensity
                    )
                )

            current = []
            current_start = word.start

        current.append(word)
        previous_end = word.end

    if current:

        text = clean_text(
            " ".join(w.text for w in current)
        )

        if text:

            start = current[0].start
            end = current[-1].end

            intensity = energy_at_time(
                energy,
                (start + end) / 2
            )

            phrases.append(
                Phrase(
                    text=text,
                    start=start,
                    end=end,
                    intensity=intensity
                )
            )

    return phrases


def create_phrases_from_manual_lyrics(
    lyrics,
    duration,
    energy
):
    """
    Fallback para quem já possui a letra.
    O tempo é distribuído proporcionalmente.
    """

    lines = [
        clean_text(x)
        for x in lyrics.splitlines()
        if clean_text(x)
    ]

    if not lines:
        return []

    weights = []

    for line in lines:
        weights.append(
            max(
                1.0,
                len(line)
            )
        )

    total = sum(weights)

    phrases = []

    cursor = 0.0

    for index, line in enumerate(lines):

        segment_duration = (
            duration * weights[index] / total
        )

        start = cursor
        end = min(
            duration,
            cursor + segment_duration
        )

        phrases.append(
            Phrase(
                text=line,
                start=start,
                end=end,
                intensity=energy_at_time(
                    energy,
                    (start + end) / 2
                )
            )
        )

        cursor = end

    return phrases


# ============================================================
# ESTILO AUTÔNOMO
# ============================================================

def choose_style(phrase, index, total):
    """
    Decide automaticamente o tratamento visual.

    Não é uma seleção manual:
    intensidade + comprimento + posição na música
    influenciam a decisão.
    """

    intensity = phrase.intensity

    length = len(phrase.text)

    if intensity >= 0.78:
        style = 0
    elif intensity >= 0.60:
        style = 1
    elif length <= 18:
        style = 2
    elif index % 5 == 0:
        style = 3
    elif index % 7 == 0:
        style = 4
    else:
        style = 5

    phrase.style_index = style

    return style


# ============================================================
# TEXTO
# ============================================================

def wrap_words(draw, words, font, max_width):
    """
    Quebra o texto em linhas sem ultrapassar o limite.
    """

    lines = []
    current = []

    for word in words:

        test = (
            " ".join(current + [word])
        )

        box = draw.textbbox(
            (0, 0),
            test,
            font=font,
            stroke_width=0
        )

        width = box[2] - box[0]

        if (
            width <= max_width
            or not current
        ):
            current.append(word)

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


def choose_accent_word(words):
    """
    Escolhe uma palavra de destaque.
    """

    if not words:
        return ""

    # Evita dar destaque a palavras muito pequenas.
    meaningful = [
        w for w in words
        if len(re.sub(r"\W", "", w)) >= 5
    ]

    if not meaningful:
        return words[-1]

    # Palavras mais longas costumam ser mais expressivas
    # visualmente.
    return max(
        meaningful,
        key=lambda x: len(x)
    )


def make_text_image(
    text,
    font_path,
    intensity,
    style_index,
    output_path
):
    """
    Cria PNG transparente da frase.

    Isso evita completamente a dependência do
    ImageFont.truetype(None) que causava o erro anterior.
    """

    # Segurança absoluta.
    if not font_path:
        font_path = get_font(0)

    if not os.path.exists(font_path):
        font_path = get_font(0)

    # Tamanho autônomo.
    if intensity >= 0.80:
        size = 92
    elif intensity >= 0.60:
        size = 84
    else:
        size = 76

    if len(text) <= 18:
        size += 8

    font = ImageFont.truetype(
        font_path,
        size=size
    )

    words = text.split()

    # Tela de texto transparente.
    canvas_width = 1000
    canvas_height = 500

    image = Image.new(
        "RGBA",
        (canvas_width, canvas_height),
        (0, 0, 0, 0)
    )

    draw = ImageDraw.Draw(image)

    lines = wrap_words(
        draw,
        words,
        font,
        900
    )

    # Altura total.
    line_height = int(size * 1.18)

    total_height = (
        len(lines) * line_height
    )

    y = (
        canvas_height - total_height
    ) // 2

    accent_word = choose_accent_word(words)

    # Paleta simples para manter a aparência elegante.
    normal_color = (255, 255, 255, 255)

    accent_colors = [
        (255, 255, 255, 255),
        (245, 215, 150, 255),
        (255, 230, 190, 255),
        (220, 230, 255, 255)
    ]

    accent_color = accent_colors[
        style_index % len(accent_colors)
    ]

    for line in lines:

        line_words = line.split()

        # Medidas individuais.
        measurements = []

        total_width = 0

        for word in line_words:

            box = draw.textbbox(
                (0, 0),
                word,
                font=font,
                stroke_width=0
            )

            word_width = (
                box[2] - box[0]
            )

            space_box = draw.textbbox(
                (0, 0),
                " ",
                font=font
            )

            space_width = (
                space_box[2] - space_box[0]
            )

            measurements.append(
                (
                    word,
                    word_width,
                    space_width
                )
            )

            total_width += (
                word_width
                + space_width
            )

        x = (
            canvas_width - total_width
        ) / 2

        for word, word_width, space_width in measurements:

            is_accent = (
                word.lower().strip(
                    ".,!?;:"
                )
                ==
                accent_word.lower().strip(
                    ".,!?;:"
                )
            )

            # Sombra forte para leitura sobre qualquer fundo.
            draw.text(
                (x + 5, y + 6),
                word,
                font=font,
                fill=(0, 0, 0, 210),
                stroke_width=7,
                stroke_fill=(0, 0, 0, 220)
            )

            draw.text(
                (x, y),
                word,
                font=font,
                fill=(
                    accent_color
                    if is_accent
                    else normal_color
                ),
                stroke_width=3,
                stroke_fill=(0, 0, 0, 190)
            )

            x += (
                word_width
                + space_width
            )

        y += line_height

    # Remove espaços vazios excessivos.
    bbox = image.getbbox()

    if bbox:
        image = image.crop(
            (
                max(0, bbox[0] - 35),
                max(0, bbox[1] - 35),
                min(
                    image.width,
                    bbox[2] + 35
                ),
                min(
                    image.height,
                    bbox[3] + 35
                )
            )
        )

    image.save(
        output_path,
        "PNG"
    )

    return output_path


# ============================================================
# BACKGROUND
# ============================================================

def build_background_filter(duration):
    """
    Fundo abstrato original.

    Não utiliza imagens de terceiros.
    """

    # Gradiente animado.
    return (
        "nullsrc="
        f"s={WIDTH}x{HEIGHT}:"
        f"r={FPS},"
        "geq="
        "r='12+12*sin(X/170+N/40)+"
        "8*sin(Y/300-N/55)',"
        "g='8+8*sin(X/240-N/50)+"
        "7*sin(Y/260+N/70)',"
        "b='18+18*sin(X/300+N/70)+"
        "12*sin(Y/200-N/60)',"
        "format=yuv420p,"
        "vignette=PI/4"
    )


def build_video_background_filter():
    """
    Prepara vídeo enviado pelo usuário para 9:16.
    """

    return (
        "[0:v]"
        f"scale={WIDTH}:{HEIGHT}:"
        "force_original_aspect_ratio=increase,"
        f"crop={WIDTH}:{HEIGHT},"
        "setsar=1,"
        "eq=brightness=-0.10:"
        "saturation=0.82,"
        "format=yuv420p"
        "[bg]"
    )


# ============================================================
# RENDER
# ============================================================

def render_video(
    audio_path,
    background_path,
    phrases,
    workdir,
    output_path
):
    """
    Renderiza o vídeo final.

    Usa FFmpeg diretamente.
    Não usa MoviePy.
    """

    workdir = Path(workdir)
    output_path = Path(output_path)

    duration = ffprobe_duration(
        audio_path
    )

    duration = max(
        1.0,
        float(duration)
    )

    # --------------------------------------------------------
    # Geração das imagens de texto
    # --------------------------------------------------------

    text_files = []

    fonts = build_font_library()

    for i, phrase in enumerate(phrases):

        style = choose_style(
            phrase,
            i,
            len(phrases)
        )

        font_path = fonts[
            style % len(fonts)
        ]

        text_path = (
            workdir
            / f"text_{i:04d}.png"
        )

        make_text_image(
            phrase.text,
            font_path,
            phrase.intensity,
            style,
            text_path
        )

        text_files.append(
            text_path
        )

    # --------------------------------------------------------
    # Construção dos inputs
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

        audio_index = 0
        first_text_index = 1

    command += [
        "-i",
        str(audio_path)
    ]

    # Cada PNG é transformado em vídeo temporário.
    for i, text_path in enumerate(text_files):

        phrase = phrases[i]

        phrase_duration = max(
            0.25,
            phrase.end - phrase.start
        )

        command += [
            "-loop",
            "1",
            "-t",
            str(
                phrase_duration + 0.25
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
            build_video_background_filter()
        )

    else:

        filters.append(
            build_background_filter(
                duration
            )
            + "[bg]"
        )

    current = "bg"

    for i, phrase in enumerate(phrases):

        input_index = (
            first_text_index + i
        )

        text_label = f"txt{i}"

        output_label = f"v{i}"

        phrase_duration = max(
            0.25,
            phrase.end - phrase.start
        )

        fade = min(
            0.16,
            phrase_duration / 4
        )

        # Movimento pequeno para não deixar
        # as letras completamente estáticas.
        motion_x = (
            f"(W-w)/2+"
            f"12*sin(2*PI*t/1.7)"
        )

        if i % 3 == 0:
            base_y = "(H-h)/2-130"
        elif i % 3 == 1:
            base_y = "(H-h)/2"
        else:
            base_y = "(H-h)/2+110"

        # Zoom extremamente discreto.
        scale_expr_w = (
            "iw*(1+0.025*sin(2*PI*t/2.0))"
        )

        scale_expr_h = (
            "ih*(1+0.025*sin(2*PI*t/2.0))"
        )

        filters.append(
            f"[{input_index}:v]"
            "format=rgba,"
            f"fade=t=in:st=0:d={fade}:alpha=1,"
            f"fade=t=out:"
            f"st={max(0.01, phrase_duration-fade)}:"
            f"d={fade}:alpha=1,"
            f"scale="
            f"w='{scale_expr_w}':"
            f"h='{scale_expr_h}':"
            "eval=frame,"
            f"setpts=PTS-STARTPTS"
            f"[{text_label}]"
        )

        filters.append(
            f"[{current}]"
            f"[{text_label}]"
            "overlay="
            f"x='{motion_x}':"
            f"y='{base_y}':"
            "eval=frame:"
            f"enable='between(t,"
            f"{phrase.start},"
            f"{phrase.end})'"
            f"[{output_label}]"
        )

        current = output_label

    filter_complex = ";".join(filters)

    command += [
        "-filter_complex",
        filter_complex,
        "-map",
        f"[{current}]",
        "-map",
        f"{audio_index}:a:0",
        "-t",
        str(duration),
        "-r",
        str(FPS),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "22",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        "-shortest",
        str(output_path)
    ]

    # Renderização.
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
            "O FFmpeg falhou durante a renderização.\n\n"
            + error[-9000:]
        )

    if (
        not output_path.exists()
        or output_path.stat().st_size < 10000
    ):
        raise RuntimeError(
            "O vídeo não foi criado corretamente."
        )

    return output_path


# ============================================================
# INTERFACE DE ENTRADA
# ============================================================

st.subheader("1. Música")

audio_upload = st.file_uploader(
    "Envie a música",
    type=[
        "mp3",
        "wav",
        "m4a",
        "aac",
        "flac",
        "ogg"
    ],
    help="A música será usada para gerar a sincronização das letras."
)

st.subheader("2. Fundo visual — opcional")

background_upload = st.file_uploader(
    "Envie um vídeo de show, cantor ou outro fundo",
    type=[
        "mp4",
        "mov",
        "m4v",
        "webm"
    ],
    help=(
        "Opcional. Se você não enviar nada, "
        "a IA cria um fundo abstrato original."
    )
)

rights_confirmed = False

if background_upload:

    rights_confirmed = st.checkbox(
        "Confirmo que tenho autorização/direito de utilizar "
        "o vídeo enviado como fundo."
    )

st.subheader("3. Letras")

lyrics_mode = st.radio(
    "Como deseja obter a letra?",
    [
        "IA transcreve automaticamente",
        "Eu vou fornecer a letra"
    ],
    horizontal=False
)

manual_lyrics = ""

if lyrics_mode == "Eu vou fornecer a letra":

    manual_lyrics = st.text_area(
        "Cole a letra aqui",
        height=220,
        placeholder=(
            "Uma frase por linha.\n\n"
            "Exemplo:\n"
            "Primeira frase da música\n"
            "Segunda frase da música\n"
            "Terceira frase da música"
        )
    )

st.subheader("4. Inteligência da transcrição")

model_choice = st.selectbox(
    "Modelo",
    [
        "tiny",
        "base"
    ],
    index=0,
    help=(
        "tiny é mais rápido e recomendado para o primeiro teste. "
        "base tende a transcrever melhor, mas demora mais."
    )
)

st.subheader("5. Estilo")

st.info(
    "A seleção visual é automática. "
    "O sistema decide fonte, tamanho, posição, "
    "destaques e intensidade de acordo com cada trecho."
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
            "Envie pelo menos a música ou um vídeo contendo a música."
        )

        st.stop()

    if background_upload and not rights_confirmed:

        st.error(
            "Para usar um vídeo enviado como fundo, "
            "confirme que você tem autorização para utilizá-lo."
        )

        st.stop()

    session_dir = (
        TEMP_ROOT
        / next(
            tempfile._get_candidate_names()
        )
    )

    session_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    try:

        # ----------------------------------------------------
        # Salvar uploads
        # ----------------------------------------------------

        with st.status(
            "Preparando os arquivos...",
            expanded=True
        ) as status:

            st.write(
                "📦 Salvando arquivos..."
            )

            audio_path = None
            background_path = None

            if audio_upload:

                audio_original = save_uploaded_file(
                    audio_upload,
                    session_dir,
                    "input_audio"
                    + Path(
                        audio_upload.name
                    ).suffix
                )

                audio_path = (
                    session_dir
                    / "audio.wav"
                )

                normalize_audio(
                    audio_original,
                    audio_path
                )

            if background_upload:

                background_path = save_uploaded_file(
                    background_upload,
                    session_dir,
                    "background"
                    + Path(
                        background_upload.name
                    ).suffix
                )

            # ------------------------------------------------
            # Se só veio vídeo, extrair áudio.
            # ------------------------------------------------

            if (
                audio_path is None
                and background_path is not None
            ):

                audio_path = (
                    session_dir
                    / "audio.wav"
                )

                st.write(
                    "🎧 Extraindo áudio do vídeo..."
                )

                extract_audio_from_video(
                    background_path,
                    audio_path
                )

            duration = ffprobe_duration(
                audio_path
            )

            st.write(
                f"⏱️ Duração detectada: "
                f"{duration:.1f} segundos"
            )

            # ------------------------------------------------
            # Energia
            # ------------------------------------------------

            st.write(
                "🎚️ Analisando intensidade da música..."
            )

            energy = calculate_audio_energy(
                audio_path
            )

            # ------------------------------------------------
            # Letras
            # ------------------------------------------------

            if (
                lyrics_mode
                == "Eu vou fornecer a letra"
            ):

                if not manual_lyrics.strip():

                    raise RuntimeError(
                        "Você escolheu fornecer a letra, "
                        "mas não colocou nenhum texto."
                    )

                st.write(
                    "📝 Organizando a letra fornecida..."
                )

                phrases = (
                    create_phrases_from_manual_lyrics(
                        manual_lyrics,
                        duration,
                        energy
                    )
                )

            else:

                st.write(
                    "🤖 Transcrevendo a música..."
                )

                words, info = transcribe_audio(
                    audio_path,
                    model_choice
                )

                if not words:

                    raise RuntimeError(
                        "A IA não encontrou palavras "
                        "suficientes na música. "
                        "Tente o modelo 'base' ou forneça "
                        "a letra manualmente."
                    )

                phrases = create_phrases(
                    words,
                    energy
                )

            if not phrases:

                raise RuntimeError(
                    "Nenhuma frase foi criada."
                )

            st.write(
                f"📝 {len(phrases)} frases detectadas."
            )

            # ------------------------------------------------
            # Renderização
            # ------------------------------------------------

            st.write(
                "🎨 Construindo estilo visual automaticamente..."
            )

            output_path = (
                session_dir
                / "lyric_video.mp4"
            )

            st.write(
                "🎬 Renderizando vídeo..."
            )

            render_video(
                audio_path,
                background_path,
                phrases,
                session_dir,
                output_path
            )

            status.update(
                label="Vídeo criado com sucesso!",
                state="complete",
                expanded=False
            )

        # ----------------------------------------------------
        # RESULTADO
        # ----------------------------------------------------

        st.success(
            "🎉 Seu lyric video está pronto!"
        )

        st.subheader(
            "▶️ Resultado"
        )

        st.video(
            str(output_path)
        )

        with open(
            output_path,
            "rb"
        ) as video_file:

            video_bytes = video_file.read()

        filename = (
            safe_filename(
                audio_upload.name
                if audio_upload
                else background_upload.name
            )
            + "_LYRIC_AI.mp4"
        )

        st.download_button(
            label="⬇️ BAIXAR VÍDEO MP4",
            data=video_bytes,
            file_name=filename,
            mime="video/mp4",
            type="primary"
        )

        # ----------------------------------------------------
        # RELATÓRIO
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "🧠 Decisões da IA"
        )

        st.write(
            f"• Frases: **{len(phrases)}**"
        )

        st.write(
            "• Formato: **9:16 / 1080×1920**"
        )

        st.write(
            "• Sincronização: **automática**"
        )

        st.write(
            "• Fonte: **selecionada automaticamente**"
        )

        st.write(
            "• Animações: **fade + movimento + escala sutil**"
        )

        st.write(
            "• Fundo: "
            + (
                "**vídeo enviado pelo usuário**"
                if background_path
                else "**gerado automaticamente**"
            )
        )

        st.caption(
            "A versão atual não busca imagens ou vídeos "
            "de terceiros automaticamente. "
            "Quando nenhum fundo é fornecido, "
            "o visual é criado pelo próprio programa."
        )

    except Exception as error:

        st.error(
            "❌ A geração falhou."
        )

        with st.expander(
            "Ver detalhes técnicos"
        ):

            st.code(
                str(error),
                language="text"
            )

        st.warning(
            "Se aparecer um erro aqui, envie-me "
            "uma captura dessa tela. A próxima correção "
            "será feita em cima do erro real."
        )

    finally:

        # Mantém o resultado durante a sessão.
        # Não apagamos imediatamente os arquivos.
        pass