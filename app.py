from __future__ import annotations

import tempfile
import re
from pathlib import Path

import numpy as np
import streamlit as st


# ============================================================
# CONFIGURAÇÃO
# ============================================================

st.set_page_config(
    page_title="Lyric AI",
    page_icon="🎵",
    layout="centered",
)


# ============================================================
# ESTADO
# ============================================================

if "result_video" not in st.session_state:
    st.session_state.result_video = None

if "font_library" not in st.session_state:
    st.session_state.font_library = None


# ============================================================
# BIBLIOTECA AUTOMÁTICA DE FONTES
# ============================================================

def build_font_library():

    """
    Obtém fontes que vêm junto com o matplotlib.

    Isso evita depender de fontes instaladas no servidor
    do Streamlit.
    """

    from matplotlib import font_manager

    all_fonts = font_manager.findSystemFonts(
        fontext="ttf"
    )

    library = {}

    # --------------------------------------------------------
    # Procurar famílias interessantes
    # --------------------------------------------------------

    for path in all_fonts:

        name = Path(path).stem.lower()

        # DejaVu Serif
        if name == "DejaVuSerif".lower():
            library["serif"] = path

        elif name == "DejaVuSerif-Bold".lower():
            library["serif_bold"] = path

        elif name == "DejaVuSerif-Italic".lower():
            library["serif_italic"] = path

        elif name == "DejaVuSerif-BoldItalic".lower():
            library["serif_bold_italic"] = path

        # DejaVu Sans
        elif name == "DejaVuSans".lower():
            library["sans"] = path

        elif name == "DejaVuSans-Bold".lower():
            library["sans_bold"] = path

        elif name == "DejaVuSans-Oblique".lower():
            library["sans_italic"] = path

        # Condensed
        elif name == "DejaVuSansCondensed".lower():
            library["condensed"] = path

        elif name == "DejaVuSansCondensed-Bold".lower():
            library["condensed_bold"] = path

        # Mono
        elif name == "DejaVuSansMono".lower():
            library["mono"] = path

        # STIX
        elif name == "STIXGeneral".lower():
            library["stix"] = path

        elif name == "STIXGeneral-Bold".lower():
            library["stix_bold"] = path

        elif name == "STIXGeneral-Italic".lower():
            library["stix_italic"] = path

    # --------------------------------------------------------
    # Fallbacks
    # --------------------------------------------------------

    if "serif" not in library:
        library["serif"] = library.get(
            "sans",
            None
        )

    if "serif_bold" not in library:
        library["serif_bold"] = library.get(
            "serif"
        )

    if "serif_italic" not in library:
        library["serif_italic"] = library.get(
            "serif"
        )

    if "sans" not in library:
        library["sans"] = library.get(
            "serif"
        )

    if "sans_bold" not in library:
        library["sans_bold"] = library.get(
            "sans"
        )

    if "condensed" not in library:
        library["condensed"] = library.get(
            "sans"
        )

    if "condensed_bold" not in library:
        library["condensed_bold"] = library.get(
            "sans_bold"
        )

    if "stix" not in library:
        library["stix"] = library.get(
            "serif"
        )

    if "stix_bold" not in library:
        library["stix_bold"] = library.get(
            "serif_bold"
        )

    if "stix_italic" not in library:
        library["stix_italic"] = library.get(
            "serif_italic"
        )

    return library


@st.cache_resource
def get_font_library():

    library = build_font_library()

    if not library:

        raise RuntimeError(
            "Não foi possível carregar nenhuma fonte."
        )

    return library


# ============================================================
# ESCOLHA AUTÔNOMA DA FONTE
# ============================================================

def choose_font(
    phrase,
    energy,
    impact,
    beat,
    index,
    library,
):
    """
    O sistema escolhe a fonte sem intervenção do usuário.

    A escolha depende de:
    - energia musical
    - força da frase
    - presença de batida
    - tamanho da frase
    - pontuação
    - posição dentro da música
    """

    text = phrase["text"]

    length = len(text)

    upper_ratio = (
        sum(
            1 for c in text
            if c.isupper()
        )
        / max(
            1,
            sum(
                1 for c in text
                if c.isalpha()
            ),
        )
    )

    has_exclamation = (
        "!" in text
    )

    has_question = (
        "?" in text
    )

    # --------------------------------------------------------
    # FRASE MUITO FORTE
    # --------------------------------------------------------

    if (
        impact >= 0.85
        or energy >= 0.90
        or has_exclamation
    ):

        if library.get("serif_bold"):
            return (
                library["serif_bold"],
                "serif_bold",
            )

    # --------------------------------------------------------
    # TRECHO EMOCIONAL
    # --------------------------------------------------------

    if (
        impact >= 0.68
        and energy < 0.55
    ):

        if library.get("serif_italic"):
            return (
                library["serif_italic"],
                "serif_italic",
            )

    # --------------------------------------------------------
    # TRECHO MUITO RÁPIDO
    # --------------------------------------------------------

    if length >= 38:

        if library.get("condensed"):
            return (
                library["condensed"],
                "condensed",
            )

    # --------------------------------------------------------
    # BATIDA FORTE
    # --------------------------------------------------------

    if beat >= 0.75:

        if library.get("sans_bold"):
            return (
                library["sans_bold"],
                "sans_bold",
            )

    # --------------------------------------------------------
    # FRASES CURTAS DE IMPACTO
    # --------------------------------------------------------

    if (
        length <= 16
        and impact >= 0.60
    ):

        if library.get("stix_bold"):
            return (
                library["stix_bold"],
                "stix_bold",
            )

    # --------------------------------------------------------
    # PERGUNTAS
    # --------------------------------------------------------

    if has_question:

        if library.get("stix_italic"):
            return (
                library["stix_italic"],
                "stix_italic",
            )

    # --------------------------------------------------------
    # ALTERNÂNCIA CONTROLADA
    # --------------------------------------------------------

    if index % 9 == 6:

        if library.get("serif_italic"):
            return (
                library["serif_italic"],
                "serif_italic",
            )

    if index % 13 == 8:

        if library.get("sans"):
            return (
                library["sans"],
                "sans",
            )

    # --------------------------------------------------------
    # FONTE PRINCIPAL
    # --------------------------------------------------------

    if library.get("serif"):
        return (
            library["serif"],
            "serif",
        )

    return (
        library["sans"],
        "sans",
    )


# ============================================================
# SALVAR UPLOAD
# ============================================================

def save_upload(
    uploaded_file,
    directory,
    filename,
):

    suffix = (
        Path(
            uploaded_file.name
        ).suffix.lower()
    )

    path = (
        directory
        / (
            filename
            + suffix
        )
    )

    path.write_bytes(
        uploaded_file.getbuffer()
    )

    return path


# ============================================================
# TRANSCRIÇÃO
# ============================================================

@st.cache_resource
def load_whisper(model_size):

    from faster_whisper import WhisperModel

    return WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
    )


def transcribe(
    path,
    model_size,
):

    model = load_whisper(
        model_size
    )

    segments, info = model.transcribe(
        path,
        language="pt",
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 450
        },
        condition_on_previous_text=False,
    )

    words = []

    for segment in segments:

        if not segment.words:
            continue

        for word in segment.words:

            if (
                word.start is not None
                and word.end is not None
                and word.word.strip()
            ):

                words.append(
                    {
                        "text": word.word.strip(),
                        "start": float(
                            word.start
                        ),
                        "end": float(
                            word.end
                        ),
                    }
                )

    if not words:

        raise RuntimeError(
            "Não foi possível detectar a letra."
        )

    return (
        words,
        info.language,
    )


# ============================================================
# ANÁLISE DO ÁUDIO
# ============================================================

def analyze_audio(path):

    import librosa

    y, sr = librosa.load(
        path,
        sr=22050,
        mono=True,
    )

    if len(y) == 0:

        raise RuntimeError(
            "O áudio está vazio."
        )

    onset = librosa.onset.onset_strength(
        y=y,
        sr=sr,
    )

    tempo, frames = (
        librosa.beat.beat_track(
            onset_envelope=onset,
            sr=sr,
            units="frames",
        )
    )

    beats = librosa.frames_to_time(
        frames,
        sr=sr,
    )

    rms = librosa.feature.rms(
        y=y,
        frame_length=2048,
        hop_length=512,
    )[0]

    times = librosa.frames_to_time(
        np.arange(
            len(rms)
        ),
        sr=sr,
        hop_length=512,
    )

    low, high = np.percentile(
        rms,
        [5, 95],
    )

    energy = np.clip(
        (
            rms - low
        )
        / max(
            high - low,
            1e-8,
        ),
        0,
        1,
    )

    return {
        "duration": float(
            len(y) / sr
        ),
        "tempo": float(
            np.asarray(
                tempo
            ).reshape(-1)[0]
        ),
        "beats": beats,
        "energy": energy,
        "times": times,
    }


def energy_at(
    audio,
    time,
):

    if len(audio["times"]) == 0:
        return 0.5

    index = np.searchsorted(
        audio["times"],
        time,
    )

    index = int(
        np.clip(
            index,
            0,
            len(
                audio["energy"]
            ) - 1,
        )
    )

    return float(
        audio["energy"][index]
    )


def beat_strength(
    audio,
    start,
    end,
):

    beats = audio["beats"]

    if len(beats) == 0:
        return 0

    amount = np.mean(
        (
            beats >= start
        )
        &
        (
            beats <= end
        )
    )

    return float(
        np.clip(
            amount * 2,
            0,
            1,
        )
    )


# ============================================================
# AGRUPAR PALAVRAS
# ============================================================

def group_words(words):

    groups = []

    current = []

    for word in words:

        if not current:

            current = [word]

            continue

        previous = current[-1]

        gap = (
            word["start"]
            - previous["end"]
        )

        candidate = " ".join(
            x["text"]
            for x in (
                current
                + [word]
            )
        )

        punctuation = bool(
            re.search(
                r"[.!?,;:]$",
                previous["text"],
            )
        )

        too_many = (
            len(current) >= 7
        )

        too_long = (
            len(candidate) > 34
        )

        too_slow = (
            word["end"]
            - current[0]["start"]
            > 2.8
        )

        if (
            gap >= 0.38
            or punctuation
            or too_many
            or too_long
            or too_slow
        ):

            groups.append(
                current
            )

            current = [word]

        else:

            current.append(
                word
            )

    if current:
        groups.append(
            current
        )

    phrases = []

    for group in groups:

        phrases.append(
            {
                "text": " ".join(
                    x["text"]
                    for x in group
                ),
                "start": group[0][
                    "start"
                ],
                "end": group[-1][
                    "end"
                ],
            }
        )

    return phrases


# ============================================================
# ANALISAR IMPACTO
# ============================================================

def score_phrases(
    phrases,
    audio,
):

    for phrase in phrases:

        middle = (
            phrase["start"]
            + phrase["end"]
        ) / 2

        energy = energy_at(
            audio,
            middle,
        )

        beat = beat_strength(
            audio,
            phrase["start"],
            phrase["end"],
        )

        duration = max(
            0.1,
            phrase["end"]
            - phrase["start"],
        )

        short_phrase = np.clip(
            1
            - duration / 2.5,
            0,
            1,
        )

        phrase["energy"] = energy

        phrase["beat"] = beat

        phrase["impact"] = float(
            np.clip(
                0.48 * energy
                + 0.27 * beat
                + 0.25 * short_phrase,
                0,
                1,
            )
        )

    return phrases


# ============================================================
# RENDERIZAR TEXTO
# ============================================================

def make_text_image(
    text,
    font_path,
    font_size,
    color,
):

    from PIL import (
        Image,
        ImageDraw,
        ImageFont,
    )

    WIDTH = 960
    HEIGHT = 600

    image = Image.new(
        "RGBA",
        (
            WIDTH,
            HEIGHT,
        ),
        (
            0,
            0,
            0,
            0,
        ),
    )

    draw = ImageDraw.Draw(
        image
    )

    font = ImageFont.truetype(
        font_path,
        font_size,
    )

    # --------------------------------------------------------
    # QUEBRA DE LINHA
    # --------------------------------------------------------

    words = text.split()

    lines = []

    current = ""

    for word in words:

        test = (
            word
            if not current
            else current
            + " "
            + word
        )

        bbox = draw.textbbox(
            (0, 0),
            test,
            font=font,
        )

        if bbox[2] <= WIDTH - 80:

            current = test

        else:

            if current:
                lines.append(
                    current
                )

            current = word

    if current:
        lines.append(
            current
        )

    # --------------------------------------------------------
    # ALTURA
    # --------------------------------------------------------

    spacing = 16

    boxes = [
        draw.textbbox(
            (0, 0),
            line,
            font=font,
        )
        for line in lines
    ]

    heights = [
        box[3] - box[1]
        for box in boxes
    ]

    total = (
        sum(heights)
        + spacing
        * max(
            0,
            len(heights) - 1,
        )
    )

    y = (
        HEIGHT - total
    ) / 2

    # --------------------------------------------------------
    # TEXTO
    # --------------------------------------------------------

    for line, box, height in zip(
        lines,
        boxes,
        heights,
    ):

        line_width = (
            box[2]
            - box[0]
        )

        x = (
            WIDTH
            - line_width
        ) / 2

        # Sombra
        draw.text(
            (
                x + 4,
                y + 4,
            ),
            line,
            font=font,
            fill=(
                0,
                0,
                0,
                160,
            ),
        )

        # Texto
        draw.text(
            (
                x,
                y,
            ),
            line,
            font=font,
            fill=color,
        )

        y += (
            height
            + spacing
        )

    return np.asarray(
        image
    )


# ============================================================
# RENDER FINAL
# ============================================================

def render_video(
    audio_path,
    show_path,
    phrases,
    audio,
    library,
    output_path,
):

    from moviepy import (
        AudioFileClip,
        ColorClip,
        CompositeVideoClip,
        ImageClip,
        VideoFileClip,
    )

    from moviepy import vfx

    WIDTH = 1080
    HEIGHT = 1920

    audio_clip = AudioFileClip(
        str(audio_path)
    )

    duration = min(
        float(
            audio_clip.duration
        ),
        audio["duration"],
    )

    layers = []

    background = (
        ColorClip(
            size=(
                WIDTH,
                HEIGHT,
            ),
            color=(
                0,
                0,
                0,
            ),
        )
        .with_duration(
            duration
        )
    )

    layers.append(
        background
    )

    show_clip = None

    if show_path:

        show_clip = (
            VideoFileClip(
                str(show_path)
            )
            .without_audio()
        )

    try:

        for index, phrase in enumerate(
            phrases
        ):

            start = max(
                0,
                phrase["start"],
            )

            end = min(
                duration,
                phrase["end"],
            )

            length = end - start

            if length <= 0:
                continue

            impact = phrase[
                "impact"
            ]

            energy = phrase[
                "energy"
            ]

            beat = phrase[
                "beat"
            ]

            # =================================================
            # ESCOLHA AUTÔNOMA DA FONTE
            # =================================================

            selected_font, font_name = (
                choose_font(
                    phrase,
                    energy,
                    impact,
                    beat,
                    index,
                    library,
                )
            )

            # =================================================
            # TAMANHO
            # =================================================

            font_size = int(
                64
                + impact * 28
            )

            if len(
                phrase["text"]
            ) > 36:

                font_size -= 8

            font_size = max(
                48,
                font_size,
            )

            # =================================================
            # COR
            # =================================================

            light = (
                index % 5 == 3
                and impact < 0.60
            )

            if light:

                text_color = (
                    10,
                    10,
                    10,
                )

                bg_color = (
                    245,
                    245,
                    245,
                )

                layers.append(
                    ColorClip(
                        size=(
                            WIDTH,
                            HEIGHT,
                        ),
                        color=bg_color,
                    )
                    .with_start(
                        start
                    )
                    .with_duration(
                        length
                    )
                )

            else:

                text_color = (
                    255,
                    255,
                    255,
                )

            # =================================================
            # VÍDEO DO CANTOR
            # =================================================

            if (
                show_clip is not None
                and (
                    impact >= 0.72
                    or (
                        beat >= 0.70
                        and index % 3 == 0
                    )
                )
            ):

                show_duration = (
                    show_clip.duration
                )

                if show_duration > 0:

                    show_start = (
                        start
                        % show_duration
                    )

                    show_end = min(
                        show_start
                        + length,
                        show_duration,
                    )

                    if (
                        show_end
                        > show_start
                    ):

                        live = (
                            show_clip
                            .subclipped(
                                show_start,
                                show_end,
                            )
                            .resized(
                                height=HEIGHT
                            )
                        )

                        if live.w < WIDTH:

                            live = (
                                live
                                .resized(
                                    width=WIDTH
                                )
                            )

                        if (
                            live.w
                            >= WIDTH
                            and live.h
                            >= HEIGHT
                        ):

                            live = (
                                live
                                .cropped(
                                    x_center=(
                                        live.w / 2
                                    ),
                                    y_center=(
                                        live.h / 2
                                    ),
                                    width=WIDTH,
                                    height=HEIGHT,
                                )
                            )

                            live = (
                                live
                                .with_start(
                                    start
                                )
                            )

                            layers.append(
                                live
                            )

            # =================================================
            # TEXTO
            # =================================================

            image = make_text_image(
                phrase["text"],
                selected_font,
                font_size,
                text_color,
            )

            text_clip = (
                ImageClip(
                    image,
                    transparent=True,
                )
                .with_start(
                    start
                )
                .with_duration(
                    length
                )
            )

            # =================================================
            # FADE AUTOMÁTICO
            # =================================================

            fade = np.clip(
                0.20
                - impact * 0.10,
                0.06,
                0.20,
            )

            fade = min(
                fade,
                length / 3,
            )

            text_clip = (
                text_clip
                .with_effects(
                    [
                        vfx.CrossFadeIn(
                            fade
                        ),
                        vfx.CrossFadeOut(
                            fade
                        ),
                    ]
                )
            )

            layers.append(
                text_clip
            )

        # =====================================================
        # COMPOSIÇÃO
        # =====================================================

        final = CompositeVideoClip(
            layers,
            size=(
                WIDTH,
                HEIGHT,
            ),
        )

        final = final.with_audio(
            audio_clip
        )

        final.write_videofile(
            str(output_path),
            fps=30,
            codec="libx264",
            audio_codec="aac",
            preset="ultrafast",
            threads=2,
            logger=None,
        )

        final.close()

    finally:

        if show_clip:

            show_clip.close()

        audio_clip.close()


# ============================================================
# INTERFACE
# ============================================================

st.title(
    "🎵 Lyric AI"
)

st.caption(
    "Seu gerador automático de lyric videos."
)

music = st.file_uploader(
    "🎵 Música ou vídeo com áudio",
    type=[
        "mp3",
        "wav",
        "m4a",
        "mp4",
        "mov",
        "webm",
    ],
)

show = st.file_uploader(
    "🎤 Vídeo do cantor/show — opcional",
    type=[
        "mp4",
        "mov",
        "webm",
        "m4v",
    ],
)

references = st.file_uploader(
    "🎞️ Vídeos de referência — opcional",
    type=[
        "mp4",
        "mov",
        "webm",
        "m4v",
    ],
    accept_multiple_files=True,
)

model = st.selectbox(
    "🧠 Modelo de transcrição",
    [
        "base",
        "small",
        "tiny",
    ],
    index=0,
)


# ============================================================
# BIBLIOTECA DE FONTES
# ============================================================

try:

    font_library = get_font_library()

except Exception as error:

    st.error(
        "Não consegui carregar a biblioteca "
        "de fontes."
    )

    st.exception(
        error
    )

    st.stop()


# ============================================================
# MOSTRAR FONTES DETECTADAS
# ============================================================

with st.expander(
    "🔤 Biblioteca tipográfica da IA",
    expanded=False,
):

    st.write(
        "A IA escolhe automaticamente entre "
        "as fontes disponíveis."
    )

    st.write(
        "Fontes carregadas:"
    )

    st.write(
        ", ".join(
            sorted(
                font_library.keys()
            )
        )
    )


# ============================================================
# GERAR
# ============================================================

if st.button(
    "🚀 CRIAR LYRIC VIDEO",
    type="primary",
    use_container_width=True,
):

    if not music:

        st.error(
            "🎵 Envie uma música primeiro."
        )

        st.stop()

    with st.status(
        "🎬 Criando seu lyric video...",
        expanded=True,
    ) as status:

        try:

            from moviepy import (
                VideoFileClip
            )

            with tempfile.TemporaryDirectory() as temp:

                directory = Path(
                    temp
                )

                # ---------------------------------------------
                # MÚSICA
                # ---------------------------------------------

                music_path = save_upload(
                    music,
                    directory,
                    "music",
                )

                # ---------------------------------------------
                # SHOW
                # ---------------------------------------------

                show_path = None

                if show:

                    show_path = save_upload(
                        show,
                        directory,
                        "show",
                    )

                # ---------------------------------------------
                # ÁUDIO
                # ---------------------------------------------

                audio_path = (
                    music_path
                )

                if music_path.suffix.lower() in {
                    ".mp4",
                    ".mov",
                    ".webm",
                    ".m4v",
                }:

                    status.write(
                        "🎧 Extraindo áudio..."
                    )

                    video = (
                        VideoFileClip(
                            str(
                                music_path
                            )
                        )
                    )

                    if video.audio is None:

                        video.close()

                        raise RuntimeError(
                            "O vídeo não possui áudio."
                        )

                    audio_path = (
                        directory
                        / "audio.wav"
                    )

                    video.audio.write_audiofile(
                        str(
                            audio_path
                        ),
                        logger=None,
                    )

                    video.close()

                # ---------------------------------------------
                # ÁUDIO
                # ---------------------------------------------

                status.write(
                    "🥁 Analisando música..."
                )

                audio = analyze_audio(
                    str(
                        audio_path
                    )
                )

                status.write(
                    f"✓ BPM estimado: "
                    f"{audio['tempo']:.1f}"
                )

                # ---------------------------------------------
                # TRANSCRIÇÃO
                # ---------------------------------------------

                status.write(
                    "🗣️ Transcrevendo letra..."
                )

                words, language = (
                    transcribe(
                        str(
                            audio_path
                        ),
                        model,
                    )
                )

                status.write(
                    f"✓ {len(words)} palavras "
                    f"identificadas."
                )

                # ---------------------------------------------
                # FRASES
                # ---------------------------------------------

                status.write(
                    "✂️ Criando cortes inteligentes..."
                )

                phrases = group_words(
                    words
                )

                phrases = score_phrases(
                    phrases,
                    audio,
                )

                # ---------------------------------------------
                # FONTES
                # ---------------------------------------------

                status.write(
                    "🔤 A IA está escolhendo "
                    "a tipografia de cada trecho..."
                )

                # ---------------------------------------------
                # RENDER
                # ---------------------------------------------

                output = (
                    directory
                    / "lyric_ai.mp4"
                )

                status.write(
                    "🎬 Renderizando vídeo 9:16..."
                )

                render_video(
                    audio_path,
                    show_path,
                    phrases,
                    audio,
                    font_library,
                    output,
                )

                # ---------------------------------------------
                # GUARDAR
                # ---------------------------------------------

                st.session_state.result_video = (
                    output.read_bytes()
                )

            status.update(
                label="✅ Vídeo pronto!",
                state="complete",
                expanded=False,
            )

        except Exception as error:

            status.update(
                label="❌ A geração falhou",
                state="error",
                expanded=True,
            )

            st.error(
                str(error)
            )

            st.exception(
                error
            )

            st.session_state.result_video = None


# ============================================================
# RESULTADO
# ============================================================

if st.session_state.result_video:

    st.divider()

    st.header(
        "🎬 Resultado"
    )

    st.video(
        st.session_state.result_video
    )

    st.download_button(
        "⬇️ SALVAR VÍDEO",
        st.session_state.result_video,
        "lyric_ai.mp4",
        "video/mp4",
        use_container_width=True,
    )