from __future__ import annotations

import re
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

from style_engine import StyleDNA, analyze_references


# ============================================================
# CONFIGURAÇÃO
# ============================================================

st.set_page_config(
    page_title="Lyric AI",
    page_icon="🎵",
    layout="centered",
)

if "result_video" not in st.session_state:
    st.session_state.result_video = None

if "style_dna" not in st.session_state:
    st.session_state.style_dna = StyleDNA()


# ============================================================
# UTILIDADES
# ============================================================

def save_upload(uploaded_file, directory, name=None):
    suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"

    path = directory / (
        name or ("upload" + suffix)
    )

    path.write_bytes(uploaded_file.getbuffer())

    return path


# ============================================================
# TRANSCRIÇÃO
# ============================================================

def transcribe(path, model_size):

    from faster_whisper import WhisperModel

    @st.cache_resource(show_spinner=False)
    def load_model(size):

        return WhisperModel(
            size,
            device="cpu",
            compute_type="int8",
        )

    model = load_model(model_size)

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

        if segment.words:

            for word in segment.words:

                if (
                    word.start is not None
                    and word.end is not None
                    and word.word.strip()
                ):

                    words.append(
                        {
                            "text": word.word.strip(),
                            "start": float(word.start),
                            "end": float(word.end),
                        }
                    )

    if not words:

        raise RuntimeError(
            "Nenhuma palavra foi detectada na música."
        )

    return words, info.language


# ============================================================
# ANÁLISE MUSICAL
# ============================================================

def audio_analysis(path):

    import librosa

    y, sr = librosa.load(
        path,
        sr=22050,
        mono=True,
    )

    if len(y) == 0:

        raise RuntimeError(
            "O áudio está vazio ou ilegível."
        )

    onset = librosa.onset.onset_strength(
        y=y,
        sr=sr,
    )

    tempo, frames = librosa.beat.beat_track(
        onset_envelope=onset,
        sr=sr,
        units="frames",
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
        np.arange(len(rms)),
        sr=sr,
        hop_length=512,
    )

    low, high = np.percentile(
        rms,
        [5, 95],
    )

    energy = np.clip(
        (rms - low) / max(high - low, 1e-8),
        0,
        1,
    )

    return {
        "duration": float(len(y) / sr),
        "tempo": float(
            np.asarray(tempo).reshape(-1)[0]
        ),
        "beats": beats,
        "energy": energy,
        "times": times,
    }


# ============================================================
# ENERGIA
# ============================================================

def energy_at(audio_data, time):

    if len(audio_data["times"]) == 0:
        return 0.5

    index = int(
        np.clip(
            np.searchsorted(
                audio_data["times"],
                time,
            ),
            0,
            len(audio_data["energy"]) - 1,
        )
    )

    return float(
        audio_data["energy"][index]
    )


# ============================================================
# FORÇA DA BATIDA
# ============================================================

def beat_strength(
    audio_data,
    start,
    end,
):

    beats = audio_data["beats"]

    if len(beats) == 0:
        return 0

    amount = np.mean(
        (beats >= start)
        & (beats <= end)
    )

    return float(
        np.clip(amount * 2, 0, 1)
    )


# ============================================================
# DIVISÃO INTELIGENTE DA LETRA
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
            for x in current + [word]
        )

        punctuation = re.search(
            r"[.!?,;:]$",
            previous["text"],
        )

        too_many_words = (
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
            or too_many_words
            or too_long
            or too_slow
        ):

            groups.append(current)

            current = [word]

        else:

            current.append(word)

    if current:
        groups.append(current)

    phrases = []

    for group in groups:

        phrases.append(
            {
                "text": " ".join(
                    x["text"]
                    for x in group
                ),
                "start": group[0]["start"],
                "end": group[-1]["end"],
            }
        )

    return phrases


# ============================================================
# IMPACTO VISUAL
# ============================================================

def score_phrases(
    phrases,
    audio_data,
    dna,
):

    for phrase in phrases:

        middle = (
            phrase["start"]
            + phrase["end"]
        ) / 2

        energy = energy_at(
            audio_data,
            middle,
        )

        beat = beat_strength(
            audio_data,
            phrase["start"],
            phrase["end"],
        )

        duration = max(
            0.1,
            phrase["end"]
            - phrase["start"],
        )

        short_phrase = np.clip(
            1 - duration / 2.5,
            0,
            1,
        )

        phrase["impact"] = float(
            np.clip(
                0.45 * energy
                + 0.20 * beat
                + 0.20 * short_phrase
                + 0.08 * dna.motion
                + 0.07 * dna.cut_rate,
                0,
                1,
            )
        )

    return phrases


# ============================================================
# FONTE AUTOMÁTICA
# ============================================================

def font_path(directory):

    candidates = [

        # Serifada elegante
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",

        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",

        # Alternativas
        "/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf",

        "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",

        # Último fallback
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for candidate in candidates:

        if Path(candidate).exists():
            return candidate

    raise RuntimeError(
        "Nenhuma fonte compatível foi encontrada "
        "no ambiente do Streamlit."
    )


# ============================================================
# TEXTO → IMAGEM
# ============================================================

def text_image(
    text,
    font_path_value,
    size,
    color,
):

    from PIL import (
        Image,
        ImageDraw,
        ImageFont,
    )

    font = ImageFont.truetype(
        font_path_value,
        size,
    )

    width = 930
    height = 540

    image = Image.new(
        "RGBA",
        (width, height),
        (0, 0, 0, 0),
    )

    draw = ImageDraw.Draw(image)

    words = text.split()

    lines = []
    current = ""

    for word in words:

        test = (
            word
            if not current
            else current + " " + word
        )

        box = draw.textbbox(
            (0, 0),
            test,
            font=font,
        )

        if box[2] <= width - 80:

            current = test

        else:

            if current:
                lines.append(current)

            current = word

    if current:
        lines.append(current)

    gap = 14

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

    total_height = (
        sum(heights)
        + gap * max(
            0,
            len(heights) - 1,
        )
    )

    y = max(
        0,
        (height - total_height) / 2,
    )

    for line, box, line_height in zip(
        lines,
        boxes,
        heights,
    ):

        line_width = (
            box[2] - box[0]
        )

        x = (
            width - line_width
        ) / 2

        # Sombra
        draw.text(
            (x + 3, y + 3),
            line,
            font=font,
            fill=(0, 0, 0, 150),
        )

        # Texto
        draw.text(
            (x, y),
            line,
            font=font,
            fill=color,
        )

        y += (
            line_height
            + gap
        )

    return np.asarray(image)


# ============================================================
# RENDERIZAÇÃO
# ============================================================

def render(
    music_path,
    show_path,
    phrases,
    audio_data,
    dna,
    font,
    output,
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

    audio = AudioFileClip(
        music_path
    )

    duration = min(
        float(audio.duration),
        audio_data["duration"],
    )

    show_clip = None

    if show_path:

        show_clip = (
            VideoFileClip(show_path)
            .without_audio()
        )

    background = (
        ColorClip(
            size=(WIDTH, HEIGHT),
            color=(0, 0, 0),
        )
        .with_duration(duration)
    )

    layers = [background]

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

            impact = phrase["impact"]

            # =================================================
            # VÍDEO DO CANTOR
            # =================================================

            use_live = (
                show_clip is not None
                and (
                    impact >= 0.68
                    or (
                        index % 5 == 3
                        and dna.live_probability >= 0.45
                    )
                )
            )

            if use_live:

                show_duration = (
                    show_clip.duration
                )

                if show_duration > 0:

                    show_start = (
                        start
                        % show_duration
                    )

                    show_end = min(
                        show_start + length,
                        show_duration,
                    )

                    if show_end > show_start:

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

                        live = (
                            live
                            .cropped(
                                x_center=live.w / 2,
                                y_center=live.h / 2,
                                width=WIDTH,
                                height=HEIGHT,
                            )
                            .with_start(start)
                        )

                        layers.append(live)

            # =================================================
            # ALTERNÂNCIA VISUAL
            # =================================================

            light_background = (
                index % 4 == 2
                and impact < 0.72
            )

            if light_background:

                layers.append(
                    ColorClip(
                        size=(WIDTH, HEIGHT),
                        color=(245, 245, 245),
                    )
                    .with_start(start)
                    .with_duration(length)
                )

            # =================================================
            # TAMANHO DINÂMICO
            # =================================================

            font_size = max(
                52,
                int(
                    66
                    + 22 * impact
                    - (
                        8
                        if len(
                            phrase["text"]
                        ) > 38
                        else 0
                    )
                ),
            )

            text_color = (
                (15, 15, 15)
                if light_background
                else (255, 255, 255)
            )

            image = text_image(
                phrase["text"],
                font,
                font_size,
                text_color,
            )

            text_clip = (
                ImageClip(
                    image,
                    transparent=True,
                )
                .with_start(start)
                .with_duration(length)
            )

            # =================================================
            # FADE DINÂMICO
            # =================================================

            fade = min(
                0.20 - impact * 0.10,
                length / 3,
            )

            fade = max(
                0.06,
                fade,
            )

            text_clip = text_clip.with_effects(
                [
                    vfx.CrossFadeIn(fade),
                    vfx.CrossFadeOut(fade),
                ]
            )

            layers.append(text_clip)

        # =====================================================
        # COMPOSIÇÃO FINAL
        # =====================================================

        final = (
            CompositeVideoClip(
                layers,
                size=(WIDTH, HEIGHT),
            )
            .with_audio(audio)
        )

        final.write_videofile(
            output,
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

        audio.close()


# ============================================================
# INTERFACE
# ============================================================

st.title(
    "🎵 Lyric AI"
)

st.caption(
    "Gerador automático de lyric videos "
    "9:16 com sincronização, energia, "
    "batidas e análise visual."
)


music = st.file_uploader(
    "🎵 Música / vídeo com áudio",
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
    "🎞️ Vídeos-base — envie 3 a 5",
    type=[
        "mp4",
        "mov",
        "webm",
        "m4v",
    ],
    accept_multiple_files=True,
)


lyrics = st.text_area(
    "📝 Letra — opcional",
    height=140,
    placeholder=(
        "Deixe vazio para a IA transcrever "
        "automaticamente."
    ),
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
# STYLE DNA
# ============================================================

if references:

    with st.spinner(
        "🧬 Analisando seus vídeos-base..."
    ):

        st.session_state.style_dna = (
            analyze_references(
                references
            )
        )


dna = st.session_state.style_dna


with st.expander(
    "🧬 Style DNA",
    expanded=False,
):

    st.json(
        dna.to_dict()
    )


# ============================================================
# GERAR VÍDEO
# ============================================================

if st.button(
    "🚀 CRIAR LYRIC VIDEO",
    type="primary",
    use_container_width=True,
):

    if not music:

        st.error(
            "🎵 Envie uma música ou vídeo primeiro."
        )

        st.stop()

    with st.status(
        "🎬 Construindo seu vídeo...",
        expanded=True,
    ) as status:

        try:

            from moviepy import VideoFileClip

            with tempfile.TemporaryDirectory() as temp:

                directory = Path(temp)

                # =============================================
                # MÚSICA
                # =============================================

                music_path = save_upload(
                    music,
                    directory,
                    "music"
                    + Path(
                        music.name
                    ).suffix,
                )

                # =============================================
                # VÍDEO DO CANTOR
                # =============================================

                show_path = (
                    save_upload(
                        show,
                        directory,
                        "show"
                        + Path(
                            show.name
                        ).suffix,
                    )
                    if show
                    else None
                )

                # =============================================
                # FONTE AUTOMÁTICA
                # =============================================

                selected_font = font_path(
                    directory
                )

                # =============================================
                # EXTRAÇÃO DO ÁUDIO
                # =============================================

                audio_path = music_path

                video_extensions = {
                    ".mp4",
                    ".mov",
                    ".webm",
                    ".m4v",
                }

                if (
                    music_path.suffix.lower()
                    in video_extensions
                ):

                    status.write(
                        "🎧 Extraindo áudio..."
                    )

                    video = VideoFileClip(
                        str(music_path)
                    )

                    if video.audio is None:

                        video.close()

                        raise RuntimeError(
                            "O vídeo enviado não possui áudio."
                        )

                    audio_path = (
                        directory
                        / "audio.wav"
                    )

                    video.audio.write_audiofile(
                        str(audio_path),
                        logger=None,
                    )

                    video.close()

                # =============================================
                # ANÁLISE MUSICAL
                # =============================================

                status.write(
                    "🥁 Analisando batidas e energia..."
                )

                audio_data = audio_analysis(
                    str(audio_path)
                )

                status.write(
                    f"✓ BPM estimado: "
                    f"{audio_data['tempo']:.1f}"
                    f" | duração: "
                    f"{audio_data['duration']:.1f}s"
                )

                # =============================================
                # TRANSCRIÇÃO
                # =============================================

                status.write(
                    "🗣️ Identificando a letra "
                    "com timestamps..."
                )

                words, language = transcribe(
                    str(audio_path),
                    model,
                )

                status.write(
                    f"✓ {len(words)} palavras "
                    f"| idioma: {language}"
                )

                # =============================================
                # FRASES
                # =============================================

                status.write(
                    "✂️ Escolhendo os melhores cortes "
                    "das frases..."
                )

                phrases = group_words(
                    words
                )

                phrases = score_phrases(
                    phrases,
                    audio_data,
                    dna,
                )

                if lyrics.strip():

                    status.write(
                        "✓ Letra fornecida detectada. "
                        "O timing continua baseado na voz."
                    )

                # =============================================
                # RENDER
                # =============================================

                output = (
                    directory
                    / "lyric_ai_definitive.mp4"
                )

                status.write(
                    "🎬 Renderizando vídeo 9:16..."
                )

                render(
                    str(audio_path),
                    (
                        str(show_path)
                        if show_path
                        else None
                    ),
                    phrases,
                    audio_data,
                    dna,
                    selected_font,
                    str(output),
                )

                # =============================================
                # SALVAR RESULTADO
                # =============================================

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
        "🎬 Seu lyric video"
    )

    st.video(
        st.session_state.result_video
    )

    st.download_button(
        "⬇️ SALVAR MP4",
        st.session_state.result_video,
        "lyric_ai_definitive.mp4",
        "video/mp4",
        use_container_width=True,
    )