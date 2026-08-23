import os
import re
import subprocess
import tempfile
import shutil
import unicodedata
import difflib
from pathlib import Path

import streamlit as st


# ============================================================
# CONFIGURAÇÃO
# ============================================================

APP = "16.0-LITE"

# Cores em formato BGR usado pelo ASS
ROYAL = "&H00FF5A2D"
WHITE = "&H00F8F8F8"
BLACK = "&H00050507"

# Fontes que já existem no ambiente Linux/Streamlit
FONT = "DejaVu Sans Condensed"
FONT2 = "Liberation Sans"
SERIF = "DejaVu Serif"


# ============================================================
# FFmpeg
# ============================================================

def ffmpeg():
    try:
        import imageio_ffmpeg

        path = imageio_ffmpeg.get_ffmpeg_exe()

        if path and os.path.exists(path):
            return path

    except Exception:
        pass

    path = shutil.which("ffmpeg")

    if not path:
        raise RuntimeError(
            "FFmpeg não encontrado. "
            "Verifique se imageio-ffmpeg está no requirements.txt."
        )

    return path


def run(cmd, timeout=300):
    process = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )

    if process.returncode != 0:
        raise RuntimeError(process.stderr[-3500:])

    return process.stdout, process.stderr


# ============================================================
# DURAÇÃO
# ============================================================

def duration(path):
    _, error = run(
        [
            ffmpeg(),
            "-hide_banner",
            "-i",
            str(path),
            "-f",
            "null",
            "-",
        ],
        90,
    )

    match = re.search(
        r"Duration:\s*(\d+):(\d+):([\d.]+)",
        error,
    )

    if not match:
        return 0

    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))

    return hours * 3600 + minutes * 60 + seconds


# ============================================================
# TEXTO
# ============================================================

def norm(text):
    text = unicodedata.normalize("NFC", text or "")

    text = (
        text
        .replace("\ufeff", "")
        .replace("\u200b", "")
    )

    text = "".join(
        char
        for char in text
        if unicodedata.category(char)[0] != "C"
    )

    return re.sub(r"\s+", " ", text).strip()


def key(text):
    text = unicodedata.normalize("NFKD", text or "")

    text = "".join(
        char
        for char in text.lower()
        if not unicodedata.combining(char)
    )

    return re.sub(r"[^a-z0-9]", "", text)


def similarity(a, b):
    a = key(a)
    b = key(b)

    if not a or not b:
        return 0

    if a == b:
        return 1

    if a in b or b in a:
        return 0.9

    return difflib.SequenceMatcher(
        None,
        a,
        b,
        autojunk=False,
    ).ratio()


# ============================================================
# TEMPOS DA LETRA
# ============================================================

def parse_time(value):
    value = value.replace(",", ".").strip()

    if ":" in value:
        minutes, seconds = value.split(":", 1)
        return int(minutes) * 60 + float(seconds)

    return float(value)


def parse_lyrics(text):
    """
    Aceita:

    00:02.3 - 00:06.8
    Minha frase

    ou:

    00:02.3 - 00:06.8 | Minha frase
    """

    lines = [
        line.strip()
        for line in (text or "")
        .replace("\r", "")
        .split("\n")
    ]

    pattern = re.compile(
        r"^"
        r"(\d{1,2}:\d{2}(?:[.,]\d{1,3})?|\d+(?:[.,]\d+)?)"
        r"\s*[-–—]\s*"
        r"(\d{1,2}:\d{2}(?:[.,]\d{1,3})?|\d+(?:[.,]\d+)?)"
        r"(?:\s*\|\s*(.*))?"
        r"$"
    )

    result = []

    index = 0

    while index < len(lines):

        if not lines[index]:
            index += 1
            continue

        match = pattern.match(lines[index])

        if match:

            start = parse_time(match.group(1))
            end = parse_time(match.group(2))

            text_line = norm(match.group(3) or "")

            if (
                not text_line
                and index + 1 < len(lines)
                and not pattern.match(lines[index + 1])
            ):
                text_line = norm(lines[index + 1])
                index += 1

            if text_line and end > start:
                result.append(
                    {
                        "start": start,
                        "end": end,
                        "text": text_line,
                    }
                )

        index += 1

    return sorted(
        result,
        key=lambda item: item["start"],
    )


# ============================================================
# WHISPER
# ============================================================

@st.cache_resource(show_spinner=False)
def get_model():

    from faster_whisper import WhisperModel

    return WhisperModel(
        "small",
        device="cpu",
        compute_type="int8",
        cpu_threads=2,
        num_workers=1,
    )


def transcribe(path):

    model = get_model()

    segments, info = model.transcribe(
        path,
        language="pt",
        word_timestamps=True,
        beam_size=1,
        best_of=1,
        temperature=0,
        condition_on_previous_text=False,
        vad_filter=True,
    )

    words = []

    for segment in segments:

        for word in segment.words or []:

            text = norm(word.word)

            if not text:
                continue

            if len(text) > 40:
                continue

            letters = sum(
                char.isalpha()
                for char in text
            )

            digits = sum(
                char.isdigit()
                for char in text
            )

            # Evita artefatos como A1A
            if letters and digits:
                continue

            words.append(
                {
                    "word": text,
                    "start": float(word.start),
                    "end": float(word.end),
                }
            )

    words.sort(
        key=lambda item: item["start"]
    )

    return words, getattr(
        info,
        "language",
        "pt",
    )


# ============================================================
# ALINHAMENTO PALAVRA POR PALAVRA
# ============================================================

def align_phrase(text, start, end, words):

    tokens = re.findall(
        r"\S+",
        text,
    )

    candidates = [
        word
        for word in words
        if word["end"] >= start - 0.5
        and word["start"] <= end + 0.5
    ]

    used = -1

    mapped = []

    known = []

    for index, token in enumerate(tokens):

        best_index = None
        best_score = 0

        limit = min(
            len(candidates),
            used + 31,
        )

        for candidate_index in range(
            used + 1,
            limit,
        ):

            score = similarity(
                token,
                candidates[candidate_index]["word"],
            )

            if (
                start
                <= candidates[candidate_index]["start"]
                <= end
            ):
                score += 0.05

            if score > best_score:
                best_score = score
                best_index = candidate_index

        if (
            best_index is not None
            and best_score >= 0.38
        ):

            used = best_index

            mapped.append(
                [
                    token,
                    candidates[best_index]["start"],
                    candidates[best_index]["end"],
                ]
            )

            known.append(index)

        else:

            mapped.append(
                [
                    token,
                    None,
                    None,
                ]
            )

    # Preenche palavras que o Whisper não encontrou
    for index, item in enumerate(mapped):

        if item[1] is not None:
            continue

        previous = max(
            [
                value
                for value in known
                if value < index
            ],
            default=-1,
        )

        next_index = min(
            [
                value
                for value in known
                if value > index
            ],
            default=len(mapped),
        )

        left = (
            mapped[previous][2]
            if previous >= 0
            else start
        )

        right = (
            mapped[next_index][1]
            if next_index < len(mapped)
            else end
        )

        if right <= left:
            right = left + 0.08

        step = (
            right - left
        ) / max(
            1,
            next_index - previous,
        )

        item[1] = max(
            start,
            min(
                end - 0.06,
                left + step * (index - previous),
            ),
        )

        item[2] = min(
            end,
            max(
                item[1] + 0.06,
                left + step * (index - previous + 1),
            ),
        )

    return [
        {
            "word": item[0],
            "start": max(
                start,
                item[1],
            ),
            "end": min(
                end,
                item[2],
            ),
        }
        for item in mapped
    ]


# ============================================================
# CRIAÇÃO DAS FRASES
# ============================================================

def build_scenes(lyrics, words, audio_end):

    timed = parse_lyrics(lyrics)

    scenes = []

    # -----------------------------------------
    # LETRA COM TEMPOS
    # -----------------------------------------

    if timed:

        for index, line in enumerate(timed):

            start = max(
                0,
                min(
                    audio_end,
                    line["start"],
                ),
            )

            end = min(
                audio_end,
                line["end"],
            )

            if index + 1 < len(timed):

                end = min(
                    end,
                    timed[index + 1]["start"],
                )

            if end <= start:
                continue

            words_aligned = align_phrase(
                line["text"],
                start,
                end,
                words,
            )

            if words_aligned:

                scenes.append(
                    {
                        "start": start,
                        "end": end,
                        "words": words_aligned,
                        "text": line["text"],
                    }
                )

        return scenes

    # -----------------------------------------
    # LETRA SEM TEMPOS
    # -----------------------------------------

    lines = [
        norm(line)
        for line in (lyrics or "").splitlines()
        if norm(line)
    ]

    cursor = 0

    for line in lines:

        tokens = re.findall(
            r"\S+",
            line,
        )

        matches = []

        for token in tokens:

            best_index = None
            best_score = 0

            for index in range(
                cursor,
                min(
                    len(words),
                    cursor + 80,
                ),
            ):

                score = similarity(
                    token,
                    words[index]["word"],
                )

                if score > best_score:
                    best_score = score
                    best_index = index

            if (
                best_index is not None
                and best_score >= 0.38
            ):

                matches.append(best_index)
                cursor = best_index + 1

        if matches:

            start = words[
                matches[0]
            ]["start"]

            end = words[
                matches[-1]
            ]["end"]

        elif cursor < len(words):

            start = words[
                cursor
            ]["start"]

            end = min(
                audio_end,
                start + max(
                    0.8,
                    0.3 * len(tokens),
                ),
            )

            cursor += 1

        else:

            break

        end = min(
            audio_end,
            max(
                end,
                start + 0.4,
            ),
        )

        words_aligned = align_phrase(
            line,
            start,
            end,
            words,
        )

        if words_aligned:

            scenes.append(
                {
                    "start": start,
                    "end": end,
                    "words": words_aligned,
                    "text": line,
                }
            )

    return scenes


# ============================================================
# FALLBACK: APENAS WHISPER
# ============================================================

def auto_scenes(words, audio_end):

    groups = []

    current = []

    for word in words:

        if (
            current
            and word["start"]
            - current[-1]["end"]
            > 0.65
        ):

            groups.append(current)
            current = []

        current.append(word)

    if current:
        groups.append(current)

    scenes = []

    for group in groups:

        start = max(
            0,
            group[0]["start"],
        )

        end = min(
            audio_end,
            group[-1]["end"] + 0.20,
        )

        if end > start:

            scenes.append(
                {
                    "start": start,
                    "end": end,
                    "words": group,
                    "text": " ".join(
                        word["word"]
                        for word in group
                    ),
                }
            )

    return scenes


# ============================================================
# COMPLETA O FINAL DA MÚSICA
# ============================================================

def add_missing_tail(
    scenes,
    words,
    audio_end,
):

    if not words:
        return scenes

    covered_until = max(
        (
            scene["end"]
            for scene in scenes
        ),
        default=0,
    )

    tail = [
        word
        for word in words
        if word["start"]
        >= covered_until - 0.15
        and word["start"]
        < audio_end
    ]

    if not tail:
        return scenes

    groups = []

    current = []

    for word in tail:

        if (
            current
            and word["start"]
            - current[-1]["end"]
            > 0.65
        ):

            groups.append(current)
            current = []

        current.append(word)

    if current:
        groups.append(current)

    for group in groups:

        start = max(
            covered_until,
            group[0]["start"],
        )

        end = min(
            audio_end,
            group[-1]["end"] + 0.20,
        )

        if end > start:

            scenes.append(
                {
                    "start": start,
                    "end": end,
                    "words": group,
                    "text": " ".join(
                        word["word"]
                        for word in group
                    ),
                }
            )

    return sorted(
        scenes,
        key=lambda scene: scene["start"],
    )


# ============================================================
# ASS
# ============================================================

def ass_time(value):

    value = max(
        0,
        float(value),
    )

    hours = int(
        value // 3600
    )

    value -= hours * 3600

    minutes = int(
        value // 60
    )

    seconds = value - minutes * 60

    return (
        f"{hours}:"
        f"{minutes:02d}:"
        f"{seconds:05.2f}"
    )


def ass_text(text):

    return (
        norm(text)
        .replace("\\", "/")
        .replace("{", "(")
        .replace("}", ")")
        .replace("\n", " ")
    )


def make_ass(scenes):

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 720
PlayResY: 1280
ScaledBorderAndShadow: yes
WrapStyle: 2

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding

Style: Main,{FONT},72,{WHITE},&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,5,25,25,0,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""

    lines = [header]

    for scene_index, scene in enumerate(scenes):

        word_count = len(
            scene["words"]
        )

        if word_count <= 4:
            size = 82
        elif word_count <= 7:
            size = 72
        elif word_count <= 10:
            size = 62
        else:
            size = 54

        for current_index, word in enumerate(
            scene["words"]
        ):

            start = word["start"]

            if (
                current_index + 1
                < len(scene["words"])
            ):

                end = scene[
                    "words"
                ][current_index + 1]["start"]

            else:

                end = scene["end"]

            end = max(
                start + 0.08,
                min(
                    scene["end"],
                    end,
                ),
            )

            pieces = []

            for word_index in range(
                current_index + 1
            ):

                word_text = ass_text(
                    scene["words"][
                        word_index
                    ]["word"]
                ).upper()

                # Fonte grossa é majoritária.
                # A cada frase algumas palavras mudam.
                cycle = word_index % 6

                if cycle in (0, 1, 3, 5):
                    font_name = FONT
                elif cycle == 2:
                    font_name = FONT2
                else:
                    font_name = SERIF

                # Azul aparece ocasionalmente.
                blue = (
                    scene_index % 2 == 0
                    and current_index % 2 == 0
                    and (
                        word_index == 1
                        or (
                            word_count >= 5
                            and word_index
                            == word_count // 2
                        )
                    )
                )

                if blue:
                    color = ROYAL
                else:
                    color = (
                        BLACK
                        if scene_index % 2
                        else WHITE
                    )

                pieces.append(
                    "{\\fn"
                    + font_name
                    + "\\fs"
                    + str(size)
                    + "\\c"
                    + color
                    + "}"
                    + word_text
                )

            text = " ".join(pieces)

            # Frases grandes recebem uma quebra.
            if (
                len(text) > 48
                and word_count > 7
            ):

                visible_words = [
                    ass_text(
                        scene["words"][i]["word"]
                    ).upper()
                    for i in range(
                        current_index + 1
                    )
                ]

                half = (
                    len(visible_words)
                    + 1
                ) // 2

                first_line = " ".join(
                    visible_words[:half]
                )

                second_line = " ".join(
                    visible_words[half:]
                )

                text = (
                    "{\\fs"
                    + str(size)
                    + "\\c"
                    + (
                        BLACK
                        if scene_index % 2
                        else WHITE
                    )
                    + "}"
                    + first_line
                    + "\\N"
                    + second_line
                )

            dialogue = (
                "Dialogue: 0,"
                + ass_time(start)
                + ","
                + ass_time(end)
                + ",Main,,0,0,0,,"
                + "{\\an5\\pos(360,640)}"
                + text
            )

            lines.append(dialogue)

    return "\n".join(lines)


# ============================================================
# RENDERIZAÇÃO
# ============================================================

def make_video(
    audio,
    scenes,
    output,
    audio_end,
    progress,
):

    temporary = Path(
        tempfile.mkdtemp(
            prefix="lyric_lite_"
        )
    )

    try:

        ass_file = (
            temporary
            / "lyrics.ass"
        )

        ass_file.write_text(
            make_ass(scenes),
            encoding="utf-8",
        )

        # ----------------------------------------------------
        # Fundo inteiro do vídeo.
        # Não usamos PIL.
        # Não criamos milhares de PNGs.
        # ----------------------------------------------------

        segments = []

        cursor = 0

        for index, scene in enumerate(
            scenes
        ):

            # Espaço antes da frase
            if scene["start"] > cursor + 0.02:

                segments.append(
                    (
                        "black",
                        scene["start"]
                        - cursor,
                    )
                )

            background = (
                "white"
                if index % 2
                else "black"
            )

            segments.append(
                (
                    background,
                    max(
                        0.08,
                        scene["end"]
                        - scene["start"],
                    ),
                )
            )

            cursor = scene["end"]

        # Garante vídeo até o fim do áudio
        if cursor < audio_end:

            segments.append(
                (
                    "black",
                    audio_end - cursor,
                )
            )

        graph = []
        names = []

        for index, (
            background,
            segment_duration,
        ) in enumerate(segments):

            graph.append(
                f"color=c={background}:"
                f"s=720x1280:"
                f"r=24:"
                f"d={segment_duration:.3f}"
                f"[bg{index}]"
            )

            names.append(
                f"[bg{index}]"
            )

        if not names:
            raise RuntimeError(
                "Não foi possível criar o fundo do vídeo."
            )

        graph.append(
            "".join(names)
            + f"concat=n={len(names)}:v=1:a=0,"
              "format=yuv420p[background]"
        )

        ass_path = ass_file.as_posix()

        filter_graph = (
            ";".join(graph)
            + ";"
            + "[background]"
            + "subtitles="
            + "filename='"
            + ass_path
            + "'"
            + ":fontsdir=/usr/share/fonts/truetype/dejavu"
            + "[video]"
        )

        progress.progress(
            0.65,
            text="Renderizando legendas…",
        )

        run(
            [
                ffmpeg(),
                "-y",
                "-i",
                str(audio),
                "-filter_complex",
                filter_graph,
                "-map",
                "[video]",
                "-map",
                "0:a:0",
                "-t",
                f"{audio_end:.3f}",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "25",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-movflags",
                "+faststart",
                str(output),
            ],
            timeout=max(
                240,
                int(audio_end * 12),
            ),
        )

        progress.progress(
            1,
            text="Vídeo concluído.",
        )

    finally:

        shutil.rmtree(
            temporary,
            ignore_errors=True,
        )


# ============================================================
# INTERFACE
# ============================================================

st.set_page_config(
    page_title="Lyric AI Studio Lite",
    page_icon="🎵",
)

st.title(
    "🎵 Lyric AI Studio"
)

st.caption(
    f"{APP} · "
    "versão otimizada para Streamlit Community Cloud"
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
    "2. Letra oficial (recomendado)",
    height=220,
    placeholder=(
        "Uma frase por linha.\n\n"
        "Ou:\n"
        "00:02.3 - 00:06.8\n"
        "Minha frase"
    ),
)


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

    temporary = Path(
        tempfile.mkdtemp(
            prefix="lyric_app_"
        )
    )

    try:

        source = (
            temporary
            / (
                Path(audio.name).stem
                + ".mp4"
            )
        )

        source.write_bytes(
            audio.getbuffer()
        )

        audio_duration = duration(
            source
        )

        if audio_duration <= 0:

            raise RuntimeError(
                "Não foi possível identificar "
                "a duração do arquivo."
            )

        st.write(
            f"⏱️ Duração: "
            f"{audio_duration:.2f}s"
        )

        progress = st.progress(
            0.05,
            text="Reconhecendo palavras…",
        )

        words, language = transcribe(
            source
        )

        progress.progress(
            0.35,
            text="Sincronizando letra…",
        )

        if lyrics.strip():

            scenes = build_scenes(
                lyrics,
                words,
                audio_duration,
            )

        else:

            scenes = auto_scenes(
                words,
                audio_duration,
            )

        if not scenes:

            raise RuntimeError(
                "Nenhuma legenda foi criada. "
                "Cole a letra oficial ou use "
                "uma frase por linha."
            )

        # Se a letra oficial tiver acabado
        # antes do áudio, tenta continuar
        # com o reconhecimento real.
        scenes = add_missing_tail(
            scenes,
            words,
            audio_duration,
        )

        total_words = sum(
            len(scene["words"])
            for scene in scenes
        )

        progress.progress(
            0.45,
            text=(
                f"{len(scenes)} frases · "
                f"{total_words} palavras"
            ),
        )

        output = (
            temporary
            / "lyric_ai_final.mp4"
        )

        make_video(
            source,
            scenes,
            output,
            audio_duration,
            progress,
        )

        st.success(
            "✅ Vídeo criado."
        )

        st.video(
            str(output)
        )

        st.download_button(
            "⬇️ BAIXAR MP4",
            output.read_bytes(),
            "lyric_ai_final.mp4",
            "video/mp4",
            use_container_width=True,
        )

        with st.expander(
            "Diagnóstico"
        ):

            st.write(
                f"Idioma: {language}"
            )

            st.write(
                f"Frases: {len(scenes)}"
            )

            st.write(
                f"Palavras: {total_words}"
            )

            st.code(
                "\n".join(
                    f"{scene['start']:.2f}"
                    f"-"
                    f"{scene['end']:.2f}"
                    f" | "
                    f"{scene['text']}"
                    for scene in scenes
                )
            )

    except Exception as error:

        st.error(
            "❌ A geração falhou."
        )

        st.code(
            str(error)
        )

    finally:

        shutil.rmtree(
            temporary,
            ignore_errors=True,
        )