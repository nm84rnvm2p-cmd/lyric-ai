import os
import re
import math
import shutil
import subprocess
import tempfile
import urllib.request
import difflib
import unicodedata
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# LYRIC AI STUDIO — STREAMLIT SAFE
# ============================================================

APP_VERSION = "9.0-NO-CROP"
FPS = 30

BLACK = (5, 5, 7)
WHITE = (248, 248, 246)
ROYAL = (45, 92, 255)

CACHE = Path(".lyric_cache")
FONT_DIR = CACHE / "fonts"
FONT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FONTES
# ============================================================

FONTS = {
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

    "Libre Baskerville":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/librebaskerville/LibreBaskerville-Regular.ttf",

    "Space Mono":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/spacemono/SpaceMono-Regular.ttf",
}


MAIN_FONTS = [
    "Anton",
    "Bebas Neue",
    "Archivo Black",
    "Montserrat",
    "Oswald",
]


ALT_FONTS = [
    "Montserrat",
    "Oswald",
    "DM Serif Display",
    "Playfair Display",
    "Libre Baskerville",
    "Bebas Neue",
]


SYSTEM_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
]


# ============================================================
# UTILIDADES
# ============================================================

def clamp(x, a=0, b=1):
    return max(a, min(b, x))


def ease(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def safe(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:100]


def norm(s):
    return re.sub(r"\s+", " ", s or "").strip()


def normalize_token(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(
        c for c in s
        if not unicodedata.combining(c)
    )
    return re.sub(
        r"[^a-zA-Z0-9]",
        "",
        s
    ).lower()


def similarity(a, b):
    a = normalize_token(a)
    b = normalize_token(b)

    if not a or not b:
        return 0

    if a == b:
        return 1

    if a in b or b in a:
        return 0.90

    return difflib.SequenceMatcher(
        None,
        a,
        b,
        autojunk=False
    ).ratio()


# ============================================================
# FFMPEG
# ============================================================

def ffmpeg():
    try:
        import imageio_ffmpeg

        p = imageio_ffmpeg.get_ffmpeg_exe()

        if p and os.path.exists(p):
            return p

    except Exception:
        pass

    p = shutil.which("ffmpeg")

    if p:
        return p

    raise RuntimeError(
        "FFmpeg não encontrado. "
        "Verifique o requirements.txt."
    )


def run(cmd, timeout=None):
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout
    )

    if p.returncode:
        raise RuntimeError(
            p.stderr[-5000:]
        )

    return p.stdout


def duration(path):
    p = subprocess.run(
        [
            ffmpeg(),
            "-hide_banner",
            "-i",
            path,
            "-f",
            "null",
            "-"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60
    )

    m = re.search(
        r"Duration:\s*(\d+):(\d+):([\d.]+)",
        p.stderr
    )

    if not m:
        return 0

    return (
        int(m.group(1)) * 3600
        + int(m.group(2)) * 60
        + float(m.group(3))
    )


# ============================================================
# DETECÇÃO DO FIM REAL DA MÚSICA
# ============================================================

def real_end(path, dur):
    """
    Detecta silêncio no final.
    Evita continuar o vídeo quando a letra acabou
    antes da duração total do arquivo.
    """

    try:
        p = subprocess.run(
            [
                ffmpeg(),
                "-hide_banner",
                "-i",
                path,
                "-af",
                "silencedetect=noise=-38dB:d=0.55",
                "-f",
                "null",
                "-"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=max(
                60,
                int(dur * 2)
            )
        )

        starts = [
            float(x)
            for x in re.findall(
                r"silence_start:\s*([\d.]+)",
                p.stderr
            )
        ]

        ends = [
            float(x)
            for x in re.findall(
                r"silence_end:\s*([\d.]+)",
                p.stderr
            )
        ]

        if starts and starts[-1] > dur * 0.78:

            if ends and ends[-1] >= starts[-1]:
                return min(
                    dur,
                    ends[-1] + 0.12
                )

            return min(
                dur,
                starts[-1] + 0.18
            )

    except Exception:
        pass

    return dur


# ============================================================
# FONTES
# ============================================================

@st.cache_resource(show_spinner=False)
def registry():

    result = {}

    for name, url in FONTS.items():

        target = FONT_DIR / (
            safe(name) + ".ttf"
        )

        if (
            not target.exists()
            or target.stat().st_size < 10000
        ):

            try:

                request = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent":
                        "LyricAIStudio"
                    }
                )

                with urllib.request.urlopen(
                    request,
                    timeout=25
                ) as response:

                    target.write_bytes(
                        response.read()
                    )

            except Exception:

                target.unlink(
                    missing_ok=True
                )

        if (
            target.exists()
            and target.stat().st_size >= 10000
        ):

            result[name] = str(target)

    if not result:

        for path in SYSTEM_FONTS:

            if os.path.exists(path):

                result["System"] = path
                break

    return result


def font(name, r, size):

    path = r.get(
        name,
        next(iter(r.values()))
    )

    return ImageFont.truetype(
        path,
        max(12, int(size))
    )


def box(text, f):

    b = f.getbbox(text)

    return (
        b[0],
        b[1],
        b[2] - b[0],
        b[3] - b[1]
    )


def fit(
    text,
    name,
    r,
    size,
    max_width
):

    current = int(size)

    while current >= 24:

        f = font(
            name,
            r,
            current
        )

        if box(text, f)[2] <= max_width:
            return f

        current -= 2

    return font(
        name,
        r,
        24
    )


# ============================================================
# WHISPER
# ============================================================

def transcribe(
    path,
    model,
    status
):

    from faster_whisper import WhisperModel

    whisper = WhisperModel(
        model,
        device="cpu",
        compute_type="int8",
        cpu_threads=max(
            2,
            min(
                8,
                os.cpu_count() or 4
            )
        ),
        num_workers=1
    )

    if status:
        status.write(
            f"🎙️ Reconhecendo palavra por palavra "
            f"com **{model}**…"
        )

    segments, info = whisper.transcribe(
        path,
        language="pt",
        task="transcribe",
        beam_size=5,
        best_of=5,
        patience=1.0,
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=False,
        word_timestamps=True,
        initial_prompt=(
            "Letra de música brasileira "
            "em português. "
            "Reconheça todas as palavras, "
            "repetições, gírias e contrações. "
            "Não resuma nem traduza."
        )
    )

    result = []

    for segment in segments:

        if not segment.words:
            continue

        for word in segment.words:

            text = norm(
                word.word
            )

            if (
                text
                and len(text) <= 40
            ):

                result.append({
                    "word": text,
                    "start": float(
                        word.start
                    ),
                    "end": float(
                        word.end
                    ),
                    "prob": float(
                        getattr(
                            word,
                            "probability",
                            0
                        ) or 0
                    )
                })

    return (
        result,
        getattr(
            info,
            "language",
            "pt"
        )
    )


# ============================================================
# LETRA COM TEMPOS
# ============================================================

def parse_timed(text):

    result = []

    for raw in text.splitlines():

        raw = raw.strip()

        if not raw:
            continue

        # Exemplo:
        # 00:02.3 - 00:06.8 | Minha frase
        match = re.match(
            r"^\s*"
            r"(\d{1,2}):([0-5]\d)"
            r"(?:[.,](\d{1,3}))?"
            r"\s*(?:-|–|—)\s*"
            r"(\d{1,2}):([0-5]\d)"
            r"(?:[.,](\d{1,3}))?"
            r"\s*\|\s*(.+)$",
            raw
        )

        if match:

            start = (
                int(match.group(1)) * 60
                + int(match.group(2))
                + float(
                    "0." +
                    (
                        match.group(3)
                        or "0"
                    )
                )
            )

            end = (
                int(match.group(4)) * 60
                + int(match.group(5))
                + float(
                    "0." +
                    (
                        match.group(6)
                        or "0"
                    )
                )
            )

            result.append({
                "start": start,
                "end": end,
                "text": norm(
                    match.group(7)
                )
            })

            continue

        # Exemplo:
        # 00:02.3 | Minha frase
        match = re.match(
            r"^\s*"
            r"(\d{1,2}):([0-5]\d)"
            r"(?:[.,](\d{1,3}))?"
            r"\s*\|\s*(.+)$",
            raw
        )

        if match:

            start = (
                int(match.group(1)) * 60
                + int(match.group(2))
                + float(
                    "0." +
                    (
                        match.group(3)
                        or "0"
                    )
                )
            )

            result.append({
                "start": start,
                "text": norm(
                    match.group(4)
                )
            })

            continue

        # Exemplo:
        # 2.3 | Minha frase
        match = re.match(
            r"^\s*"
            r"(\d+(?:[.,]\d+)?)"
            r"\s*\|\s*(.+)$",
            raw
        )

        if match:

            result.append({
                "start": float(
                    match.group(1).replace(
                        ",",
                        "."
                    )
                ),
                "text": norm(
                    match.group(2)
                )
            })

    return sorted(
        result,
        key=lambda x: x["start"]
    )


def plain(text):

    return [
        norm(line)
        for line in text.splitlines()
        if norm(line)
    ]


# ============================================================
# ALINHAMENTO
# ============================================================

def align_phrase(
    text,
    start,
    end,
    asr
):

    tokens = re.findall(
        r"\S+",
        text
    )

    candidates = [
        (i, word)
        for i, word in enumerate(asr)
        if (
            word["end"] >= start - 0.7
            and
            word["start"] <= end + 0.7
        )
    ]

    mapped = []
    cursor = 0

    for token in tokens:

        best = None
        best_score = 0

        for j in range(
            cursor,
            min(
                len(candidates),
                cursor + 24
            )
        ):

            index, word = candidates[j]

            score = similarity(
                token,
                word["word"]
            )

            if (
                start
                <= word["start"]
                <= end
            ):
                score += 0.05

            if score > best_score:
                best_score = score
                best = (
                    j,
                    word
                )

            if score >= 1:
                break

        if (
            best
            and best_score >= 0.43
        ):

            j, word = best

            mapped.append(
                (
                    token,
                    word,
                    best_score
                )
            )

            cursor = j + 1

        else:

            mapped.append(
                (
                    token,
                    None,
                    0
                )
            )

    known = [
        i
        for i, item in enumerate(mapped)
        if item[1] is not None
    ]

    result = []

    for i, (
        token,
        word,
        score
    ) in enumerate(mapped):

        if word is not None:

            result.append({
                "word": token,
                "start": max(
                    start,
                    word["start"]
                ),
                "end": min(
                    end,
                    max(
                        word["start"] + 0.055,
                        word["end"]
                    )
                ),
                "prob": max(
                    word.get("prob", 0),
                    score * 0.75
                )
            })

            continue

        previous = max(
            [
                k
                for k in known
                if k < i
            ],
            default=-1
        )

        following = min(
            [
                k
                for k in known
                if k > i
            ],
            default=len(tokens)
        )

        a = (
            result[previous]["end"]
            if previous >= 0
            else start
        )

        b = (
            mapped[following][1]["start"]
            if following < len(tokens)
            else end
        )

        gap = max(
            0.08,
            b - a
        )

        total = max(
            1,
            following - previous
        )

        word_start = (
            a +
            gap *
            (
                (i - previous)
                / total
            )
        )

        word_end = (
            a +
            gap *
            (
                (i - previous + 1)
                / total
            )
        )

        result.append({
            "word": token,
            "start": clamp(
                word_start,
                start,
                end
            ),
            "end": clamp(
                max(
                    word_start + 0.055,
                    word_end
                ),
                start + 0.055,
                end
            ),
            "prob": 0.45
        })

    return result


def build_timed(
    lines,
    asr,
    audio_end
):

    scenes = []

    for i, line in enumerate(lines):

        start = line["start"]

        if start >= audio_end:
            continue

        if "end" in line:

            end = line["end"]

        elif i + 1 < len(lines):

            end = lines[
                i + 1
            ]["start"]

        else:

            end = audio_end

        if i + 1 < len(lines):

            end = min(
                end,
                lines[
                    i + 1
                ]["start"]
            )

        end = min(
            end,
            audio_end
        )

        if end <= start:
            continue

        words = align_phrase(
            line["text"],
            start,
            end,
            asr
        )

        for word in words:

            word["phrase_id"] = i
            word["phrase_text"] = line["text"]

        if words:

            scenes.append({
                "start": start,
                "end": end,
                "words": words,
                "phrase_text": line["text"]
            })

    return scenes


def align_plain(
    text,
    asr,
    audio_end
):

    lines = plain(text)

    scenes = []
    cursor = 0

    for phrase_id, line in enumerate(lines):

        tokens = re.findall(
            r"\S+",
            line
        )

        matches = []

        for token in tokens:

            best = None
            best_score = 0

            for j in range(
                cursor,
                min(
                    len(asr),
                    cursor + 35
                )
            ):

                score = similarity(
                    token,
                    asr[j]["word"]
                )

                if score > best_score:
                    best_score = score
                    best = j

                if score >= 0.99:
                    break

            if (
                best is not None
                and best_score >= 0.43
            ):

                matches.append(
                    (
                        token,
                        best
                    )
                )

                cursor = best + 1

        if not matches:

            if cursor >= len(asr):
                break

            start = asr[
                cursor
            ]["start"]

            end = min(
                audio_end,
                start +
                max(
                    0.8,
                    0.30 *
                    len(tokens)
                )
            )

        else:

            start = asr[
                matches[0][1]
            ]["start"]

            end = asr[
                matches[-1][1]
            ]["end"]

        words = align_phrase(
            line,
            start,
            end,
            asr
        )

        for word in words:

            word["phrase_id"] = phrase_id
            word["phrase_text"] = line

        if words:

            scenes.append({
                "start": start,
                "end": min(
                    audio_end,
                    end + 0.18
                ),
                "words": words,
                "phrase_text": line
            })

    return scenes


def auto_scenes(
    words,
    audio_end
):

    if not words:
        return []

    groups = []
    current = []

    phrase_id = words[
        0
    ].get("phrase_id")

    for word in words:

        word_phrase = word.get(
            "phrase_id"
        )

        if (
            current
            and word_phrase is not None
            and word_phrase != phrase_id
        ):

            groups.append(current)
            current = []
            phrase_id = word_phrase

        elif (
            current
            and word_phrase is None
            and
            word["start"]
            - current[-1]["end"]
            > 0.65
        ):

            groups.append(current)
            current = []

        current.append(word)

    if current:
        groups.append(current)

    return [
        {
            "start": max(
                0,
                group[0]["start"] - 0.03
            ),
            "end": min(
                audio_end,
                group[-1]["end"] + 0.18
            ),
            "words": group,
            "phrase_text":
                " ".join(
                    x["word"]
                    for x in group
                )
        }
        for group in groups
    ]


# ============================================================
# DIREÇÃO VISUAL
# ============================================================

def choose_font(
    scene_index,
    word_index,
    phrase_length,
    registry
):

    main = [
        x for x in MAIN_FONTS
        if x in registry
    ] or list(registry)

    alt = [
        x for x in ALT_FONTS
        if x in registry
    ] or main

    # Fonte grossa continua dominante.
    if word_index == 0:

        return main[
            scene_index
            % len(main)
        ]

    # Frases grandes recebem
    # mais variação.
    if (
        phrase_length >= 7
        and word_index % 4 == 2
    ):

        return alt[
            (scene_index + word_index)
            % len(alt)
        ]

    if (
        phrase_length >= 5
        and word_index
        == phrase_length // 2
    ):

        return alt[
            (scene_index * 2 + word_index)
            % len(alt)
        ]

    return main[
        (scene_index + word_index // 3)
        % len(main)
    ]


def blue_word(
    scene_index,
    word_index,
    phrase_length
):

    # Aproximadamente uma frase sim,
    # outra não.
    if scene_index % 2 != 0:
        return False

    if phrase_length < 3:
        return False

    candidates = {
        1,
        phrase_length // 2
    }

    if word_index not in candidates:
        return False

    return True


# ============================================================
# FUNDO MONOCROMÁTICO
# ============================================================

def background(
    width,
    height,
    scene_index,
    local_time,
    scene_duration
):

    current = np.full(
        (height, width, 3),
        BLACK
        if scene_index % 2 == 0
        else WHITE,
        dtype=np.float32
    )

    opposite = np.full(
        (height, width, 3),
        WHITE
        if scene_index % 2 == 0
        else BLACK,
        dtype=np.float32
    )

    # Fade extremamente sutil
    # somente no final.
    fade = clamp(
        (
            scene_duration
            - local_time
        ) / 0.22
    )

    amount = 0

    if local_time > scene_duration - 0.22:
        amount = (
            1 - fade
        ) * 0.18

    array = (
        current * (1 - amount)
        +
        opposite * amount
    )

    return Image.fromarray(
        np.uint8(array),
        "RGB"
    )


# ============================================================
# DESENHO DA PALAVRA
# ============================================================

def draw_word(
    image,
    text,
    font,
    center_x,
    y,
    color,
    progress
):

    """
    CORREÇÃO PRINCIPAL DO BUG.

    O Pillow utiliza offsets no bbox de cada fonte.
    Antes, a camada era criada com a altura do bbox,
    mas o texto era desenhado usando apenas 'pad'.

    Isso podia cortar letras como:
    E
    F
    J
    Á
    Ã
    Ç
    etc.

    Agora usamos:

        tx = pad - bbox[0]
        ty = pad - bbox[1]

    Assim a área real da letra fica totalmente
    dentro da camada.
    """

    progress = clamp(progress)

    e = ease(progress)

    alpha = int(
        255 * e
    )

    bbox = font.getbbox(
        text
    )

    x0 = bbox[0]
    y0 = bbox[1]
    x1 = bbox[2]
    y1 = bbox[3]

    ink_width = max(
        1,
        x1 - x0
    )

    ink_height = max(
        1,
        y1 - y0
    )

    pad = max(
        24,
        int(font.size * 0.18)
    )

    layer_width = (
        ink_width
        + pad * 2
    )

    layer_height = (
        ink_height
        + pad * 2
    )

    layer = Image.new(
        "RGBA",
        (
            layer_width,
            layer_height
        ),
        (0, 0, 0, 0)
    )

    draw = ImageDraw.Draw(
        layer
    )

    # ========================================================
    # CORREÇÃO DO BBOX
    # ========================================================

    text_x = pad - x0
    text_y = pad - y0

    if color == WHITE:

        shadow = (
            0,
            0,
            0,
            int(alpha * 0.25)
        )

    else:

        shadow = (
            255,
            255,
            255,
            int(alpha * 0.25)
        )

    draw.text(
        (
            text_x + 2,
            text_y + 3
        ),
        text,
        font=font,
        fill=shadow
    )

    draw.text(
        (
            text_x,
            text_y
        ),
        text,
        font=font,
        fill=color + (
            alpha,
        )
    )

    # Entrada suave.
    scale = (
        0.94
        + 0.06 * e
    )

    layer = layer.resize(
        (
            max(
                1,
                int(
                    layer.width
                    * scale
                )
            ),
            max(
                1,
                int(
                    layer.height
                    * scale
                )
            )
        ),
        Image.Resampling.LANCZOS
    )

    # ========================================================
    # SEGURANÇA EXTRA
    # ========================================================

    # A camada inteira precisa permanecer
    # dentro da tela.

    px = int(
        center_x
        - layer.width / 2
    )

    py = int(
        y
        + (1 - e) * 22
    )

    px = max(
        0,
        min(
            px,
            image.width
            - layer.width
        )
    )

    py = max(
        0,
        min(
            py,
            image.height
            - layer.height
        )
    )

    image.alpha_composite(
        layer,
        (
            px,
            py
        )
    )


# ============================================================
# RENDER DE FRASE
# ============================================================

def render_scene(
    scene,
    scene_index,
    width,
    height,
    local_time,
    registry
):

    scene_duration = max(
        0.1,
        scene["end"]
        - scene["start"]
    )

    image = background(
        width,
        height,
        scene_index,
        local_time,
        scene_duration
    ).convert("RGBA")

    words = scene.get(
        "words",
        []
    )

    visible = []

    for index, word in enumerate(words):

        relative = (
            local_time
            -
            (
                word["start"]
                -
                scene["start"]
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

        return image.convert(
            "RGB"
        )

    phrase_length = len(
        words
    )

    normal_color = (
        WHITE
        if scene_index % 2 == 0
        else BLACK
    )

    # ========================================================
    # TAMANHO MAIOR
    # ========================================================

    if phrase_length <= 3:

        base_size = int(
            height * 0.105
        )

    elif phrase_length <= 6:

        base_size = int(
            height * 0.090
        )

    else:

        base_size = int(
            height * 0.078
        )

    base_size = max(
        88,
        base_size
    )

    max_width = int(
        width * 0.84
    )

    # ========================================================
    # FRASES EMPILHADAS
    # ========================================================

    stacked = (
        phrase_length >= 7
        and scene_index % 3 == 1
    )

    rows = []

    if stacked:

        for item in visible:

            index, word, relative = item

            font_name = choose_font(
                scene_index,
                index,
                phrase_length,
                registry
            )

            f = fit(
                word["word"].upper(),
                font_name,
                registry,
                base_size,
                int(
                    max_width * 0.92
                )
            )

            word_width = box(
                word["word"].upper(),
                f
            )[2]

            rows.append(
                [
                    (
                        index,
                        word,
                        relative,
                        f,
                        word_width
                    )
                ]
            )

    else:

        current = []
        current_width = 0

        spacing = int(
            base_size * 0.10
        )

        for item in visible:

            index, word, relative = item

            font_name = choose_font(
                scene_index,
                index,
                phrase_length,
                registry
            )

            f = fit(
                word["word"].upper(),
                font_name,
                registry,
                base_size,
                int(
                    max_width * 0.92
                )
            )

            word_width = box(
                word["word"].upper(),
                f
            )[2]

            if (
                current
                and
                current_width
                + spacing
                + word_width
                > max_width
            ):

                rows.append(
                    current
                )

                current = []
                current_width = 0

            current.append(
                (
                    index,
                    word,
                    relative,
                    f,
                    word_width
                )
            )

            current_width += (
                word_width
                +
                (
                    spacing
                    if len(current) > 1
                    else 0
                )
            )

        if current:

            rows.append(
                current
            )

    # ========================================================
    # POSIÇÃO
    # ========================================================

    line_gap = int(
        base_size * 1.10
    )

    total_height = (
        len(rows)
        * line_gap
    )

    start_y = max(
        30,
        int(
            height / 2
            -
            total_height / 2
        )
    )

    # ========================================================
    # DESENHAR
    # ========================================================

    for row_index, row in enumerate(rows):

        spacing = max(
            4,
            int(
                base_size * 0.10
            )
        )

        line_width = (
            sum(
                item[4]
                for item in row
            )
            +
            spacing
            * max(
                0,
                len(row) - 1
            )
        )

        cursor_x = (
            width
            - line_width
        ) / 2

        y = (
            start_y
            +
            row_index
            * line_gap
        )

        for (
            index,
            word,
            relative,
            f,
            word_width
        ) in row:

            text = word[
                "word"
            ].upper()

            use_blue = blue_word(
                scene_index,
                index,
                phrase_length
            )

            color = (
                ROYAL
                if use_blue
                else normal_color
            )

            progress = clamp(
                relative / 0.20
            )

            draw_word(
                image,
                text,
                f,
                cursor_x
                + word_width / 2,
                y,
                color,
                progress
            )

            cursor_x += (
                word_width
                + spacing
            )

    # ========================================================
    # FADE FINAL
    # ========================================================

    if local_time > (
        scene_duration - 0.20
    ):

        amount = int(
            55
            *
            clamp(
                (
                    local_time
                    -
                    (
                        scene_duration
                        - 0.20
                    )
                )
                / 0.20
            )
        )

        overlay = Image.new(
            "RGBA",
            (
                width,
                height
            ),
            (
                0,
                0,
                0,
                amount
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
# RENDERIZAÇÃO
# ============================================================

def render_video(
    audio_path,
    scenes,
    registry,
    output,
    resolution,
    quality,
    progress
):

    width, height = resolution

    song_duration = duration(
        audio_path
    )

    if song_duration <= 0:

        raise RuntimeError(
            "Não foi possível identificar "
            "a duração da música."
        )

    if scenes:

        song_duration = min(
            song_duration,
            max(
                scene["end"]
                for scene in scenes
            )
        )

    ff = ffmpeg()

    silent = Path(
        output
    ).with_name(
        "silent.mp4"
    )

    crf = (
        "14"
        if quality == "Alta qualidade"
        else "18"
    )

    preset = (
        "slow"
        if quality == "Alta qualidade"
        else "medium"
    )

    command = [
        ff,
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

    total_frames = max(
        1,
        int(
            math.ceil(
                song_duration
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
                frame_number
                / FPS
            )

            while (
                scene_index + 1
                < len(scenes)
                and
                current_time
                >=
                scenes[
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
                    "start": 0,
                    "end": song_duration,
                    "words": []
                }

            local_time = (
                current_time
                -
                scene["start"]
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
                progress
                and
                frame_number % FPS == 0
            ):

                progress.progress(
                    min(
                        0.94,
                        frame_number
                        /
                        total_frames
                        * 0.94
                    ),
                    text=(
                        f"Renderizando "
                        f"{int(frame_number / total_frames * 100)}%"
                    )
                )

        process.stdin.close()

        error = (
            process.stderr
            .read()
            .decode(
                "utf-8",
                "replace"
            )
        )

        code = process.wait()

        if code:

            raise RuntimeError(
                "FFmpeg falhou:\n"
                +
                error[-5000:]
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

    # ========================================================
    # ÁUDIO ORIGINAL
    # ========================================================

    final_command = [
        ff,
        "-y",

        "-i",
        str(silent),

        "-i",
        audio_path,

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
        f"{song_duration:.3f}",

        "-movflags",
        "+faststart",

        str(output)
    ]

    run(
        final_command,
        timeout=max(
            180,
            int(
                song_duration
                * 8
            )
        )
    )

    silent.unlink(
        missing_ok=True
    )

    if progress:

        progress.progress(
            1.0,
            text="Vídeo concluído."
        )


# ============================================================
# STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Lyric AI Studio",
    page_icon="🎵",
    layout="centered"
)

st.title(
    "🎵 Lyric AI Studio"
)

st.caption(
    f"Word-Sync Kinetic Engine · "
    f"{APP_VERSION}"
)


audio = st.file_uploader(
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
    "2. Letra oficial",
    height=210,
    placeholder=(
        "Uma frase por linha.\n\n"
        "Com tempos:\n"
        "00:02.3 - 00:06.8 | "
        "Não tenho vergonha de dizer que sou maluco por você"
    )
)


col1, col2 = st.columns(2)

with col1:

    model = st.selectbox(
        "Reconhecimento",
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
            "Alta qualidade"
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
    "Correção principal: o desenho das palavras "
    "agora respeita o bbox real de cada fonte. "
    "Letras como E, F, J, Á e Ã não devem mais "
    "ser cortadas."
)


fonts_registry = registry()

st.caption(
    f"Fontes disponíveis: "
    f"{len(fonts_registry)}"
)


# ============================================================
# BOTÃO
# ============================================================

if st.button(
    "🚀 CRIAR LYRIC VIDEO",
    type="primary",
    use_container_width=True
):

    if not audio:

        st.error(
            "Envie a música primeiro."
        )

        st.stop()

    temp = Path(
        tempfile.mkdtemp(
            prefix="lyric_ai_"
        )
    )

    audio_path = (
        temp
        /
        safe(
            audio.name
        )
    )

    audio_path.write_bytes(
        audio.getbuffer()
    )

    status = st.empty()

    progress = st.progress(
        0,
        text="Preparando…"
    )

    try:

        # ====================================================
        # DURAÇÃO
        # ====================================================

        song_duration = duration(
            str(audio_path)
        )

        if song_duration <= 0:

            raise RuntimeError(
                "Não foi possível identificar "
                "a duração do áudio."
            )

        end_time = real_end(
            str(audio_path),
            song_duration
        )

        status.write(
            f"⏱️ Duração: "
            f"{song_duration:.2f}s · "
            f"fim útil detectado: "
            f"{end_time:.2f}s"
        )


        # ====================================================
        # RECONHECIMENTO
        # ====================================================

        try:

            asr, language = transcribe(
                str(audio_path),
                model,
                status
            )

        except Exception:

            if model != "small":

                status.warning(
                    "Tentando o modelo "
                    "small automaticamente…"
                )

                asr, language = transcribe(
                    str(audio_path),
                    "small",
                    status
                )

            else:

                raise


        # Nunca deixa palavras
        # depois do fim real.

        asr = [
            word
            for word in asr
            if word["start"]
            < end_time
        ]


        # ====================================================
        # LETRA
        # ====================================================

        timed = (
            parse_timed(
                lyrics
            )
            if lyrics.strip()
            else []
        )


        if timed:

            status.write(
                "🕒 Usando os tempos fornecidos "
                "e refinando cada palavra "
                "pelo áudio…"
            )

            scenes = build_timed(
                timed,
                asr,
                end_time
            )

        elif lyrics.strip():

            status.write(
                "🧠 Alinhando a letra oficial "
                "ao canto real…"
            )

            scenes = align_plain(
                lyrics,
                asr,
                end_time
            )

        else:

            scenes = auto_scenes(
                asr,
                end_time
            )


        if not scenes:

            raise RuntimeError(
                "Não foi possível criar "
                "as frases. "
                "Cole a letra oficial "
                "e tente novamente."
            )


        # ====================================================
        # GARANTIR FIM REAL
        # ====================================================

        for scene in scenes:

            scene["end"] = min(
                scene["end"],
                end_time
            )

            scene["words"] = [
                word
                for word in scene["words"]
                if word["start"]
                < end_time
            ]

            for word in scene["words"]:

                word["end"] = min(
                    word["end"],
                    end_time
                )


        progress.progress(
            0.20,
            text="Sincronização pronta."
        )


        # ====================================================
        # RESOLUÇÃO
        # ====================================================

        if resolution.startswith(
            "1080"
        ):

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
            temp
            /
            "lyric_ai_final.mp4"
        )

        status.write(
            "🎬 Renderizando…"
        )

        render_video(
            str(audio_path),
            scenes,
            fonts_registry,
            str(output),
            size,
            quality,
            progress
        )


        status.success(
            "✅ Vídeo criado."
        )


        st.video(
            str(output)
        )


        st.download_button(
            "⬇️ BAIXAR MP4",
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
                f"Fim útil: "
                f"**{end_time:.2f}s**"
            )

            st.write(
                f"Palavras: "
                f"**{sum(len(s['words']) for s in scenes)}**"
            )

            st.write(
                f"Frases: "
                f"**{len(scenes)}**"
            )

            st.write(
                f"Idioma: "
                f"**{language}**"
            )

            st.write(
                "Fundo: "
                "**monocromático preto/branco**"
            )

            st.write(
                "Azul: "
                "**Royal Blue somente em palavras selecionadas**"
            )

            st.write(
                "Tipografia: "
                "**fonte grossa dominante + "
                "variações dentro da própria frase**"
            )


    except Exception as error:

        st.error(
            "❌ A geração falhou."
        )

        st.code(
            str(error)
        )

        st.info(
            "Se o modelo pesado falhar, "
            "selecione 'small'."
        )