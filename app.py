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

APP_VERSION = "10.0-FINAL"
FPS = 30

BLACK = (5, 5, 7)
WHITE = (248, 248, 246)
ROYAL = (45, 92, 255)

CACHE = Path(".lyric_cache")
FONT_DIR = CACHE / "fonts"
FONT_DIR.mkdir(parents=True, exist_ok=True)

FONT_URLS = {
    "Anton": "https://raw.githubusercontent.com/google/fonts/main/ofl/anton/Anton-Regular.ttf",
    "Bebas Neue": "https://raw.githubusercontent.com/google/fonts/main/ofl/bebasneue/BebasNeue-Regular.ttf",
    "Montserrat": "https://raw.githubusercontent.com/google/fonts/main/ofl/montserrat/Montserrat%5Bwght%5D.ttf",
    "Oswald": "https://raw.githubusercontent.com/google/fonts/main/ofl/oswald/Oswald%5Bwght%5D.ttf",
    "Archivo Black": "https://raw.githubusercontent.com/google/fonts/main/ofl/archivoblack/ArchivoBlack-Regular.ttf",
    "DM Serif Display": "https://raw.githubusercontent.com/google/fonts/main/ofl/dmserifdisplay/DMSerifDisplay-Regular.ttf",
    "Playfair Display": "https://raw.githubusercontent.com/google/fonts/main/ofl/playfairdisplay/PlayfairDisplay%5Bwght%5D.ttf",
}

MAIN_FONTS = ["Anton", "Bebas Neue", "Archivo Black", "Oswald"]
ALT_FONTS = ["Montserrat", "DM Serif Display", "Playfair Display"]

def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))

def ease(x):
    x = clamp(x)
    return 1.0 - (1.0 - x) ** 3

def safe(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text or "")[:100] or "audio"

def norm(text):
    return re.sub(r"\s+", " ", text or "").strip()

def normalize_token(text):
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-zA-Z0-9]", "", text).lower()

def similarity(a, b):
    a = normalize_token(a)
    b = normalize_token(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.92
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()

@st.cache_resource(show_spinner=False)
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
    raise RuntimeError("FFmpeg não encontrado. O imageio-ffmpeg precisa estar instalado.")

def run_cmd(cmd, timeout=None):
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-6000:] or "Comando externo falhou.")
    return p.stdout

def media_duration(path):
    p = subprocess.run(
        [get_ffmpeg(), "-hide_banner", "-i", str(path), "-f", "null", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=90,
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", p.stderr)
    if not match:
        return 0.0
    return (
        int(match.group(1)) * 3600
        + int(match.group(2)) * 60
        + float(match.group(3))
    )

def extract_audio(input_path, output_path):
    run_cmd(
        [
            get_ffmpeg(), "-y",
            "-i", str(input_path),
            "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le",
            str(output_path),
        ],
        timeout=180,
    )

def detect_useful_end(path, duration):
    if duration <= 0:
        return duration

    try:
        p = subprocess.run(
            [
                get_ffmpeg(), "-hide_banner", "-i", str(path),
                "-af", "silencedetect=noise=-42dB:d=0.80",
                "-f", "null", "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=max(60, int(duration * 2)),
        )

        starts = [
            float(x)
            for x in re.findall(
                r"silence_start:\s*([\d.]+)",
                p.stderr,
            )
        ]

        ends = [
            float(x)
            for x in re.findall(
                r"silence_end:\s*([\d.]+)",
                p.stderr,
            )
        ]

        if not starts:
            return duration

        threshold = max(
            duration * 0.92,
            duration - 4.0,
        )

        for start in reversed(starts):
            if start < threshold:
                break

            following = [
                e for e in ends
                if e >= start
            ]

            if following:
                return min(
                    duration,
                    following[0] + 0.12,
                )

            return min(
                duration,
                start + 0.18,
            )

    except Exception:
        pass

    return duration

@st.cache_resource(show_spinner=False)
def load_fonts():
    registry = {}

    for name, url in FONT_URLS.items():
        target = FONT_DIR / f"{safe(name)}.ttf"

        if (
            not target.exists()
            or target.stat().st_size < 10000
        ):
            try:
                request = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "LyricAIStudio/10.0"
                    },
                )

                with urllib.request.urlopen(
                    request,
                    timeout=25,
                ) as response:
                    data = response.read()

                if len(data) >= 10000:
                    target.write_bytes(data)

            except Exception:
                target.unlink(
                    missing_ok=True
                )

        if (
            target.exists()
            and target.stat().st_size >= 10000
        ):
            registry[name] = str(target)

    if not registry:
        for path in SYSTEM_FONTS:
            if os.path.exists(path):
                registry["System"] = path
                break

    if not registry:
        raise RuntimeError(
            "Nenhuma fonte utilizável foi encontrada."
        )

    return registry

SYSTEM_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

def get_font(name, registry, size):
    path = (
        registry.get(name)
        or next(iter(registry.values()))
    )

    return ImageFont.truetype(
        path,
        max(12, int(size)),
    )

def text_box(text, font):
    b = font.getbbox(text)

    return (
        b[0],
        b[1],
        max(1, b[2] - b[0]),
        max(1, b[3] - b[1]),
    )

def fit_font(
    text,
    name,
    registry,
    size,
    max_width,
):
    current = max(
        24,
        int(size),
    )

    while current >= 24:
        f = get_font(
            name,
            registry,
            current,
        )

        if text_box(text, f)[2] <= max_width:
            return f

        current -= 2

    return get_font(
        name,
        registry,
        24,
    )

@st.cache_resource(show_spinner=False)
def transcribe_cached(
    audio_path,
    model,
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
                os.cpu_count() or 4,
            ),
        ),
        num_workers=1,
    )

    segments, info = whisper.transcribe(
        str(audio_path),
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
            "Letra de música brasileira em português. "
            "Reconheça todas as palavras, repetições, "
            "gírias, contrações e palavras cantadas rapidamente."
        ),
    )

    words = []

    for segment in segments:
        for w in (segment.words or []):
            word = norm(w.word)

            if not word:
                continue

            start = float(
                w.start or 0.0
            )

            end = float(
                w.end
                or start + 0.08
            )

            if end <= start:
                end = start + 0.08

            words.append(
                {
                    "word": word,
                    "start": start,
                    "end": end,
                    "prob": float(
                        w.probability or 0.0
                    ),
                }
            )

    return (
        words,
        getattr(
            info,
            "language",
            "pt",
        ),
    )

def parse_time(value):
    value = (
        value
        .strip()
        .replace(",", ".")
    )

    parts = value.split(":")

    if len(parts) == 2:
        return (
            int(parts[0]) * 60
            + float(parts[1])
        )

    if len(parts) == 3:
        return (
            int(parts[0]) * 3600
            + int(parts[1]) * 60
            + float(parts[2])
        )

    raise ValueError

def parse_lyrics(text):
    timed = []
    plain_lines = []

    for raw in text.splitlines():
        raw = raw.strip()

        if not raw:
            continue

        m = re.match(
            r"^\s*(\d{1,2}:\d{2}(?:[.,]\d{1,3})?)\s*[-–—]\s*"
            r"(\d{1,2}:\d{2}(?:[.,]\d{1,3})?)\s*\|\s*(.+)$",
            raw,
        )

        if m:
            timed.append(
                {
                    "start": parse_time(
                        m.group(1)
                    ),
                    "end": parse_time(
                        m.group(2)
                    ),
                    "text": norm(
                        m.group(3)
                    ),
                }
            )
            continue

        m = re.match(
            r"^\s*(\d{1,2}:\d{2}(?:[.,]\d{1,3})?)\s*\|\s*(.+)$",
            raw,
        )

        if m:
            timed.append(
                {
                    "start": parse_time(
                        m.group(1)
                    ),
                    "text": norm(
                        m.group(2)
                    ),
                }
            )
            continue

        plain_lines.append(
            norm(raw)
        )

    timed.sort(
        key=lambda x: x["start"]
    )

    return (
        timed,
        plain_lines,
    )

def align_phrase(
    text,
    start,
    end,
    asr,
):
    tokens = re.findall(
        r"\S+",
        text,
    )

    if not tokens:
        return []

    candidates = [
        (i, w)
        for i, w in enumerate(asr)
        if (
            w["end"] >= start - 0.8
            and w["start"] <= end + 0.8
        )
    ]

    mapped = []
    cursor = 0

    for token in tokens:
        best = None
        best_score = 0.0

        upper = min(
            len(candidates),
            cursor + 32,
        )

        for j in range(
            cursor,
            upper,
        ):
            idx, word = candidates[j]

            score = similarity(
                token,
                word["word"],
            )

            if (
                start
                <= word["start"]
                <= end
            ):
                score += 0.06

            if score > best_score:
                best_score = score
                best = (
                    j,
                    idx,
                )

        if (
            best is not None
            and best_score >= 0.38
        ):
            j, idx = best

            mapped.append(
                (
                    token,
                    idx,
                )
            )

            cursor = j + 1

    result = []

    for pos, token in enumerate(tokens):
        if pos < len(mapped):
            _, idx = mapped[pos]
            w = asr[idx]

            result.append(
                {
                    "word": token,
                    "start": max(
                        start,
                        min(
                            end,
                            w["start"],
                        ),
                    ),
                    "end": max(
                        start + 0.06,
                        min(
                            end,
                            w["end"],
                        ),
                    ),
                    "prob": w.get(
                        "prob",
                        0.5,
                    ),
                }
            )

            continue

        previous = [
            i
            for i in range(
                len(mapped)
            )
            if i < pos
        ]

        following = [
            i
            for i in range(
                len(mapped)
            )
            if i >= pos
        ]

        if previous:
            p = mapped[
                previous[-1]
            ][1]

            a = asr[p]["end"]

        else:
            a = start

        if following:
            n = mapped[
                following[0]
            ][1]

            b = asr[n]["start"]

        else:
            b = end

        gap = max(
            0.10,
            b - a,
        )

        left = len(previous)
        right = len(following)

        slots = max(
            1,
            right + 1,
        )

        word_start = (
            a
            + gap
            * (
                left
                / slots
            )
        )

        word_end = (
            a
            + gap
            * (
                (left + 1)
                / slots
            )
        )

        result.append(
            {
                "word": token,
                "start": max(
                    start,
                    min(
                        end,
                        word_start,
                    ),
                ),
                "end": max(
                    start + 0.06,
                    min(
                        end,
                        max(
                            word_start + 0.06,
                            word_end,
                        ),
                    ),
                ),
                "prob": 0.35,
            }
        )

    result.sort(
        key=lambda x: x["start"]
    )

    last_start = start

    for word in result:
        word["start"] = max(
            last_start,
            word["start"],
        )

        word["end"] = min(
            end,
            max(
                word["start"] + 0.055,
                word["end"],
            ),
        )

        if (
            word["end"]
            <= word["start"]
        ):
            word["end"] = min(
                end,
                word["start"] + 0.055,
            )

        last_start = word["start"]

    return result

def build_timed_scenes(
    lines,
    asr,
    lyric_end,
):
    scenes = []

    for i, line in enumerate(lines):
        start = float(
            line["start"]
        )

        if start >= lyric_end:
            continue

        if "end" in line:
            end = float(
                line["end"]
            )

        elif i + 1 < len(lines):
            end = float(
                lines[i + 1]["start"]
            )

        else:
            end = lyric_end

        if i + 1 < len(lines):
            end = min(
                end,
                float(
                    lines[i + 1]["start"]
                ),
            )

        end = max(
            start + 0.15,
            min(
                lyric_end,
                end,
            ),
        )

        if end <= start:
            continue

        words = align_phrase(
            line["text"],
            start,
            end,
            asr,
        )

        if not words:
            continue

        scenes.append(
            {
                "start": start,
                "end": min(
                    lyric_end,
                    end + 0.18,
                ),
                "words": words,
                "phrase_text": line["text"],
            }
        )

    return scenes

def align_plain_scenes(
    lines,
    asr,
    lyric_end,
):
    scenes = []
    cursor = 0

    for phrase_id, line in enumerate(lines):
        tokens = re.findall(
            r"\S+",
            line,
        )

        if not tokens:
            continue

        matches = []

        search_end = min(
            len(asr),
            cursor
            + max(
                40,
                len(tokens) * 6,
            ),
        )

        for token in tokens:
            best = None
            best_score = 0.0

            for j in range(
                cursor,
                search_end,
            ):
                score = similarity(
                    token,
                    asr[j]["word"],
                )

                if score > best_score:
                    best_score = score
                    best = j

                if score >= 0.99:
                    break

            if (
                best is not None
                and best_score >= 0.40
            ):
                matches.append(best)

                cursor = best + 1

                search_end = min(
                    len(asr),
                    cursor
                    + max(
                        40,
                        len(tokens) * 6,
                    ),
                )

        if matches:
            start = asr[
                matches[0]
            ]["start"]

            end = asr[
                matches[-1]
            ]["end"]

        elif cursor < len(asr):
            start = asr[
                cursor
            ]["start"]

            end = (
                start
                + max(
                    0.9,
                    0.32
                    * len(tokens),
                )
            )

        else:
            break

        start = max(
            0.0,
            min(
                lyric_end,
                start,
            ),
        )

        end = max(
            start + 0.15,
            min(
                lyric_end,
                end,
            ),
        )

        if cursor < len(asr):
            end = min(
                end,
                asr[
                    cursor
                ]["start"],
            )

        if end <= start + 0.05:
            end = min(
                lyric_end,
                start
                + max(
                    0.25,
                    0.12
                    * len(tokens),
                ),
            )

        words = align_phrase(
            line,
            start,
            end,
            asr,
        )

        if words:
            for w in words:
                w["phrase_id"] = phrase_id

            scenes.append(
                {
                    "start": start,
                    "end": min(
                        lyric_end,
                        end + 0.18,
                    ),
                    "words": words,
                    "phrase_text": line,
                }
            )

    return scenes

def auto_scenes(
    asr,
    lyric_end,
):
    if not asr:
        return []

    groups = []
    current = []

    for word in asr:
        if word["start"] >= lyric_end:
            break

        if (
            current
            and word["start"]
            - current[-1]["end"]
            > 0.70
        ):
            groups.append(
                current
            )
            current = []

        current.append(word)

    if current:
        groups.append(
            current
        )

    return [
        {
            "start": max(
                0.0,
                group[0]["start"]
                - 0.03,
            ),
            "end": min(
                lyric_end,
                group[-1]["end"]
                + 0.18,
            ),
            "words": group,
            "phrase_text": " ".join(
                w["word"]
                for w in group
            ),
        }
        for group in groups
    ]

def choose_font(
    scene_index,
    word_index,
    phrase_length,
    registry,
):
    main = [
        x
        for x in MAIN_FONTS
        if x in registry
    ]

    alt = [
        x
        for x in ALT_FONTS
        if x in registry
    ]

    main = main or list(registry)
    alt = alt or main

    if word_index == 0:
        return main[
            scene_index
            % len(main)
        ]

    if (
        phrase_length >= 7
        and word_index % 3 == 1
    ):
        return alt[
            (
                scene_index
                + word_index
            )
            % len(alt)
        ]

    if (
        phrase_length >= 5
        and word_index % 5 == 3
    ):
        return alt[
            (
                scene_index * 2
                + word_index
            )
            % len(alt)
        ]

    return main[
        (
            scene_index
            + word_index // 3
        )
        % len(main)
    ]

def use_blue(
    scene_index,
    word_index,
    phrase_length,
):
    if (
        phrase_length < 3
        or scene_index % 2 != 0
    ):
        return False

    return word_index in {
        1,
        phrase_length // 2,
    }

def make_background(
    width,
    height,
    scene_index,
    local_time,
    scene_duration,
):
    current = (
        WHITE
        if scene_index % 2
        else BLACK
    )

    opposite = (
        BLACK
        if scene_index % 2
        else WHITE
    )

    image = Image.new(
        "RGB",
        (width, height),
        current,
    )

    fade_duration = 0.22

    if (
        scene_duration
        > fade_duration
        and local_time
        > scene_duration
        - fade_duration
    ):
        t = clamp(
            (
                local_time
                - (
                    scene_duration
                    - fade_duration
                )
            )
            / fade_duration
        )

        alpha = int(
            255
            * t
            * 0.16
        )

        overlay = Image.new(
            "RGBA",
            (width, height),
            opposite
            + (alpha,),
        )

        image = Image.alpha_composite(
            image.convert("RGBA"),
            overlay,
        ).convert("RGB")

    return image

def draw_word(
    image,
    text,
    font,
    center_x,
    y,
    color,
    progress,
):
    progress = clamp(progress)
    e = ease(progress)

    bbox = font.getbbox(
        text
    )

    x0, y0, x1, y1 = bbox

    ink_width = max(
        1,
        x1 - x0,
    )

    ink_height = max(
        1,
        y1 - y0,
    )

    pad = max(
        28,
        int(
            font.size
            * 0.22
        ),
    )

    layer = Image.new(
        "RGBA",
        (
            ink_width
            + pad * 2,
            ink_height
            + pad * 2,
        ),
        (0, 0, 0, 0),
    )

    draw = ImageDraw.Draw(
        layer
    )

    tx = pad - x0
    ty = pad - y0

    if color in (
        WHITE,
        ROYAL,
    ):
        shadow_color = (
            0,
            0,
            0,
            int(85 * e),
        )
    else:
        shadow_color = (
            255,
            255,
            255,
            int(85 * e),
        )

    draw.text(
        (
            tx + 2,
            ty + 3,
        ),
        text,
        font=font,
        fill=shadow_color,
    )

    draw.text(
        (
            tx,
            ty,
        ),
        text,
        font=font,
        fill=color
        + (
            int(255 * e),
        ),
    )

    scale = (
        0.96
        + 0.04 * e
    )

    layer = layer.resize(
        (
            max(
                1,
                int(
                    layer.width
                    * scale
                ),
            ),
            max(
                1,
                int(
                    layer.height
                    * scale
                ),
            ),
        ),
        Image.Resampling.LANCZOS,
    )

    px = int(
        center_x
        - layer.width / 2
    )

    py = int(y)

    px = max(
        0,
        min(
            px,
            image.width
            - layer.width,
        ),
    )

    py = max(
        0,
        min(
            py,
            image.height
            - layer.height,
        ),
    )

    image.alpha_composite(
        layer,
        (px, py),
    )

def render_scene(
    scene,
    scene_index,
    width,
    height,
    current_time,
    registry,
):
    scene_duration = max(
        0.1,
        scene["end"]
        - scene["start"],
    )

    local_time = (
        current_time
        - scene["start"]
    )

    image = make_background(
        width,
        height,
        scene_index,
        local_time,
        scene_duration,
    ).convert("RGBA")

    words = scene.get(
        "words",
        [],
    )

    visible = []

    for index, word in enumerate(words):
        relative = (
            current_time
            - word["start"]
        )

        if relative >= 0:
            visible.append(
                (
                    index,
                    word,
                    relative,
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
        base_size,
    )

    max_width = int(
        width * 0.86
    )

    stacked = (
        phrase_length >= 7
        and scene_index % 3 == 1
    )

    rows = []

    if stacked:
        for (
            index,
            word,
            relative,
        ) in visible:
            name = choose_font(
                scene_index,
                index,
                phrase_length,
                registry,
            )

            f = fit_font(
                word["word"].upper(),
                name,
                registry,
                base_size,
                int(
                    max_width
                    * 0.88
                ),
            )

            rows.append(
                [
                    (
                        index,
                        word,
                        relative,
                        f,
                        text_box(
                            word["word"].upper(),
                            f,
                        )[2],
                    )
                ]
            )

    else:
        current = []
        current_width = 0
        spacing = max(
            8,
            int(
                base_size
                * 0.08
            ),
        )

        for item in visible:
            (
                index,
                word,
                relative,
            ) = item

            name = choose_font(
                scene_index,
                index,
                phrase_length,
                registry,
            )

            f = fit_font(
                word["word"].upper(),
                name,
                registry,
                base_size,
                int(
                    max_width
                    * 0.92
                ),
            )

            word_width = text_box(
                word["word"].upper(),
                f,
            )[2]

            if (
                current
                and current_width
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
                    word_width,
                )
            )

            current_width += (
                word_width
                + (
                    spacing
                    if len(current) > 1
                    else 0
                )
            )

        if current:
            rows.append(
                current
            )

    line_gap = int(
        base_size
        * 1.05
    )

    total_height = (
        len(rows)
        * line_gap
    )

    start_y = max(
        40,
        int(
            height / 2
            - total_height / 2
        ),
    )

    for row_index, row in enumerate(rows):
        spacing = max(
            8,
            int(
                base_size
                * 0.08
            ),
        )

        line_width = (
            sum(
                item[4]
                for item in row
            )
            + spacing
            * max(
                0,
                len(row) - 1,
            )
        )

        cursor_x = (
            width
            - line_width
        ) / 2

        y = (
            start_y
            + row_index
            * line_gap
        )

        for (
            index,
            word,
            relative,
            f,
            word_width,
        ) in row:
            text = word[
                "word"
            ].upper()

            if (
                word["start"]
                < word["end"]
            ):
                word_progress = clamp(
                    (
                        current_time
                        - word["start"]
                    )
                    / max(
                        0.10,
                        min(
                            0.35,
                            word["end"]
                            - word["start"],
                        ),
                    )
                )
            else:
                word_progress = 1.0

            color = (
                ROYAL
                if use_blue(
                    scene_index,
                    index,
                    phrase_length,
                )
                else normal_color
            )

            draw_word(
                image,
                text,
                f,
                cursor_x
                + word_width / 2,
                y,
                color,
                word_progress,
            )

            cursor_x += (
                word_width
                + spacing
            )

    return image.convert(
        "RGB"
    )

def render_video(
    audio_path,
    scenes,
    registry,
    output,
    resolution,
    quality,
    progress,
):
    width, height = resolution

    song_duration = media_duration(
        audio_path
    )

    if song_duration <= 0:
        raise RuntimeError(
            "Não foi possível identificar a duração da música."
        )

    ff = get_ffmpeg()

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
        str(silent),
    ]

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    total_frames = max(
        1,
        int(
            math.ceil(
                song_duration
                * FPS
            )
        ),
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
                and current_time
                >= scenes[
                    scene_index + 1
                ]["start"]
            ):
                scene_index += 1

            if (
                scenes
                and current_time
                >= scenes[0]["start"]
            ):
                scene = scenes[
                    min(
                        scene_index,
                        len(scenes) - 1,
                    )
                ]

                if current_time < scene["end"]:
                    frame = render_scene(
                        scene,
                        scene_index,
                        width,
                        height,
                        current_time,
                        registry,
                    )

                else:
                    base = (
                        BLACK
                        if scene_index % 2 == 0
                        else WHITE
                    )

                    frame = Image.new(
                        "RGB",
                        (
                            width,
                            height,
                        ),
                        base,
                    )

            else:
                frame = Image.new(
                    "RGB",
                    (
                        width,
                        height,
                    ),
                    BLACK,
                )

            process.stdin.write(
                np.asarray(
                    frame,
                    dtype=np.uint8,
                ).tobytes()
            )

            if (
                progress
                and frame_number % FPS == 0
            ):
                progress.progress(
                    min(
                        0.94,
                        frame_number
                        / total_frames
                        * 0.94,
                    ),
                    text=(
                        f"Renderizando… "
                        f"{current_time:.0f}s / "
                        f"{song_duration:.0f}s"
                    ),
                )

        process.stdin.close()

        stderr = (
            process.stderr
            .read()
            .decode(
                "utf-8",
                errors="replace",
            )
        )

        code = process.wait()

        if code != 0:
            raise RuntimeError(
                stderr[-6000:]
                or "Falha ao renderizar o vídeo."
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

    final_command = [
        ff,
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
        "192k",
        "-t",
        f"{song_duration:.3f}",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output),
    ]

    run_cmd(
        final_command,
        timeout=max(
            240,
            int(
                song_duration
                * 10
            ),
        ),
    )

    silent.unlink(
        missing_ok=True
    )

    if progress:
        progress.progress(
            1.0,
            text="Vídeo concluído.",
        )

st.set_page_config(
    page_title="Lyric AI Studio",
    page_icon="🎵",
    layout="centered",
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
        "webm",
    ],
)

lyrics = st.text_area(
    "2. Letra oficial",
    height=240,
    placeholder=(
        "Uma frase por linha.\n\n"
        "Com tempos:\n"
        "00:02.3 - 00:06.8 | "
        "Não tenho vergonha de dizer que sou maluco por você\n\n"
        "Ou:\n"
        "00:02.3 | "
        "Não tenho vergonha de dizer que sou maluco por você"
    ),
)

col1, col2 = st.columns(2)

with col1:
    model = st.selectbox(
        "Reconhecimento",
        [
            "small",
            "medium",
            "large-v3-turbo",
            "large-v3",
        ],
        index=2,
    )

with col2:
    quality = st.selectbox(
        "Qualidade",
        [
            "Equilibrado",
            "Alta qualidade",
        ],
        index=1,
    )

resolution = st.selectbox(
    "Resolução",
    [
        "1080 × 1920",
        "720 × 1280",
    ],
    index=0,
)

st.info(
    "Versão final: a duração do vídeo segue "
    "a duração real do áudio. As letras são "
    "independentes da duração do render; "
    "as fontes variam dentro das frases, "
    "o azul royal aparece pontualmente e "
    "o fundo permanece monocromático."
)

fonts_registry = load_fonts()

st.caption(
    f"Fontes disponíveis: "
    f"{len(fonts_registry)}"
)

if st.button(
    "🚀 CRIAR LYRIC VIDEO",
    type="primary",
    use_container_width=True,
):
    if not audio:
        st.error(
            "Envie a música ou o vídeo primeiro."
        )
        st.stop()

    temp = Path(
        tempfile.mkdtemp(
            prefix="lyric_ai_"
        )
    )

    input_path = (
        temp
        / safe(audio.name)
    )

    input_path.write_bytes(
        audio.getbuffer()
    )

    status = st.empty()

    progress = st.progress(
        0,
        text="Preparando…",
    )

    try:
        status.write(
            "⏱️ Analisando duração do arquivo…"
        )

        original_duration = media_duration(
            input_path
        )

        if original_duration <= 0:
            raise RuntimeError(
                "Não foi possível ler a duração do arquivo."
            )

        audio_path = (
            temp
            / "audio_16k.wav"
        )

        status.write(
            "🎧 Extraindo o áudio…"
        )

        extract_audio(
            input_path,
            audio_path,
        )

        song_duration = media_duration(
            audio_path
        )

        if song_duration <= 0:
            song_duration = original_duration

        lyric_end = detect_useful_end(
            audio_path,
            song_duration,
        )

        status.write(
            f"⏱️ Música: "
            f"{song_duration:.2f}s · "
            f"fim útil para sincronização: "
            f"{lyric_end:.2f}s"
        )

        status.write(
            "🎙️ Reconhecendo palavra por palavra…"
        )

        try:
            asr, language = transcribe_cached(
                audio_path,
                model,
            )

        except Exception:
            if model == "small":
                raise

            status.warning(
                "O modelo selecionado falhou. "
                "Tentando small automaticamente…"
            )

            asr, language = transcribe_cached(
                audio_path,
                "small",
            )

        asr = [
            w
            for w in asr
            if w["start"] < lyric_end
        ]

        timed_lines, plain_lines = parse_lyrics(
            lyrics
        )

        if timed_lines:
            status.write(
                "🕒 Usando os tempos fornecidos "
                "e refinando palavra por palavra…"
            )

            scenes = build_timed_scenes(
                timed_lines,
                asr,
                lyric_end,
            )

        elif plain_lines:
            status.write(
                "🧠 Alinhando a letra oficial "
                "ao canto real…"
            )

            scenes = align_plain_scenes(
                plain_lines,
                asr,
                lyric_end,
            )

        else:
            status.write(
                "🧠 Gerando cenas automaticamente "
                "pelo reconhecimento…"
            )

            scenes = auto_scenes(
                asr,
                lyric_end,
            )

        if not scenes:
            raise RuntimeError(
                "Não foi possível criar as legendas. "
                "Cole a letra oficial ou verifique o áudio."
            )

        clean_scenes = []

        for scene in scenes:
            scene["start"] = max(
                0.0,
                min(
                    song_duration,
                    scene["start"],
                ),
            )

            scene["end"] = max(
                scene["start"] + 0.08,
                min(
                    lyric_end,
                    scene["end"],
                ),
            )

            scene["words"] = [
                w
                for w in scene["words"]
                if w["start"] < lyric_end
            ]

            for w in scene["words"]:
                w["start"] = max(
                    scene["start"],
                    min(
                        lyric_end,
                        w["start"],
                    ),
                )

                w["end"] = max(
                    w["start"] + 0.05,
                    min(
                        lyric_end,
                        w["end"],
                    ),
                )

            if (
                scene["words"]
                and scene["end"]
                > scene["start"]
            ):
                clean_scenes.append(
                    scene
                )

        scenes = sorted(
            clean_scenes,
            key=lambda s: s["start"],
        )

        if not scenes:
            raise RuntimeError(
                "As legendas não puderam ser sincronizadas."
            )

        progress.progress(
            0.20,
            text="Sincronização pronta.",
        )

        size = (
            (1080, 1920)
            if resolution.startswith("1080")
            else (720, 1280)
        )

        output = (
            temp
            / "lyric_ai_final.mp4"
        )

        status.write(
            "🎬 Renderizando o vídeo inteiro — "
            f"{song_duration:.2f}s de duração…"
        )

        render_video(
            str(audio_path),
            scenes,
            fonts_registry,
            str(output),
            size,
            quality,
            progress,
        )

        status.success(
            "✅ Vídeo criado até o final do áudio."
        )

        st.video(
            str(output)
        )

        st.download_button(
            "⬇️ BAIXAR MP4",
            data=output.read_bytes(),
            file_name="lyric_ai_final.mp4",
            mime="video/mp4",
            use_container_width=True,
        )

        with st.expander(
            "Diagnóstico"
        ):
            st.write(
                f"Duração final: "
                f"**{song_duration:.2f}s**"
            )

            st.write(
                f"Fim útil para letras: "
                f"**{lyric_end:.2f}s**"
            )

            st.write(
                f"Frases: "
                f"**{len(scenes)}**"
            )

            st.write(
                f"Palavras sincronizadas: "
                f"**{sum(len(s['words']) for s in scenes)}**"
            )

            st.write(
                f"Idioma detectado: "
                f"**{language}**"
            )

    except Exception as exc:
        status.error(
            "❌ Não foi possível concluir a renderização."
        )

        st.exception(exc)

    finally:
        pass