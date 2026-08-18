import os, re, math, shutil, subprocess, tempfile, urllib.request, difflib, unicodedata
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageEnhance


# ============================================================
# LYRIC AI STUDIO 6.1
# SMOOTH PHRASE SYNC — BLACK / WHITE / ROYAL BLUE
# ============================================================

APP_VERSION = "6.1-SMOOTH-BW-ROYAL"

W, H = 1080, 1920
FPS = 30

CACHE_DIR = Path(".lyric_cache")
FONT_DIR = CACHE_DIR / "fonts"
FONT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FONTES
# ============================================================

FONT_SOURCES = {
    "Anton":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/anton/Anton-Regular.ttf",

    "Bebas Neue":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/bebasneue/BebasNeue-Regular.ttf",

    "Montserrat":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/montserrat/Montserrat%5Bwght%5D.ttf",

    "Oswald":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/oswald/Oswald%5Bwght%5D.ttf",

    "Playfair Display":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/playfairdisplay/PlayfairDisplay%5Bwght%5D.ttf",

    "Cormorant Garamond":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/cormorantgaramond/CormorantGaramond%5Bwght%5D.ttf",

    "DM Serif Display":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/dmserifdisplay/DMSerifDisplay-Regular.ttf",

    "Archivo Black":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/archivoblack/ArchivoBlack-Regular.ttf",

    "Libre Baskerville":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/librebaskerville/LibreBaskerville-Regular.ttf",

    "Space Mono":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/spacemono/SpaceMono-Regular.ttf",
}


SYSTEM_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
]


# ============================================================
# UTILIDADES
# ============================================================

def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:100]


def clamp(v, a, b):
    return max(a, min(b, v))


def ease_out(t):
    t = clamp(t, 0, 1)
    return 1 - (1 - t) ** 3


def ease_in_out(t):
    t = clamp(t, 0, 1)
    return t * t * (3 - 2 * t)


def normalize_text(s):
    return re.sub(r"\s+", " ", s or "").strip()


# ============================================================
# FALLBACK DE LETRA
# ============================================================

def words_from_manual_lyrics(text, duration):

    lines = [
        normalize_text(x)
        for x in (text or "").splitlines()
        if normalize_text(x)
    ]

    raw = [
        x
        for line in lines
        for x in re.findall(r"\S+", line)
    ]

    if not raw:
        return []

    weights = np.array(
        [
            max(
                1,
                len(re.sub(r"[^\wÀ-ÿ]", "", x))
            ) ** 0.75
            for x in raw
        ],
        float
    )

    weights /= max(weights.sum(), 1)

    out = []
    cur = 0

    for tok, wt in zip(raw, weights):

        st = cur
        en = duration * (cur + float(wt))

        out.append({
            "word": tok,
            "start": st,
            "end": max(st + 0.07, en),
            "prob": 0.4
        })

        cur = en

    return out


# ============================================================
# FFMPEG
# ============================================================

def get_ffmpeg():

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
        "FFmpeg não encontrado. Verifique o requirements.txt."
    )


def run_cmd(cmd, timeout=None):

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


def media_duration(path):

    ff = get_ffmpeg()

    try:

        p = subprocess.run(
            [
                ff,
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
            timeout=45
        )

        text = (p.stderr or "") + (p.stdout or "")

        m = re.search(
            r"Duration:\s*(\d+):(\d+):([\d.]+)",
            text
        )

        if m:

            return (
                int(m.group(1)) * 3600
                + int(m.group(2)) * 60
                + float(m.group(3))
            )

    except Exception:
        pass

    return 0.0


# ============================================================
# FONTES
# ============================================================

def download_font(name):

    target = FONT_DIR / (
        safe_name(name) + ".ttf"
    )

    if (
        target.exists()
        and target.stat().st_size > 10000
    ):
        return str(target)

    url = FONT_SOURCES.get(name)

    if not url:
        return None

    try:

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "LyricAI/6.1"
            }
        )

        with urllib.request.urlopen(
            req,
            timeout=20
        ) as r, open(target, "wb") as f:

            f.write(r.read())

        if target.stat().st_size > 10000:
            return str(target)

    except Exception:

        target.unlink(
            missing_ok=True
        )

    return None


@st.cache_resource(show_spinner=False)
def load_font_registry():

    reg = {}

    for name in FONT_SOURCES:

        p = download_font(name)

        if p:
            reg[name] = p

    if not reg:

        for p in SYSTEM_FONT_CANDIDATES:

            if os.path.exists(p):

                reg["System fallback"] = p
                break

    return reg


def font_path(name, registry):

    if name in registry:
        return registry[name]

    for p in SYSTEM_FONT_CANDIDATES:

        if os.path.exists(p):
            return p

    raise RuntimeError(
        "Nenhuma fonte compatível encontrada."
    )


def fit_font(text, max_width, start_size, path):

    size = int(start_size)

    while size >= 20:

        f = ImageFont.truetype(
            path,
            size=size
        )

        b = f.getbbox(text)

        if b[2] - b[0] <= max_width:
            return f

        size -= 2

    return ImageFont.truetype(
        path,
        size=20
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


def transcribe_audio(
    path,
    model_name,
    status=None
):

    model = get_whisper(
        model_name
    )

    if status:

        status.write(
            f"Transcrevendo com **{model_name}** "
            f"e timestamps por palavra…"
        )

    segments, info = model.transcribe(

        path,

        language="pt",
        task="transcribe",

        beam_size=5,
        best_of=5,
        patience=1,

        temperature=0,

        vad_filter=False,

        word_timestamps=True,

        condition_on_previous_text=True,

        initial_prompt=(
            "Letra de música brasileira em português. "
            "Preserve palavras, repetições, gírias, "
            "contrações e nomes próprios."
        )
    )

    words = []

    for seg in segments:

        if not seg.words:
            continue

        for w in seg.words:

            txt = normalize_text(
                w.word
            )

            if txt:

                words.append({

                    "word": txt,

                    "start": float(
                        w.start
                    ),

                    "end": float(
                        w.end
                    ),

                    "prob": float(
                        getattr(
                            w,
                            "probability",
                            0
                        ) or 0
                    )
                })

    return (
        words,
        getattr(
            info,
            "language",
            "pt"
        ),
        float(
            getattr(
                info,
                "duration",
                0
            ) or 0
        )
    )


# ============================================================
# ALINHAMENTO DA LETRA OFICIAL
# ============================================================

def _norm_token(x):

    x = unicodedata.normalize(
        "NFKD",
        x or ""
    )

    x = "".join(
        c
        for c in x
        if not unicodedata.combining(c)
    )

    return re.sub(
        r"[^a-zA-Z0-9]",
        "",
        x
    ).lower()


def _token_similarity(a, b):

    a = _norm_token(a)
    b = _norm_token(b)

    if not a or not b:
        return 0

    if a == b:
        return 1

    if a in b or b in a:
        return 0.88

    return difflib.SequenceMatcher(
        None,
        a,
        b,
        autojunk=False
    ).ratio()


def align_manual_lyrics(
    manual_text,
    asr_words,
    duration
):

    lines = [
        normalize_text(x)
        for x in (manual_text or "").splitlines()
        if normalize_text(x)
    ]

    if not lines:
        return []

    if not asr_words:
        return words_from_manual_lyrics(
            manual_text,
            duration
        )

    out = []

    cursor = 0
    n = len(asr_words)

    for pid, line in enumerate(lines):

        toks = re.findall(
            r"\S+",
            line
        )

        if not toks:
            continue

        mapped = []
        search_start = cursor

        for tok in toks:

            best_idx = None
            best_score = 0

            for j in range(
                search_start,
                min(
                    n,
                    search_start + 28
                )
            ):

                score = _token_similarity(
                    tok,
                    asr_words[j]["word"]
                )

                if score > best_score:

                    best_idx = j
                    best_score = score

                if score >= 0.995:
                    break

            if (
                best_idx is not None
                and best_score >= 0.58
            ):

                mapped.append(
                    (
                        tok,
                        best_idx,
                        best_score
                    )
                )

                search_start = (
                    best_idx + 1
                )

        if mapped:

            phrase_start = float(
                asr_words[
                    mapped[0][1]
                ]["start"]
            )

            phrase_end = float(
                asr_words[
                    mapped[-1][1]
                ]["end"]
            )

        else:

            phrase_start = float(
                asr_words[
                    min(cursor, n - 1)
                ]["start"]
            )

            phrase_end = min(
                duration,
                phrase_start + 0.2 * len(toks)
            )

        positions = [
            None
            for _ in toks
        ]

        pool = mapped.copy()

        for i, tok in enumerate(toks):

            for k, (
                mt,
                idx,
                score
            ) in enumerate(pool):

                if mt == tok:

                    positions[i] = (
                        idx,
                        score
                    )

                    pool.pop(k)
                    break

        known = [
            (i, p[0])
            for i, p in enumerate(
                positions
            )
            if p
        ]

        for i, tok in enumerate(toks):

            if positions[i]:

                idx, score = positions[i]

                st = float(
                    asr_words[idx]["start"]
                )

                en = float(
                    asr_words[idx]["end"]
                )

                prob = max(
                    float(
                        asr_words[idx].get(
                            "prob",
                            0
                        )
                    ),
                    score * 0.75
                )

            else:

                prev = [
                    x
                    for x in known
                    if x[0] < i
                ]

                nxt = [
                    x
                    for x in known
                    if x[0] > i
                ]

                if prev:

                    base = float(
                        asr_words[
                            prev[-1][1]
                        ]["end"]
                    )

                else:

                    base = phrase_start

                if nxt:

                    target = float(
                        asr_words[
                            nxt[0][1]
                        ]["start"]
                    )

                else:

                    target = phrase_end

                pi = (
                    prev[-1][0]
                    if prev
                    else -1
                )

                ni = (
                    nxt[0][0]
                    if nxt
                    else len(toks)
                )

                frac = (
                    (i - pi)
                    /
                    max(
                        1,
                        ni - pi
                    )
                )

                st = (
                    base
                    +
                    (target - base)
                    *
                    max(
                        0,
                        frac
                        -
                        1 / max(
                            1,
                            ni - pi
                        )
                    )
                )

                en = (
                    base
                    +
                    (target - base)
                    * frac
                )

                st = max(
                    phrase_start,
                    st
                )

                en = max(
                    st + 0.055,
                    min(
                        phrase_end,
                        en
                    )
                )

                prob = 0.5

            out.append({

                "word": tok,

                "start": max(
                    0,
                    st
                ),

                "end": max(
                    st + 0.055,
                    en
                ),

                "prob": prob,

                "phrase_id": pid,

                "phrase_text": line
            })

        if mapped:

            cursor = (
                mapped[-1][1]
                + 1
            )

    return out


# ============================================================
# LIMPEZA
# ============================================================

def clean_transcription(words):

    out = []

    for w in words:

        txt = normalize_text(
            w["word"]
        )

        if not txt:
            continue

        if len(txt) > 35:
            continue

        d = dict(w)
        d["word"] = txt

        out.append(d)

    prev = 0

    for w in out:

        w["start"] = max(
            float(w["start"]),
            prev
        )

        w["end"] = max(
            float(w["end"]),
            w["start"] + 0.06
        )

        prev = w["end"]

    return out


# ============================================================
# SEGMENTAÇÃO EM FRASES
# ============================================================

def segment_lyrics(
    words,
    max_words=18,
    max_seconds=8.5
):

    if not words:
        return []

    scenes = []

    cur = []

    manual = any(
        "phrase_id" in w
        for w in words
    )

    pid = None

    for w in words:

        if not cur:

            cur = [w]

            pid = (
                w.get("phrase_id")
                if manual
                else None
            )

            continue

        if (
            manual
            and
            w.get("phrase_id")
            != pid
        ):

            scenes.append(cur)

            cur = [w]

            pid = w.get(
                "phrase_id"
            )

            continue

        gap = (
            float(w["start"])
            -
            float(cur[-1]["end"])
        )

        punctuation = bool(
            re.search(
                r"[.!?;:]$",
                cur[-1]["word"]
            )
        )

        too_long = (
            len(cur) + 1
            > max_words
            or
            float(w["end"])
            -
            float(cur[0]["start"])
            > max_seconds
        )

        if (
            gap > 0.72
            or punctuation
            or too_long
        ):

            scenes.append(cur)

            cur = [w]

        else:

            cur.append(w)

    if cur:
        scenes.append(cur)

    result = []

    for ws in scenes:

        start = float(
            ws[0]["start"]
        )

        last_word_end = float(
            ws[-1]["end"]
        )

        # Dá tempo para a frase respirar
        # e desaparecer suavemente.
        end = (
            last_word_end
            + 0.42
        )

        result.append({

            "words": ws,

            "start": max(
                0,
                start - 0.03
            ),

            "end": end,

            "instrumental": False,

            "phrase_text":
                " ".join(
                    w["word"]
                    for w in ws
                )
        })

    return result


def ensure_coverage(
    scenes,
    duration
):

    if not scenes:
        return []

    out = []

    for i, s in enumerate(scenes):

        s = dict(s)

        if i == 0:

            s["start"] = max(
                0,
                min(
                    s["start"],
                    0.05
                )
            )

        if (
            i > 0
            and
            s["start"]
            -
            out[-1]["end"]
            > 1.15
        ):

            out.append({

                "words": [],

                "start":
                    out[-1]["end"],

                "end":
                    s["start"],

                "instrumental": True
            })

        s["instrumental"] = False

        out.append(s)

    if (
        out[-1]["end"]
        <
        duration - 0.15
    ):

        out.append({

            "words": [],

            "start":
                out[-1]["end"],

            "end":
                duration,

            "instrumental": True
        })

    return out


# ============================================================
# DIREÇÃO VISUAL
# ============================================================

@dataclass
class Style:

    bg: Tuple[int, int, int]

    fg: Tuple[int, int, int]

    accent: Tuple[int, int, int]

    muted: Tuple[int, int, int]

    font: str

    display_font: str

    layout: str

    grain: float


ROYAL_BLUE = (
    38,
    86,
    255
)


PALETTES = [

    (
        (7, 8, 10),
        (247, 247, 245),
        ROYAL_BLUE,
        (120, 124, 135)
    ),

    (
        (12, 13, 16),
        (250, 250, 248),
        ROYAL_BLUE,
        (105, 110, 122)
    ),

    (
        (3, 4, 7),
        (242, 244, 248),
        ROYAL_BLUE,
        (92, 98, 112)
    )
]


SERIF = [
    "Playfair Display",
    "Cormorant Garamond",
    "DM Serif Display",
    "Libre Baskerville"
]


SANS = [
    "Montserrat",
    "Oswald",
    "Anton",
    "Archivo Black"
]


def choose_style(
    scene,
    features,
    registry,
    seed=0,
    global_theme="Black & Royal Blue"
):

    text = scene.get(
        "phrase_text",
        ""
    )

    energy = energy_at(
        features,
        (
            scene["start"]
            +
            scene["end"]
        ) / 2
    )

    n = len(
        scene.get(
            "words",
            []
        )
    )

    h = (
        abs(hash(text))
        +
        seed
    )

    p = PALETTES[
        h % len(PALETTES)
    ]

    if (
        energy > 0.65
        or n <= 2
    ):

        first = SANS
        second = SERIF

    else:

        first = SERIF
        second = SANS

    fchoices = [
        x
        for x in first
        if x in registry
    ] or list(registry)

    dchoices = [
        x
        for x in second
        if x in registry
    ] or list(registry)

    return Style(

        p[0],
        p[1],
        p[2],
        p[3],

        fchoices[
            h % len(fchoices)
        ],

        dchoices[
            (h // 3)
            % len(dchoices)
        ],

        [
            "hero",
            "center",
            "editorial"
        ][h % 3],

        0.012
    )


# ============================================================
# ÁUDIO / ENERGIA
# ============================================================

def audio_features(
    path,
    duration
):

    ff = get_ffmpeg()

    try:

        p = subprocess.run(

            [
                ff,
                "-v",
                "error",
                "-i",
                path,
                "-ac",
                "1",
                "-ar",
                "8000",
                "-f",
                "s16le",
                "-"
            ],

            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,

            timeout=max(
                60,
                int(duration * 3)
            )
        )

        x = np.frombuffer(
            p.stdout,
            dtype=np.int16
        ).astype(
            np.float32
        )

        if not x.size:

            return {
                "rms": [],
                "duration": duration
            }

        hop = 2000

        arr = np.array([

            np.sqrt(
                np.mean(
                    x[i:i + hop] ** 2
                )
                + 1e-8
            )

            for i in range(
                0,
                len(x),
                hop
            )

        ])

        lo, hi = np.percentile(
            arr,
            [10, 90]
        )

        arr = np.clip(
            (
                arr - lo
            )
            /
            (
                hi - lo + 1e-6
            ),
            0,
            1
        )

        return {

            "rms":
                arr.tolist(),

            "duration":
                duration
        }

    except Exception:

        return {
            "rms": [],
            "duration": duration
        }


def energy_at(
    features,
    t
):

    a = features.get(
        "rms",
        []
    )

    if not a:
        return 0.5

    idx = int(
        clamp(
            t
            /
            max(
                features.get(
                    "duration",
                    1
                ),
                1
            )
            *
            len(a),

            0,

            len(a) - 1
        )
    )

    return float(
        a[idx]
    )


# ============================================================
# BACKGROUND
# ============================================================

def make_background(
    size,
    style,
    t,
    background_frame=None
):

    w, h = size

    if background_frame is not None:

        img = (
            Image
            .fromarray(
                background_frame
            )
            .convert("RGB")
            .resize(
                (w, h),
                Image.Resampling.LANCZOS
            )
        )

        img = ImageEnhance.Color(
            img
        ).enhance(
            0.08
        )

        img = ImageEnhance.Contrast(
            img
        ).enhance(
            1.16
        )

        img = ImageEnhance.Brightness(
            img
        ).enhance(
            0.58
        )

        return img

    base = np.zeros(
        (h, w, 3),
        np.float32
    )

    base[:] = np.array(
        style.bg,
        np.float32
    )

    yy, xx = np.mgrid[
        0:h,
        0:w
    ]

    fields = [

        (
            w * .2
            +
            w * .05
            * math.sin(t / 5),

            h * .25,

            .12,

            .65
        ),

        (
            w * .8,

            h * .7
            +
            h * .04
            * math.sin(t / 4),

            .08,

            .55
        )
    ]

    for (
        cx,
        cy,
        amp,
        scale
    ) in fields:

        d = (
            (
                (xx - cx)
                /
                (w * scale)
            ) ** 2
            +
            (
                (yy - cy)
                /
                (h * scale)
            ) ** 2
        )

        field = np.exp(
            -2.4 * d
        )[..., None]

        base = (
            base
            *
            (
                1
                -
                field * amp
            )
            +
            np.array(
                style.accent,
                np.float32
            )
            *
            field
            *
            amp
        )

    rng = np.random.default_rng(
        int(t * 1000)
        % 1000003
    )

    noise = rng.normal(
        0,
        255 * style.grain,
        (h, w, 1)
    )

    base = np.clip(
        base + noise,
        0,
        255
    )

    return Image.fromarray(
        np.uint8(base)
    )


# ============================================================
# TEXTO
# ============================================================

def text_bbox(
    draw,
    text,
    font
):

    b = draw.textbbox(
        (0, 0),
        text,
        font=font
    )

    return (
        b[2] - b[0],
        b[3] - b[1]
    )


def word_importance(
    token
):

    clean = re.sub(
        r"[^\wÀ-ÿ]",
        "",
        token.lower()
    )

    emotional = {

        "amor",
        "saudade",
        "beijo",
        "coração",
        "você",
        "voce",
        "nunca",
        "sempre",
        "volta",
        "embora",
        "ciúmes",
        "ciumes",
        "perfume",
        "vida",
        "desejo",
        "paixão",
        "paixao",
        "sofrer",
        "chora",
        "chorar",
        "quero",
        "meu",
        "minha",
        "tudo",
        "nada"
    }

    return (
        clean in emotional
        or len(clean) >= 9
    )


# ============================================================
# ELEMENTOS VISUAIS
# ============================================================

def draw_decor(
    d,
    style,
    W,
    H,
    t,
    seed
):

    pulse = (
        0.5
        +
        0.5
        *
        math.sin(
            t * 2.0
            +
            seed
        )
    )

    alpha = int(
        55
        +
        25 * pulse
    )

    # Linhas finas azuis
    d.line(
        (
            W * .10,
            H * .18,
            W * (
                .25
                +
                .04 * pulse
            ),
            H * .18
        ),
        fill=style.accent + (
            alpha,
        ),
        width=4
    )

    d.line(
        (
            W * (
                .75
                -
                .04 * pulse
            ),
            H * .82,
            W * .90,
            H * .82
        ),
        fill=style.accent + (
            alpha,
        ),
        width=4
    )

    # Partículas muito discretas
    for k in range(7):

        ang = (
            t
            *
            (
                .22
                +
                .035 * k
            )
            +
            seed * .013
            +
            k
        )

        x = W * (
            .12
            +
            .76
            *
            (
                (
                    math.sin(ang)
                    + 1
                ) / 2
            )
        )

        y = H * (
            .17
            +
            .66
            *
            (
                (
                    math.cos(
                        ang * 1.21
                    )
                    + 1
                ) / 2
            )
        )

        r = max(
            2,
            int(
                W
                *
                (
                    .0025
                    +
                    .0012
                    *
                    (k % 3)
                )
            )
        )

        d.ellipse(
            (
                x - r,
                y - r,
                x + r,
                y + r
            ),
            fill=style.accent + (
                22 + k * 4,
            )
        )

    # Barra inferior
    d.rounded_rectangle(
        (
            W * .18,
            H * .875,
            W * .82,
            H * .875 + 5
        ),
        radius=3,
        fill=style.muted + (
            35,
        )
    )


# ============================================================
# RENDER DA FRASE
# ============================================================

def render_scene_frame(
    scene,
    style,
    registry,
    W,
    H,
    t_local,
    duration,
    bg_img
):

    img = bg_img.convert(
        "RGBA"
    )

    overlay = Image.new(
        "RGBA",
        (W, H),
        (0, 0, 0, 0)
    )

    d = ImageDraw.Draw(
        overlay
    )

    words = scene.get(
        "words",
        []
    )

    # --------------------------------------------------------
    # CENA INSTRUMENTAL
    # --------------------------------------------------------

    if not words:

        draw_decor(
            d,
            style,
            W,
            H,
            t_local,
            17
        )

        return Image.alpha_composite(
            img,
            overlay
        ).convert("RGB")

    # --------------------------------------------------------
    # FADE OUT DA FRASE
    # --------------------------------------------------------

    fade_out = 0.38

    phrase_alpha = 1.0

    if duration > fade_out:

        phrase_alpha = clamp(
            (
                duration
                -
                t_local
            )
            /
            fade_out,

            0,

            1
        )

    # --------------------------------------------------------
    # DECORAÇÃO
    # --------------------------------------------------------

    seed = (
        sum(
            ord(c)
            for c
            in scene.get(
                "phrase_text",
                ""
            )
        )
        % 97
    )

    draw_decor(
        d,
        style,
        W,
        H,
        t_local,
        seed
    )

    # --------------------------------------------------------
    # PALAVRAS QUE JÁ FORAM CANTADAS
    # --------------------------------------------------------

    spoken = []

    for idx, w in enumerate(words):

        relative_start = (
            float(w["start"])
            -
            float(scene["start"])
        )

        if (
            t_local
            >=
            relative_start
            -
            0.01
        ):

            spoken.append(
                (
                    idx,
                    w,
                    relative_start
                )
            )

    if not spoken:

        return Image.alpha_composite(
            img,
            overlay
        ).convert("RGB")

    # --------------------------------------------------------
    # FONTE
    # --------------------------------------------------------

    fpath = font_path(
        style.font,
        registry
    )

    text = " ".join(
        x[1]["word"].upper()
        for x in spoken
    )

    maxw = int(
        W * .82
    )

    if len(spoken) > 7:

        base_size = int(
            H * .072
        )

    else:

        base_size = int(
            H * .082
        )

    if len(spoken) <= 3:

        base_size = int(
            H * .09
        )

    font = fit_font(
        text,
        maxw,
        base_size,
        fpath
    )

    # --------------------------------------------------------
    # QUEBRA POR LARGURA REAL
    # --------------------------------------------------------

    rows = []
    row = []
    roww = 0

    space = text_bbox(
        d,
        " ",
        font
    )[0]

    for item in spoken:

        token = item[1][
            "word"
        ].upper()

        ww = text_bbox(
            d,
            token,
            font
        )[0]

        if (
            row
            and
            roww
            +
            space
            +
            ww
            >
            maxw
        ):

            rows.append(
                (
                    row,
                    roww
                )
            )

            row = [item]
            roww = ww

        else:

            row.append(
                item
            )

            roww = (
                ww
                if not roww
                else
                roww
                +
                space
                +
                ww
            )

    if row:

        rows.append(
            (
                row,
                roww
            )
        )

    # --------------------------------------------------------
    # POSIÇÃO
    # --------------------------------------------------------

    gap = int(
        font.size
        * 1.10
    )

    total = (
        gap
        *
        len(rows)
    )

    y0 = (
        H * .50
        -
        total / 2
    )

    newest = spoken[-1][0]

    # --------------------------------------------------------
    # DESENHO PALAVRA POR PALAVRA
    # --------------------------------------------------------

    for ri, (
        row,
        roww
    ) in enumerate(rows):

        cursor = (
            W - roww
        ) / 2

        for idx, w, relative_start in row:

            token = w[
                "word"
            ].upper()

            ww = text_bbox(
                d,
                token,
                font
            )[0]

            age = (
                t_local
                -
                relative_start
            )

            # Entrada suave
            p = clamp(
                age / .26,
                0,
                1
            )

            e = ease_out(
                p
            )

            # Movimento pequeno,
            # nada de "pop".
            yy = (
                y0
                +
                ri * gap
                +
                (1 - e)
                * 18
            )

            # Fade da palavra
            alpha = int(
                255
                *
                e
                *
                phrase_alpha
            )

            important = word_importance(
                token
            )

            # Azul apenas para palavras
            # de impacto.
            if important:

                col = style.accent

            else:

                col = style.fg

            # ------------------------------------------------
            # GLOW SUAVE DURANTE A ENTRADA
            # ------------------------------------------------

            if (
                age >= 0
                and
                age < .34
            ):

                halo = int(
                    35
                    *
                    (
                        1
                        -
                        age / .34
                    )
                    *
                    phrase_alpha
                )

                d.text(

                    (
                        cursor,
                        yy
                    ),

                    token,

                    font=font,

                    fill=col + (
                        halo,
                    ),

                    stroke_width=max(
                        2,
                        int(
                            font.size
                            * .035
                        )
                    ),

                    stroke_fill=(
                        style.accent
                        +
                        (
                            halo,
                        )
                    )
                )

            # ------------------------------------------------
            # SOMBRA
            # ------------------------------------------------

            d.text(

                (
                    cursor + 2,
                    yy + 3
                ),

                token,

                font=font,

                fill=(
                    0,
                    0,
                    0,
                    int(
                        70
                        *
                        e
                        *
                        phrase_alpha
                    )
                )
            )

            # ------------------------------------------------
            # TEXTO PRINCIPAL
            # ------------------------------------------------

            d.text(

                (
                    cursor,
                    yy
                ),

                token,

                font=font,

                fill=(
                    col
                    +
                    (
                        alpha,
                    )
                )
            )

            # ------------------------------------------------
            # SUBLINHADO AZUL DISCRETO
            # ------------------------------------------------

            if (
                important
                or
                idx == newest
            ):

                underline_progress = clamp(
                    age / .55,
                    0,
                    1
                )

                underline_alpha = int(
                    125
                    *
                    (
                        1
                        -
                        underline_progress
                    )
                    *
                    phrase_alpha
                )

                uw = int(
                    ww * .52
                )

                ux = (
                    cursor
                    +
                    (
                        ww - uw
                    ) / 2
                )

                uy = (
                    yy
                    +
                    font.size
                    * 1.03
                )

                d.rounded_rectangle(

                    (
                        ux,
                        uy,
                        ux + uw,
                        uy
                        +
                        max(
                            2,
                            int(
                                font.size
                                * .018
                            )
                        )
                    ),

                    radius=2,

                    fill=(
                        style.accent
                        +
                        (
                            underline_alpha,
                        )
                    )
                )

            cursor += (
                ww
                +
                space
            )

    return Image.alpha_composite(
        img,
        overlay
    ).convert("RGB")


# ============================================================
# VÍDEO DE FUNDO
# ============================================================

def video_info(path):

    import cv2

    cap = cv2.VideoCapture(
        path
    )

    if not cap.isOpened():
        return None

    fps = (
        cap.get(
            cv2.CAP_PROP_FPS
        )
        or 30
    )

    frames = (
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
        or 0
    )

    w = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
        or 0
    )

    h = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
        or 0
    )

    cap.release()

    return {

        "fps": fps,

        "frames": frames,

        "w": w,

        "h": h,

        "duration":
            frames / fps
            if fps
            else 0
    }


def fit_crop_frame(
    frame,
    W,
    H
):

    import cv2

    if frame is None:
        return None

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    h, w = frame.shape[:2]

    target = W / H
    ratio = w / h

    if ratio > target:

        nw = int(
            h * target
        )

        x = (
            w - nw
        ) // 2

        frame = frame[
            :,
            x:x + nw
        ]

    else:

        nh = int(
            w / target
        )

        y = (
            h - nh
        ) // 2

        frame = frame[
            y:y + nh,
            :
        ]

    return cv2.resize(
        frame,
        (W, H),
        interpolation=cv2.INTER_AREA
    )


# ============================================================
# RENDERIZAÇÃO FINAL
# ============================================================

def render_video(

    audio_path,
    background_path,
    scenes,
    registry,
    theme,
    out_path,

    resolution=(720, 1280),

    fps=30,

    quality="Equilibrado",

    progress=None

):

    import cv2

    W, H = resolution

    duration = media_duration(
        audio_path
    )

    if (
        duration <= 0
        and scenes
    ):

        duration = max(
            s["end"]
            for s in scenes
        )

    bgcap = None
    bg_static = None

    if background_path:

        info = video_info(
            background_path
        )

        if info:

            bgcap = cv2.VideoCapture(
                background_path
            )

        else:

            try:

                bg_static = np.asarray(
                    Image.open(
                        background_path
                    ).convert(
                        "RGB"
                    )
                )

            except Exception:

                bg_static = None

    ff = get_ffmpeg()

    silent = Path(
        out_path
    ).with_suffix(
        ".silent.mp4"
    )

    if quality == "Alta qualidade":

        crf = "14"
        preset = "slow"

    else:

        crf = "17"
        preset = "medium"

    enc = [

        ff,

        "-y",

        "-f",
        "rawvideo",

        "-vcodec",
        "rawvideo",

        "-pix_fmt",
        "rgb24",

        "-s",
        f"{W}x{H}",

        "-r",
        str(fps),

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

    proc = subprocess.Popen(

        enc,

        stdin=subprocess.PIPE,

        stdout=subprocess.PIPE,

        stderr=subprocess.PIPE
    )

    total = max(
        1,
        int(
            math.ceil(
                duration * fps
            )
        )
    )

    scene_i = 0

    try:

        for frame_i in range(
            total
        ):

            t = (
                frame_i
                / fps
            )

            # ----------------------------------------------
            # LOCALIZA A FRASE ATUAL
            # ----------------------------------------------

            while (
                scene_i + 1
                <
                len(scenes)
                and
                t
                >=
                scenes[
                    scene_i + 1
                ]["start"]
            ):

                scene_i += 1

            scene = (

                scenes[
                    min(
                        scene_i,
                        len(scenes) - 1
                    )
                ]

                if scenes

                else {

                    "start": 0,
                    "end": duration,
                    "words": [],
                    "instrumental": True
                }
            )

            # ----------------------------------------------
            # FUNDO
            # ----------------------------------------------

            bgframe = None

            if bgcap is not None:

                bgcap.set(
                    cv2.CAP_PROP_POS_MSEC,
                    t * 1000
                )

                ok, fr = (
                    bgcap.read()
                )

                if ok:

                    bgframe = fit_crop_frame(
                        fr,
                        W,
                        H
                    )

            elif bg_static is not None:

                bgframe = fit_crop_frame(
                    bg_static,
                    W,
                    H
                )

            # ----------------------------------------------
            # ESTILO
            # ----------------------------------------------

            style = choose_style(

                scene,

                FEATURES_GLOBAL,

                registry,

                scene_i,

                theme
            )

            bg = make_background(

                (W, H),

                style,

                t,

                bgframe
            )

            local = clamp(

                t
                -
                scene["start"],

                0,

                max(
                    .001,
                    scene["end"]
                    -
                    scene["start"]
                )
            )

            scene_dur = max(

                .001,

                scene["end"]
                -
                scene["start"]
            )

            final = render_scene_frame(

                scene,

                style,

                registry,

                W,

                H,

                local,

                scene_dur,

                bg
            )

            # =================================================
            # TRANSIÇÃO SUAVE ENTRE FRASES
            # =================================================

            if scene_i > 0:

                previous = scenes[
                    scene_i - 1
                ]

                transition_time = 0.46

                incoming_age = (
                    t
                    -
                    scene["start"]
                )

                if (
                    0
                    <=
                    incoming_age
                    <
                    transition_time
                ):

                    previous_style = choose_style(

                        previous,

                        FEATURES_GLOBAL,

                        registry,

                        scene_i - 1,

                        theme
                    )

                    previous_duration = max(

                        .001,

                        previous["end"]
                        -
                        previous["start"]
                    )

                    previous_bg = make_background(

                        (W, H),

                        previous_style,

                        max(
                            0,
                            previous["end"]
                            -
                            .08
                        ),

                        bgframe
                    )

                    outgoing = render_scene_frame(

                        previous,

                        previous_style,

                        registry,

                        W,

                        H,

                        previous_duration,

                        previous_duration,

                        previous_bg
                    )

                    transition_progress = ease_in_out(

                        incoming_age
                        /
                        transition_time
                    )

                    final = Image.blend(

                        outgoing,

                        final,

                        transition_progress
                    )

            # ----------------------------------------------
            # ESCREVE FRAME
            # ----------------------------------------------

            proc.stdin.write(

                np.asarray(
                    final.convert(
                        "RGB"
                    ),
                    dtype=np.uint8
                ).tobytes()
            )

            if (
                progress
                and
                frame_i % max(
                    1,
                    fps
                ) == 0
            ):

                progress.progress(

                    min(
                        .92,
                        frame_i
                        /
                        total
                        *
                        .92
                    ),

                    text=(
                        f"Renderizando "
                        f"{int(frame_i / total * 100)}%"
                    )
                )

        proc.stdin.close()
        proc.stdin = None

        stderr = (
            proc.stderr
            .read()
            .decode(
                "utf-8",
                "replace"
            )
        )

        code = proc.wait()

        if code:

            raise RuntimeError(

                "FFmpeg falhou ao "
                "codificar o vídeo.\n"
                +
                stderr[-5000:]
            )

    except BrokenPipeError:

        try:

            proc.stdin.close()

        except Exception:

            pass

        err = (
            proc.stderr
            .read()
            .decode(
                "utf-8",
                "replace"
            )
        )

        proc.wait()

        raise RuntimeError(

            "O FFmpeg encerrou "
            "durante a renderização.\n"
            +
            err[-5000:]
        )

    finally:

        if bgcap:

            bgcap.release()

    # ========================================================
    # ADICIONA ÁUDIO ORIGINAL
    # ========================================================

    run_cmd(

        [

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

            "-shortest",

            "-movflags",
            "+faststart",

            str(out_path)
        ],

        timeout=max(
            180,
            int(
                duration * 8
            )
        )
    )

    silent.unlink(
        missing_ok=True
    )

    if progress:

        progress.progress(
            1.0,
            text="Concluído."
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
        max-width: 1050px;
        padding-top: 1.2rem;
    }

    h1 {
        letter-spacing: -0.04em;
    }

    .small {
        opacity: .72;
        font-size: .88rem;
    }

    div[data-testid="stFileUploader"] {
        border-radius: 14px;
    }

    </style>
    """,

    unsafe_allow_html=True
)


st.title(
    "🎵 Lyric AI Studio"
)

st.caption(
    f"Smooth Kinetic Engine · {APP_VERSION}"
)


with st.expander(
    "O que esta versão faz",
    expanded=False
):

    st.write(

        "Sincroniza a letra palavra por palavra "
        "com o canto, mantém cada frase em uma "
        "composição única e usa entradas, saídas "
        "e transições suaves em preto, branco "
        "e azul royal."
    )


# ============================================================
# UPLOADS
# ============================================================

audio_file = st.file_uploader(

    "1. Envie a música ou vídeo com a música",

    type=[
        "mp3",
        "wav",
        "m4a",
        "mp4",
        "mov",
        "webm"
    ]
)


bg_file = st.file_uploader(

    "2. Fundo opcional (vídeo/imagem)",

    type=[
        "mp4",
        "mov",
        "webm",
        "jpg",
        "jpeg",
        "png"
    ]
)


lyrics = st.text_area(

    "3. Letra (opcional, mas RECOMENDADA)",

    height=120,

    placeholder=(
        "Cole a letra oficial aqui."
    )
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

c1, c2 = st.columns(2)


with c1:

    model = st.selectbox(

        "Qualidade da transcrição",

        [
            "small",
            "medium",
            "large-v3-turbo",
            "large-v3"
        ],

        index=2
    )


with c2:

    theme = st.selectbox(

        "Direção visual",

        [
            "Black & Royal Blue",
            "Black & White"
        ],

        index=0
    )


c3, c4 = st.columns(2)


with c3:

    res_label = st.selectbox(

        "Resolução",

        [
            "1080×1920 — recomendada",
            "720×1280 — econômica"
        ],

        index=0
    )


with c4:

    quality = st.selectbox(

        "Render",

        [
            "Equilibrado",
            "Alta qualidade"
        ],

        index=0
    )


st.info(

    "A letra fornecida é usada como texto oficial. "
    "O Whisper detecta quando cada palavra é cantada."
)


# ============================================================
# FONTES
# ============================================================

registry = load_font_registry()


st.caption(

    f"Fontes disponíveis: "
    f"{len(registry)}/10."
)


# ============================================================
# BOTÃO
# ============================================================

if st.button(

    "🚀 CRIAR LYRIC VIDEO",

    type="primary",

    use_container_width=True
):

    if not audio_file:

        st.error(
            "Envie a música/áudio primeiro."
        )

        st.stop()

    tmpdir = Path(
        tempfile.mkdtemp(
            prefix="lyricai_"
        )
    )

    try:

        # ====================================================
        # ARQUIVOS
        # ====================================================

        audio_path = (
            tmpdir
            /
            safe_name(
                audio_file.name
            )
        )

        audio_path.write_bytes(
            audio_file.getbuffer()
        )

        bg_path = None

        if bg_file:

            bg_path = (
                tmpdir
                /
                safe_name(
                    bg_file.name
                )
            )

            bg_path.write_bytes(
                bg_file.getbuffer()
            )

        # ====================================================
        # STATUS
        # ====================================================

        status = st.empty()

        bar = st.progress(
            0.0
        )

        status.write(
            "Preparando áudio…"
        )

        duration = media_duration(
            str(audio_path)
        )

        if duration <= 0:

            duration = 60

        # ====================================================
        # TRANSCRIÇÃO
        # ====================================================

        try:

            (
                asr_words,
                lang,
                detdur
            ) = transcribe_audio(

                str(audio_path),

                model,

                status
            )

        except Exception:

            if model != "small":

                status.warning(

                    "O modelo escolhido "
                    "falhou ao iniciar. "
                    "Tentando small automaticamente."
                )

                (
                    asr_words,
                    lang,
                    detdur
                ) = transcribe_audio(

                    str(audio_path),

                    "small",

                    status
                )

            else:

                raise

        # ====================================================
        # LETRA MANUAL
        # ====================================================

        if lyrics.strip():

            status.write(

                "Alinhando a letra oficial "
                "aos timestamps reais do cantor…"
            )

            words = align_manual_lyrics(

                lyrics,

                asr_words,

                duration
            )

        else:

            words = asr_words

        words = clean_transcription(
            words
        )

        if not words:

            raise RuntimeError(

                "Nenhuma palavra foi reconhecida. "
                "Cole a letra e tente novamente."
            )

        # ====================================================
        # FRASES
        # ====================================================

        bar.progress(

            .20,

            text="Organizando frases…"
        )

        scenes = segment_lyrics(
            words
        )

        scenes = ensure_coverage(
            scenes,
            duration
        )

        # ====================================================
        # ANÁLISE DE ÁUDIO
        # ====================================================

        global FEATURES_GLOBAL

        FEATURES_GLOBAL = audio_features(

            str(audio_path),

            duration
        )

        bar.progress(

            .30,

            text="Analisando intensidade…"
        )

        # ====================================================
        # RESOLUÇÃO
        # ====================================================

        if res_label.startswith(
            "1080"
        ):

            resolution = (
                1080,
                1920
            )

        else:

            resolution = (
                720,
                1280
            )

        # ====================================================
        # RENDER
        # ====================================================

        out = (
            tmpdir
            /
            "lyric_video_final.mp4"
        )

        render_video(

            str(audio_path),

            str(bg_path)
            if bg_path
            else None,

            scenes,

            registry,

            theme,

            str(out),

            resolution=resolution,

            fps=FPS,

            quality=quality,

            progress=bar
        )

        # ====================================================
        # RESULTADO
        # ====================================================

        status.success(
            "Vídeo criado."
        )

        st.video(
            str(out)
        )

        st.download_button(

            "⬇️ Baixar MP4",

            data=out.read_bytes(),

            file_name=
                "lyric_ai_final.mp4",

            mime="video/mp4",

            use_container_width=True
        )

        # ====================================================
        # DIAGNÓSTICO
        # ====================================================

        with st.expander(
            "Diagnóstico da IA"
        ):

            avg = (

                float(
                    np.mean(
                        [
                            w["prob"]
                            for w in words
                        ]
                    )
                )

                if words

                else 0
            )

            st.write(
                f"Palavras detectadas: "
                f"**{len(words)}**"
            )

            st.write(
                f"Confiança média: "
                f"**{avg:.2f}**"
            )

            st.write(
                f"Idioma: "
                f"**{lang}**"
            )

            st.write(
                f"Cenas: "
                f"**{len([s for s in scenes if not s.get('instrumental')])}**"
            )

            st.code(

                " ".join(
                    w["word"]
                    for w in words
                )
            )

    except Exception as e:

        st.error(
            "A geração falhou."
        )

        st.code(
            str(e)
        )

        st.info(

            "Se houver erro de memória/modelo, "
            "use small. Se houver erro de FFmpeg, "
            "envie o texto do erro para eu corrigir."
        )

    finally:

        pass