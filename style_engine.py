from dataclasses import dataclass, asdict
from pathlib import Path
import tempfile

import numpy as np


@dataclass
class StyleDNA:

    brightness: float = 0.20

    contrast: float = 0.55

    cut_rate: float = 0.35

    motion: float = 0.40

    text_density: float = 0.45

    live_probability: float = 0.35

    def to_dict(self):

        return asdict(self)


def analyze_reference_video(
    uploaded_file,
    max_seconds=60,
    sample_fps=2,
):

    import cv2

    suffix = (
        Path(uploaded_file.name).suffix
        or ".mp4"
    )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as file:

        file.write(
            uploaded_file.getbuffer()
        )

        path = file.name

    capture = cv2.VideoCapture(
        path
    )

    fps = (
        capture.get(
            cv2.CAP_PROP_FPS
        )
        or 30.0
    )

    frame_count = (
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
        or 0
    )

    duration = (
        frame_count / fps
        if fps
        else 0
    )

    step = max(
        1,
        int(fps / sample_fps),
    )

    limit = int(
        min(
            duration,
            max_seconds,
        )
        * fps
    )

    brightness = []
    contrast = []
    differences = []

    previous = None

    index = 0

    while index < limit:

        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            index,
        )

        ok, frame = (
            capture.read()
        )

        if not ok:

            break

        small = cv2.resize(
            frame,
            (160, 90),
        )

        gray = cv2.cvtColor(
            small,
            cv2.COLOR_BGR2GRAY,
        ).astype(
            np.float32
        )

        brightness.append(
            float(
                gray.mean()
                / 255
            )
        )

        contrast.append(
            float(
                gray.std()
                / 255
            )
        )

        if previous is not None:

            differences.append(
                float(
                    np.mean(
                        np.abs(
                            gray
                            - previous
                        )
                    )
                    / 255
                )
            )

        previous = gray

        index += step

    capture.release()

    try:

        Path(path).unlink()

    except Exception:

        pass

    if not brightness:

        return StyleDNA()

    motion = float(
        np.clip(
            np.mean(
                differences
            )
            * 5,
            0,
            1,
        )
    ) if differences else 0.4

    cut_rate = float(
        np.clip(
            np.mean(
                np.array(
                    differences
                )
                > 0.18
            )
            * 3,
            0,
            1,
        )
    ) if differences else 0.35

    brightness_average = float(
        np.mean(brightness)
    )

    contrast_average = float(
        np.mean(contrast)
    )

    return StyleDNA(

        brightness=brightness_average,

        contrast=contrast_average,

        cut_rate=cut_rate,

        motion=motion,

        text_density=float(
            np.clip(
                0.70
                - brightness_average
                + contrast_average * 0.35,
                0.15,
                0.90,
            )
        ),

        live_probability=float(
            np.clip(
                0.25
                + motion * 0.55,
                0,
                1,
            )
        ),
    )


def analyze_references(files):

    if not files:

        return StyleDNA()

    profiles = []

    for file in files:

        try:

            profiles.append(
                analyze_reference_video(
                    file
                )
            )

        except Exception:

            pass

    if not profiles:

        return StyleDNA()

    fields = (
        StyleDNA
        .__dataclass_fields__
        .keys()
    )

    return StyleDNA(
        **{
            field: float(
                np.mean(
                    [
                        getattr(
                            profile,
                            field,
                        )
                        for profile in profiles
                    ]
                )
            )
            for field in fields
        }
    )
    