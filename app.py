import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Lyric AI", page_icon="🎵", layout="centered")
st.title("🎵 Lyric AI — V1.1")
st.caption("Correção do renderizador + fade compatível com MoviePy 2.")

song = st.file_uploader(
    "🎵 Música / vídeo da música",
    type=["mp3", "wav", "m4a", "mp4", "mov", "webm"],
)

show = st.file_uploader(
    "🎤 Vídeo do cantor/show (opcional)",
    type=["mp4", "mov", "webm", "m4v"],
)

lyrics = st.text_area(
    "📝 Letra",
    placeholder="Cole aqui a letra da música...",
)

style = st.selectbox(
    "🎨 Estilo",
    ["🤖 Automático", "🟢 Clean", "🟡 Dynamic", "🔴 Viral"],
)

if st.button("✨ GERAR VÍDEO", type="primary", use_container_width=True):
    if not song:
        st.error("Envie a música/vídeo primeiro.")
        st.stop()

    if not lyrics.strip():
        st.error("Cole a letra para este primeiro teste.")
        st.stop()

    with st.spinner("Renderizando o vídeo..."):
        try:
            from moviepy import (
                VideoFileClip,
                AudioFileClip,
                ColorClip,
                TextClip,
                CompositeVideoClip,
            )
            from moviepy import vfx
            import numpy as np

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)

                source = temp_dir / f"source{Path(song.name).suffix}"
                source.write_bytes(song.getbuffer())

                audio_path = source
                source_ext = source.suffix.lower()

                if source_ext in {".mp4", ".mov", ".webm", ".m4v"}:
                    source_clip = VideoFileClip(str(source))
                    duration = float(source_clip.duration)

                    if source_clip.audio is None:
                        source_clip.close()
                        raise RuntimeError("O vídeo enviado não possui faixa de áudio.")

                    audio_path = temp_dir / "audio.wav"
                    source_clip.audio.write_audiofile(
                        str(audio_path),
                        logger=None,
                    )
                    source_clip.close()
                else:
                    audio = AudioFileClip(str(source))
                    duration = float(audio.duration)
                    audio.close()

                # Nesta V1, a divisão é aproximada por tamanho das frases.
                # A sincronização palavra-a-palavra será adicionada na próxima versão.
                phrases = [
                    p.strip()
                    for p in lyrics.replace("\n", " ").split(".")
                    if p.strip()
                ]

                if not phrases:
                    phrases = [lyrics.strip()]

                total_chars = sum(max(1, len(p)) for p in phrases)
                cursor = 0.0
                segments = []

                for phrase in phrases:
                    seg_duration = max(
                        1.0,
                        duration * len(phrase) / total_chars,
                    )
                    seg_duration = min(
                        seg_duration,
                        duration - cursor,
                    )

                    if seg_duration <= 0:
                        break

                    segments.append(
                        (phrase, cursor, seg_duration)
                    )
                    cursor += seg_duration

                W, H = 1080, 1920
                layers = []

                # Fundo: vídeo do show, quando fornecido.
                if show:
                    show_path = temp_dir / f"show{Path(show.name).suffix}"
                    show_path.write_bytes(show.getbuffer())

                    bg = VideoFileClip(
                        str(show_path)
                    ).without_audio()

                    bg = bg.resized(height=H)

                    if bg.w < W:
                        bg = bg.resized(width=W)

                    bg = bg.cropped(
                        x_center=bg.w / 2,
                        y_center=bg.h / 2,
                        width=W,
                        height=H,
                    )

                    bg_duration = min(duration, bg.duration)
                    bg = bg.subclipped(0, bg_duration)

                    # Preto e branco.
                    bg = bg.image_transform(
                        lambda frame: np.mean(
                            frame,
                            axis=2,
                            keepdims=True,
                        ).repeat(3, axis=2).astype("uint8")
                    )

                    # Se o vídeo for menor que a música, ele fica apenas
                    # pelo tempo disponível.
                    layers.append(bg)

                else:
                    layers.append(
                        ColorClip(
                            size=(W, H),
                            color=(0, 0, 0),
                        ).with_duration(duration)
                    )

                for index, (phrase, start, seg_duration) in enumerate(segments):

                    # Alternância visual discreta.
                    white_card = index % 4 == 2

                    if white_card:
                        card = (
                            ColorClip(
                                size=(W, H),
                                color=(245, 245, 245),
                            )
                            .with_start(start)
                            .with_duration(seg_duration)
                        )
                        layers.append(card)

                    text = TextClip(
                        text=phrase,
                        font_size=76 if len(phrase) < 35 else 60,
                        color="black" if white_card else "white",
                        method="caption",
                        size=(900, 520),
                        text_align="center",
                    ).with_start(start).with_duration(seg_duration)

                    # MoviePy 2 usa Effects, não .crossfadein/.crossfadeout.
                    fade = min(0.28, seg_duration / 4)

                    text = text.with_effects(
                        [
                            vfx.CrossFadeIn(fade),
                            vfx.CrossFadeOut(fade),
                        ]
                    )

                    layers.append(text)

                audio = AudioFileClip(str(audio_path))

                final = CompositeVideoClip(
                    layers,
                    size=(W, H),
                ).with_audio(audio)

                output = temp_dir / "lyric_ai_v1_1.mp4"

                final.write_videofile(
                    str(output),
                    fps=30,
                    codec="libx264",
                    audio_codec="aac",
                    preset="ultrafast",
                    threads=2,
                    logger=None,
                )

                data = output.read_bytes()

                final.close()
                audio.close()

                st.success("🎬 Vídeo pronto!")
                st.video(data)

                st.download_button(
                    "⬇️ SALVAR MP4",
                    data=data,
                    file_name="lyric_ai_v1_1.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )

        except Exception as exc:
            st.error("A renderização falhou.")
            st.code(str(exc))
            st.info(
                "Se aparecer outro erro, me envie exatamente o texto mostrado "
                "aqui. Vamos corrigir o próximo ponto sem você precisar refazer o projeto."
            )