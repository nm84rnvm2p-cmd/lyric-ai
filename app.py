import tempfile
from pathlib import Path
import numpy as np
import streamlit as st

st.set_page_config(page_title="Lyric AI", page_icon="🎵", layout="centered")

if "result_video" not in st.session_state:
    st.session_state.result_video = None

st.title("🎵 Lyric AI — V1.4")
st.caption("A fonte agora reage à intensidade da música automaticamente.")

song = st.file_uploader("🎵 Música / vídeo", type=["mp3", "wav", "m4a", "mp4", "mov", "webm"])
show = st.file_uploader("🎤 Vídeo do cantor/show (opcional, usado como fundo)", type=["mp4", "mov", "webm", "m4v"])
lyrics = st.text_area("📝 Letra (uma frase/linha por linha)", placeholder="Cole aqui a letra da música...\nUma linha por frase")
style = st.selectbox("🎨 Estilo", ["🤖 Automático", "🟢 Clean", "🟡 Dynamic", "🔴 Viral"])
auto_intensity = st.checkbox("🎚️ Fonte reage à intensidade do áudio", value=True)


def find_font(bold: bool):
    """Fonte com suporte a acentos do português.
    1º: fonte própria do projeto (fonts/), garante resultado igual em qualquer servidor.
    2º: fallback pra fonte embutida no pacote matplotlib (sempre disponível se matplotlib
        estiver instalado — não precisa configurar nada no servidor).
    """
    name = "OpenSans-Bold.ttf" if bold else "OpenSans-Regular.ttf"
    project_font = Path(__file__).parent / "fonts" / name
    if project_font.exists():
        return str(project_font)
    try:
        import matplotlib
        mpl_name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        mpl_font = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / mpl_name
        if mpl_font.exists():
            return str(mpl_font)
    except Exception:
        pass
    return None


def compute_intensity(audio_mono, sr, start, dur):
    """RMS (energia) do áudio no trecho — retorna um valor bruto, ainda não normalizado."""
    i0 = max(0, int(start * sr))
    i1 = min(len(audio_mono), int((start + dur) * sr))
    chunk = audio_mono[i0:i1]
    if chunk.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(chunk ** 2)))


if st.button("✨ GERAR VÍDEO", type="primary", use_container_width=True):
    if not song:
        st.error("Envie a música/vídeo primeiro.")
        st.stop()
    if not lyrics.strip():
        st.error("Cole a letra para este primeiro teste.")
        st.stop()

    with st.status("🎬 Gerando seu lyric video...", expanded=True) as status:
        try:
            from moviepy import VideoFileClip, AudioFileClip, ColorClip, TextClip, CompositeVideoClip
            from moviepy import vfx

            with tempfile.TemporaryDirectory() as td:
                td = Path(td)
                source = td / f"source{Path(song.name).suffix}"
                source.write_bytes(song.getbuffer())
                audio_path = source

                if source.suffix.lower() in {".mp4", ".mov", ".webm", ".m4v"}:
                    vc = VideoFileClip(str(source))
                    if vc.audio is None:
                        vc.close()
                        raise RuntimeError("O vídeo enviado não possui áudio.")
                    duration = float(vc.duration)
                    audio_path = td / "audio.wav"
                    vc.audio.write_audiofile(str(audio_path), logger=None)
                    vc.close()
                else:
                    ac = AudioFileClip(str(source))
                    duration = float(ac.duration)
                    ac.close()

                # Divide a letra por LINHA (respeita o que a pessoa colou)
                phrases = [ln.strip() for ln in lyrics.splitlines() if ln.strip()]
                if not phrases:
                    phrases = [lyrics.strip()]

                total = sum(max(1, len(p)) for p in phrases)
                cursor, segments = 0.0, []
                for p in phrases:
                    dur = min(max(1.0, duration * len(p) / total), duration - cursor)
                    if dur <= 0:
                        break
                    segments.append((p, cursor, dur))
                    cursor += dur

                # --- Análise de intensidade do áudio (RMS por trecho) ---
                intensities = [0.5] * len(segments)  # neutro, caso a análise falhe
                if auto_intensity:
                    try:
                        sr = 11025  # taxa reduzida: suficiente pra medir energia, leve em memória
                        probe = AudioFileClip(str(audio_path))
                        arr = probe.to_soundarray(fps=sr)
                        probe.close()
                        mono = arr.mean(axis=1) if arr.ndim > 1 else arr

                        raw = [compute_intensity(mono, sr, s, d) for _, s, d in segments]
                        lo, hi = min(raw), max(raw)
                        if hi - lo > 1e-6:
                            intensities = [(v - lo) / (hi - lo) for v in raw]
                    except Exception:
                        pass  # se a análise falhar, segue com intensidade neutra

                W, H = 1080, 1920
                layers = []

                if show:
                    sp = td / f"show{Path(show.name).suffix}"
                    sp.write_bytes(show.getbuffer())
                    bg = VideoFileClip(str(sp)).without_audio().resized(height=H)
                    if bg.w < W:
                        bg = bg.resized(width=W)
                    bg = bg.cropped(x_center=bg.w / 2, y_center=bg.h / 2, width=W, height=H)
                    bg = bg.subclipped(0, min(duration, bg.duration))
                    layers.append(bg)
                else:
                    layers.append(ColorClip(size=(W, H), color=(0, 0, 0)).with_duration(duration))

                for i, (p, start, dur) in enumerate(segments):
                    intensity = intensities[i]  # 0 (suave) a 1 (forte)
                    white = i % 4 == 2

                    if white:
                        layers.append(
                            ColorClip(size=(W, H), color=(245, 245, 245))
                            .with_start(start)
                            .with_duration(dur)
                        )

                    # Tamanho base pelo comprimento do texto (evita estourar a caixa)
                    base_size = 76 if len(p) < 35 else 60
                    # Ajuste sutil pela intensidade: no máximo +/- ~18px, pra não "gritar"
                    size_boost = int((intensity - 0.5) * 36)
                    font_size = max(44, base_size + size_boost)

                    is_bold = intensity >= 0.55  # trechos mais fortes puxam pro negrito
                    font_path = find_font(bold=is_bold)

                    text_kwargs = dict(
                        text=p,
                        font_size=font_size,
                        color="black" if white else "white",
                        method="caption",
                        size=(900, 520),
                        text_align="center",
                    )
                    if font_path:
                        text_kwargs["font"] = font_path

                    # Contorno bem discreto nos trechos mais intensos, pra dar destaque sem poluir
                    if intensity > 0.7 and not white:
                        text_kwargs["stroke_color"] = "black"
                        text_kwargs["stroke_width"] = 2

                    text = TextClip(**text_kwargs)
                    text = text.with_position("center").with_start(start).with_duration(dur)

                    fade = min(0.28, dur / 4)
                    text = text.with_effects([vfx.CrossFadeIn(fade), vfx.CrossFadeOut(fade)])
                    layers.append(text)

                audio = AudioFileClip(str(audio_path))
                final = CompositeVideoClip(layers, size=(W, H)).with_audio(audio)
                output = td / "lyric_ai_result.mp4"

                final.write_videofile(
                    str(output), fps=30, codec="libx264",
                    audio_codec="aac", preset="ultrafast", threads=2, logger=None,
                )

                st.session_state.result_video = output.read_bytes()
                final.close()
                audio.close()

            status.update(label="✅ Vídeo pronto!", state="complete", expanded=False)

        except Exception as exc:
            status.update(label="❌ Erro na renderização", state="error", expanded=True)
            st.error(str(exc))
            st.session_state.result_video = None

if st.session_state.result_video:
    st.divider()
    st.header("🎬 Seu vídeo está pronto")
    st.video(st.session_state.result_video, format="video/mp4")
    st.download_button(
        "⬇️ SALVAR VÍDEO MP4", data=st.session_state.result_video,
        file_name="lyric_ai_result.mp4", mime="video/mp4",
        use_container_width=True,
    )
    if st.button("🗑️ Limpar resultado"):
        st.session_state.result_video = None
        st.rerun()