import tempfile
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Lyric AI", page_icon="🎵", layout="centered")

if "result_video" not in st.session_state:
    st.session_state.result_video = None

st.title("🎵 Lyric AI — V1.2")
st.caption("Renderização persistente: o vídeo fica disponível depois de gerar.")

song = st.file_uploader("🎵 Música / vídeo", type=["mp3","wav","m4a","mp4","mov","webm"])
show = st.file_uploader("🎤 Vídeo do cantor/show (opcional)", type=["mp4","mov","webm","m4v"])
lyrics = st.text_area("📝 Letra", placeholder="Cole aqui a letra da música...")
style = st.selectbox("🎨 Estilo", ["🤖 Automático","🟢 Clean","🟡 Dynamic","🔴 Viral"])

if st.button("✨ GERAR VÍDEO", type="primary", use_container_width=True):
    if not song:
        st.error("Envie a música/vídeo primeiro."); st.stop()
    if not lyrics.strip():
        st.error("Cole a letra para este primeiro teste."); st.stop()

    with st.status("🎬 Gerando seu lyric video...", expanded=True) as status:
        try:
            from moviepy import VideoFileClip, AudioFileClip, ColorClip, TextClip, CompositeVideoClip
            from moviepy import vfx
            import numpy as np

            with tempfile.TemporaryDirectory() as td:
                td = Path(td)
                source = td / f"source{Path(song.name).suffix}"
                source.write_bytes(song.getbuffer())
                audio_path = source

                if source.suffix.lower() in {".mp4",".mov",".webm",".m4v"}:
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

                phrases = [p.strip() for p in lyrics.replace("\n"," ").split(".") if p.strip()]
                if not phrases: phrases = [lyrics.strip()]
                total = sum(max(1,len(p)) for p in phrases)
                cursor, segments = 0.0, []
                for p in phrases:
                    dur = min(max(1.0, duration*len(p)/total), duration-cursor)
                    if dur <= 0: break
                    segments.append((p,cursor,dur)); cursor += dur

                W,H = 1080,1920
                layers = []

                if show:
                    sp = td / f"show{Path(show.name).suffix}"
                    sp.write_bytes(show.getbuffer())
                    bg = VideoFileClip(str(sp)).without_audio().resized(height=H)
                    if bg.w < W: bg = bg.resized(width=W)
                    bg = bg.cropped(x_center=bg.w/2,y_center=bg.h/2,width=W,height=H)
                    bg = bg.subclipped(0,min(duration,bg.duration))
                    layers.append(bg)
                else:
                    layers.append(ColorClip(size=(W,H),color=(0,0,0)).with_duration(duration))

                for i,(p,start,dur) in enumerate(segments):
                    white = i % 4 == 2
                    if white:
                        layers.append(ColorClip(size=(W,H),color=(245,245,245)).with_start(start).with_duration(dur))
                    text = TextClip(text=p,font_size=76 if len(p)<35 else 60,
                                    color="black" if white else "white",method="caption",
                                    size=(900,520),text_align="center").with_start(start).with_duration(dur)
                    fade=min(.28,dur/4)
                    text=text.with_effects([vfx.CrossFadeIn(fade),vfx.CrossFadeOut(fade)])
                    layers.append(text)

                audio=AudioFileClip(str(audio_path))
                final=CompositeVideoClip(layers,size=(W,H)).with_audio(audio)
                output=td/"lyric_ai_result.mp4"

                final.write_videofile(str(output),fps=30,codec="libx264",
                                      audio_codec="aac",preset="ultrafast",threads=2,logger=None)

                st.session_state.result_video=output.read_bytes()
                final.close()
                audio.close()

            status.update(label="✅ Vídeo pronto!", state="complete", expanded=False)

        except Exception as exc:
            status.update(label="❌ Erro na renderização", state="error", expanded=True)
            st.error(str(exc))
            st.session_state.result_video=None

if st.session_state.result_video:
    st.divider()
    st.header("🎬 Seu vídeo está pronto")
    st.video(st.session_state.result_video, format="video/mp4")
    st.download_button("⬇️ SALVAR VÍDEO MP4", data=st.session_state.result_video,
                       file_name="lyric_ai_result.mp4", mime="video/mp4",
                       use_container_width=True)
    if st.button("🗑️ Limpar resultado"):
        st.session_state.result_video=None
        st.rerun()