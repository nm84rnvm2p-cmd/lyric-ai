import streamlit as st

st.set_page_config(
    page_title="Lyric AI",
    page_icon="🎵"
)

st.title("🎵 Lyric AI")
st.write("Seu gerador automático de lyric videos.")

st.divider()

song = st.file_uploader(
    "🎵 Envie a música",
    type=["mp3", "wav", "m4a", "mp4", "mov"]
)

show = st.file_uploader(
    "🎤 Vídeo do cantor/show (opcional)",
    type=["mp4", "mov", "m4v"]
)

references = st.file_uploader(
    "🎨 Vídeos de referência",
    type=["mp4", "mov", "m4v"],
    accept_multiple_files=True
)

style = st.selectbox(
    "Estilo de edição",
    ["🤖 Automático", "🟢 Clean", "🟡 Dynamic", "🔴 Viral"]
)

if st.button("✨ GERAR LYRIC VIDEO"):
    if song:
        st.success("Arquivos recebidos! O Lyric AI está funcionando.")
        st.write("🎤 Transcrição")
        st.write("📈 Análise da música")
        st.write("📝 Sincronização das frases")
        st.write("🧠 Attention Engine")
        st.write("🎬 Renderização")
    else:
        st.warning("Envie uma música primeiro.")