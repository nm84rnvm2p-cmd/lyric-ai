from __future__ import annotations
import io, re, tempfile
from pathlib import Path
import numpy as np
import streamlit as st
from style_engine import StyleDNA, analyze_references

st.set_page_config(page_title="Lyric AI — Definitive", page_icon="🎵", layout="centered")
if "result_video" not in st.session_state: st.session_state.result_video=None
if "style_dna" not in st.session_state: st.session_state.style_dna=StyleDNA()

def save_upload(x, d, name=None):
    suffix=Path(x.name).suffix.lower() or ".mp4"
    p=d/(name or ("upload"+suffix)); p.write_bytes(x.getbuffer()); return p

def transcribe(path, model_size):
    from faster_whisper import WhisperModel
    @st.cache_resource(show_spinner=False)
    def load(size):
        return WhisperModel(size, device="cpu", compute_type="int8")
    model=load(model_size)
    segs, info=model.transcribe(path, language="pt", beam_size=5,
        word_timestamps=True, vad_filter=True,
        vad_parameters={"min_silence_duration_ms":450},
        condition_on_previous_text=False)
    words=[]
    for s in segs:
        if s.words:
            for w in s.words:
                if w.start is not None and w.end is not None and w.word.strip():
                    words.append({"text":w.word.strip(),"start":float(w.start),"end":float(w.end)})
    if not words: raise RuntimeError("Nenhuma palavra foi detectada.")
    return words, info.language

def audio_analysis(path):
    import librosa
    y,sr=librosa.load(path,sr=22050,mono=True)
    if len(y)==0: raise RuntimeError("Áudio vazio ou ilegível.")
    onset=librosa.onset.onset_strength(y=y,sr=sr)
    tempo,frames=librosa.beat.beat_track(onset_envelope=onset,sr=sr,units="frames")
    beats=librosa.frames_to_time(frames,sr=sr)
    rms=librosa.feature.rms(y=y,frame_length=2048,hop_length=512)[0]
    times=librosa.frames_to_time(np.arange(len(rms)),sr=sr,hop_length=512)
    lo,hi=np.percentile(rms,[5,95])
    energy=np.clip((rms-lo)/max(hi-lo,1e-8),0,1)
    return {"duration":float(len(y)/sr),"tempo":float(np.asarray(tempo).reshape(-1)[0]),
            "beats":beats,"energy":energy,"times":times}

def energy_at(a,t):
    if len(a["times"])==0:return .5
    i=int(np.clip(np.searchsorted(a["times"],t),0,len(a["energy"])-1))
    return float(a["energy"][i])

def beat_strength(a,start,end):
    b=a["beats"]
    return float(np.clip(np.mean((b>=start)&(b<=end))*2,0,1)) if len(b) else 0

def group_words(words):
    groups=[]; cur=[]
    for w in words:
        if not cur: cur=[w]; continue
        prev=cur[-1]; gap=w["start"]-prev["end"]
        candidate=" ".join(x["text"] for x in cur+[w])
        if gap>=.38 or re.search(r"[.!?,;:]$",prev["text"]) or len(cur)>=7 or len(candidate)>34 or w["end"]-cur[0]["start"]>2.8:
            groups.append(cur); cur=[w]
        else: cur.append(w)
    if cur: groups.append(cur)
    return [{"text":" ".join(x["text"] for x in g),"start":g[0]["start"],"end":g[-1]["end"]} for g in groups]

def score_phrases(ps,a,dna):
    for p in ps:
        mid=(p["start"]+p["end"])/2
        e=energy_at(a,mid); b=beat_strength(a,p["start"],p["end"])
        dur=max(.1,p["end"]-p["start"])
        short=np.clip(1-dur/2.5,0,1)
        p["impact"]=float(np.clip(.45*e+.20*b+.20*short+.08*dna.motion+.07*dna.cut_rate,0,1))
    return ps

def text_image(text,font_path,size,color):
    from PIL import Image,ImageDraw,ImageFont
    font=ImageFont.truetype(font_path,size)
    W,H=930,540; im=Image.new("RGBA",(W,H),(0,0,0,0)); d=ImageDraw.Draw(im)
    words=text.split(); lines=[]; cur=""
    for w in words:
        test=w if not cur else cur+" "+w
        if d.textbbox((0,0),test,font=font)[2]<=W-80: cur=test
        else:
            if cur: lines.append(cur)
            cur=w
    if cur: lines.append(cur)
    gap=14; boxes=[d.textbbox((0,0),x,font=font) for x in lines]
    hs=[b[3]-b[1] for b in boxes]; total=sum(hs)+gap*max(0,len(hs)-1); y=max(0,(H-total)/2)
    for line,box,h in zip(lines,boxes,hs):
        x=(W-(box[2]-box[0]))/2
        d.text((x+3,y+3),line,font=font,fill=(0,0,0,150)); d.text((x,y),line,font=font,fill=color); y+=h+gap
    return np.asarray(im)

def font_path(uploaded,d):
    if uploaded: return str(save_upload(uploaded,d,"font.ttf"))
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"]:
        if Path(p).exists(): return p
    raise RuntimeError("Fonte Unicode não encontrada. Envie uma fonte .ttf/.otf.")

def render(music,show,phrases,a,dna,font,out):
    from moviepy import AudioFileClip,ColorClip,CompositeVideoClip,ImageClip,VideoFileClip
    from moviepy import vfx
    W,H=1080,1920; audio=AudioFileClip(music); duration=min(float(audio.duration),a["duration"])
    show_clip=VideoFileClip(show).without_audio() if show else None
    layers=[ColorClip((W,H),color=(0,0,0)).with_duration(duration)]
    try:
        for i,p in enumerate(phrases):
            start=max(0,p["start"]); end=min(duration,p["end"]); length=end-start
            if length<=0: continue
            impact=p["impact"]; live=show_clip is not None and (impact>=.68 or (i%5==3 and dna.live_probability>=.45))
            if live:
                bg=show_clip.subclipped(start%show_clip.duration,min((start%show_clip.duration)+length,show_clip.duration)).resized(height=H)
                if bg.w<W:bg=bg.resized(width=W)
                bg=bg.cropped(x_center=bg.w/2,y_center=bg.h/2,width=W,height=H).with_start(start)
                layers.append(bg)
            light=i%4==2 and impact<.72
            if light: layers.append(ColorClip((W,H),color=(245,245,245)).with_start(start).with_duration(length))
            fs=max(52,int(66+22*impact-(8 if len(p["text"])>38 else 0)))
            img=text_image(p["text"],font,fs,(15,15,15) if light else (255,255,255))
            t=ImageClip(img,transparent=True).with_start(start).with_duration(length)
            fade=min(.20-impact*.10,length/3)
            t=t.with_effects([vfx.CrossFadeIn(max(.06,fade)),vfx.CrossFadeOut(max(.06,fade))])
            layers.append(t)
        final=CompositeVideoClip(layers,size=(W,H)).with_audio(audio)
        final.write_videofile(out,fps=30,codec="libx264",audio_codec="aac",preset="ultrafast",threads=2,logger=None)
        final.close()
    finally:
        if show_clip: show_clip.close()
        audio.close()

st.title("🎵 Lyric AI — Definitive Engine")
st.caption("Timestamps reais + energia/batida + Style DNA + cortes seletivos de show + 9:16.")

music=st.file_uploader("🎵 Música / vídeo com áudio",type=["mp3","wav","m4a","mp4","mov","webm"])
show=st.file_uploader("🎤 Vídeo do cantor/show (opcional)",type=["mp4","mov","webm","m4v"])
refs=st.file_uploader("🎞️ 3–5 vídeos-base (recomendado)",type=["mp4","mov","webm","m4v"],accept_multiple_files=True)
font=st.file_uploader("🔤 Fonte .ttf/.otf (opcional — envie Larken se tiver)",type=["ttf","otf"])
lyrics=st.text_area("📝 Letra (opcional)",height=140,placeholder="Se deixar vazio, a IA transcreve.")
model=st.selectbox("🧠 Transcrição",["base","small","tiny"],index=0)

if refs:
    with st.spinner("Analisando os vídeos-base..."):
        st.session_state.style_dna=analyze_references(refs)
dna=st.session_state.style_dna
with st.expander("🧬 Style DNA",expanded=False):
    st.json(dna.to_dict())

if st.button("🚀 CRIAR LYRIC VIDEO",type="primary",use_container_width=True):
    if not music: st.error("Envie a música primeiro."); st.stop()
    with st.status("🎬 Construindo...",expanded=True) as status:
        try:
            from moviepy import VideoFileClip
            with tempfile.TemporaryDirectory() as td:
                td=Path(td); mp=save_upload(music,td,"music"+Path(music.name).suffix)
                sp=save_upload(show,td,"show"+Path(show.name).suffix) if show else None
                fp=font_path(font,td)
                ap=mp
                if mp.suffix.lower() in {".mp4",".mov",".webm",".m4v"}:
                    vc=VideoFileClip(str(mp))
                    if vc.audio is None: vc.close(); raise RuntimeError("O vídeo não possui áudio.")
                    ap=td/"audio.wav"; vc.audio.write_audiofile(str(ap),logger=None); vc.close()
                status.write("🎧 Analisando batidas e energia...")
                aa=audio_analysis(str(ap))
                status.write(f"✓ BPM estimado: {aa['tempo']:.1f} | {aa['duration']:.1f}s")
                status.write("🗣️ Transcrevendo com timestamps de palavras...")
                words,lang=transcribe(str(ap),model)
                status.write(f"✓ {len(words)} palavras | idioma detectado: {lang}")
                phrases=score_phrases(group_words(words),aa,dna)
                if lyrics.strip(): status.write("✓ Letra fornecida usada como referência; timing vem da voz.")
                out=td/"lyric_ai_definitive.mp4"
                status.write("🎬 Renderizando...")
                render(str(ap),str(sp) if sp else None,phrases,aa,dna,fp,str(out))
                st.session_state.result_video=out.read_bytes()
            status.update(label="✅ Vídeo pronto!",state="complete",expanded=False)
        except Exception as e:
            status.update(label="❌ Geração falhou",state="error",expanded=True)
            st.error(str(e)); st.exception(e); st.session_state.result_video=None

if st.session_state.result_video:
    st.divider(); st.header("🎬 Resultado")
    st.video(st.session_state.result_video)
    st.download_button("⬇️ SALVAR MP4",st.session_state.result_video,"lyric_ai_definitive.mp4","video/mp4",use_container_width=True)