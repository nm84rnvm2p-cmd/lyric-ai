import os, re, math, shutil, subprocess, tempfile, difflib, unicodedata
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

APP_VERSION = "14.0-LITE"
BLACK=(5,5,7); WHITE=(248,248,246); ROYAL=(45,92,255)
FPS=24

# Only local fonts: no font downloads during Streamlit startup.
FONT_PATHS={
    "Grossa": "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
    "Limpa": "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    "Serif": "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "Safe": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
}


def ffmpeg():
    try:
        import imageio_ffmpeg
        p=imageio_ffmpeg.get_ffmpeg_exe()
        if p and os.path.exists(p): return p
    except Exception: pass
    p=shutil.which("ffmpeg")
    if p: return p
    raise RuntimeError("FFmpeg não encontrado. Verifique imageio-ffmpeg no requirements.txt.")


def cmdrun(cmd, timeout=None):
    p=subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    if p.returncode: raise RuntimeError(p.stderr[-4000:])
    return p.stdout, p.stderr


def duration(path):
    _,err=cmdrun([ffmpeg(),"-hide_banner","-i",path,"-f","null","-"],timeout=90)
    m=re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)",err)
    return int(m.group(1))*3600+int(m.group(2))*60+float(m.group(3)) if m else 0.0


def norm(s):
    s=unicodedata.normalize("NFC",s or "")
    s=s.replace("\ufeff","").replace("\u200b","")
    s="".join(c for c in s if c=="\n" or unicodedata.category(c)[0]!="C")
    return re.sub(r"\s+"," ",s).strip()


def key(s):
    s=unicodedata.normalize("NFKD",s or "")
    s="".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-zA-Z0-9]","",s).lower()


def sim(a,b):
    a,b=key(a),key(b)
    if not a or not b:return 0
    if a==b:return 1
    if a in b or b in a:return .9
    return difflib.SequenceMatcher(None,a,b,autojunk=False).ratio()


def parse_time(s):
    s=s.strip().replace(",",".")
    if ":" in s:
        a,b=s.split(":",1); return int(a)*60+float(b)
    return float(s)


def parse_lyrics(text):
    """Supports both 'time - time' on its own line + lyric next line, and inline form."""
    lines=[x.strip() for x in (text or "").replace("\r","").split("\n")]
    out=[]; i=0
    pat=re.compile(r"^(\d{1,2}:\d{2}(?:[.,]\d{1,3})?|\d+(?:[.,]\d+)?)\s*[-–—]\s*(\d{1,2}:\d{2}(?:[.,]\d{1,3})?|\d+(?:[.,]\d+)?)(?:\s*\|\s*(.*))?$")
    while i<len(lines):
        if not lines[i]: i+=1; continue
        m=pat.match(lines[i])
        if m:
            st,en=parse_time(m.group(1)),parse_time(m.group(2)); lyric=norm(m.group(3) or "")
            if not lyric and i+1<len(lines):
                j=i+1
                while j<len(lines) and not lines[j]: j+=1
                if j<len(lines) and not pat.match(lines[j]): lyric=norm(lines[j]); i=j
            if lyric and en>st: out.append({"start":st,"end":en,"text":lyric})
        i+=1
    return sorted(out,key=lambda x:x["start"])


def plain_lines(text): return [norm(x) for x in (text or "").splitlines() if norm(x)]


@st.cache_resource(show_spinner=False)
def get_model(name):
    from faster_whisper import WhisperModel
    return WhisperModel(name,device="cpu",compute_type="int8",cpu_threads=2,num_workers=1)


def transcribe(path,name,status):
    status.write(f"🎙️ Reconhecendo com **{name}**…")
    model=get_model(name)
    segs,info=model.transcribe(path,language="pt",word_timestamps=True,beam_size=1,best_of=1,patience=1.0,temperature=0.0,condition_on_previous_text=False,vad_filter=False)
    words=[]
    for seg in segs:
        for w in (seg.words or []):
            t=norm(w.word)
            if t and len(t)<=40:
                words.append({"word":t,"start":float(w.start),"end":float(w.end),"prob":float(getattr(w,"probability",0) or 0)})
    words.sort(key=lambda x:x["start"])
    return words,getattr(info,"language","pt")


def align_phrase(text,st,en,asr):
    toks=re.findall(r"\S+",text); cand=[w for w in asr if w["end"]>=st-.4 and w["start"]<=en+.4]
    used=-1; mapped=[]
    for tok in toks:
        best=None; bs=0
        for j in range(used+1,min(len(cand),used+31)):
            s=sim(tok,cand[j]["word"])
            if st<=cand[j]["start"]<=en:s+=.05
            if s>bs: bs=s; best=j
        if best is not None and bs>=.38: used=best; mapped.append((tok,cand[best]))
        else: mapped.append((tok,None))
    known=[i for i,x in enumerate(mapped) if x[1] is not None]
    result=[]
    for i,(tok,w) in enumerate(mapped):
        if w:
            a=max(st,w["start"]); b=min(en,max(a+.06,w["end"]))
        else:
            p=max([k for k in known if k<i],default=-1); n=min([k for k in known if k>i],default=len(mapped))
            a=result[p]["end"] if p>=0 else st
            b=mapped[n][1]["start"] if n<len(mapped) else en
            if b<a:b=a+.08
            a=a+(b-a)*(i-p)/max(1,n-p); b=a+(b-a)/max(1,n-p)
            a=max(st,min(a,en-.06)); b=min(en,max(b,a+.06))
        result.append({"word":tok,"start":a,"end":b})
    return result


def build_scenes(lyrics,asr,dur):
    timed=parse_lyrics(lyrics)
    scenes=[]
    if timed:
        for i,l in enumerate(timed):
            st=max(0,min(dur,l["start"])); en=min(dur,l["end"])
            if i+1<len(timed): en=min(en,timed[i+1]["start"])
            if en<=st:continue
            ws=align_phrase(l["text"],st,en,asr)
            if ws: scenes.append({"start":st,"end":en,"words":ws,"text":l["text"]})
        return scenes
    lines=plain_lines(lyrics)
    cursor=0
    for line in lines:
        toks=re.findall(r"\S+",line); matches=[]
        for tok in toks:
            best=None;bs=0
            for j in range(cursor,min(len(asr),cursor+60)):
                s=sim(tok,asr[j]["word"])
                if s>bs:bs=s;best=j
                if s>=.99:break
            if best is not None and bs>=.38: matches.append(best);cursor=best+1
        if matches:
            st=asr[matches[0]]["start"]; en=asr[matches[-1]]["end"]
        elif cursor<len(asr):
            st=asr[cursor]["start"]; en=min(dur,st+max(.8,.28*len(toks))); cursor+=1
        else: break
        en=min(dur,max(en,st+.4))
        ws=align_phrase(line,st,en,asr)
        if ws: scenes.append({"start":st,"end":en,"words":ws,"text":line})
    return scenes


def auto_scenes(asr,dur):
    groups=[];cur=[]
    for w in asr:
        if cur and w["start"]-cur[-1]["end"]>.65:groups.append(cur);cur=[]
        cur.append(w)
    if cur:groups.append(cur)
    return [{"start":max(0,g[0]["start"]-.02),"end":min(dur,g[-1]["end"]+.18),"words":g,"text":" ".join(w["word"] for w in g)} for g in groups]


def font(path,size): return ImageFont.truetype(path,max(18,int(size)))

def render_image(scene,idx,t,W,H):
    bg=BLACK if idx%2==0 else WHITE
    fg=WHITE if bg==BLACK else BLACK
    im=Image.new("RGB",(W,H),bg); d=ImageDraw.Draw(im)
    words=[]
    for wi,w in enumerate(scene["words"]):
        if t>=w["start"]-scene["start"]:
            words.append((wi,w))
    if not words:return im
    n=len(scene["words"]); size=int(H*(.075 if n<=5 else .062 if n<=8 else .052))
    maxw=int(W*.88); rows=[]; row=[]; width=0
    names=["Grossa","Grossa","Limpa","Grossa","Serif","Grossa"]
    for wi,w in words:
        name=names[(idx+wi)%len(names)]; path=FONT_PATHS.get(name,FONT_PATHS["Safe"])
        f=font(path,size)
        text=w["word"].upper(); box=d.textbbox((0,0),text,font=f); ww=box[2]-box[0]
        while ww>maxw*.43 and size>32:
            size-=2; f=font(path,size); box=d.textbbox((0,0),text,font=f); ww=box[2]-box[0]
        if row and width+ww+max(8,size//14)>maxw:rows.append(row);row=[];width=0
        row.append((wi,w,f,ww));width+=ww+(max(8,size//14) if len(row)>1 else 0)
    if row:rows.append(row)
    gap=int(size*1.12); total=len(rows)*gap; y=H//2-total//2
    for ri,row in enumerate(rows):
        spacing=max(8,size//14); totalw=sum(x[3] for x in row)+spacing*(len(row)-1); x=(W-totalw)/2
        for wi,w,f,ww in row:
            text=w["word"].upper(); color=ROYAL if idx%2==0 and (wi==1 or (n>=5 and wi==n//2)) else fg
            p=max(0,min(1,(t-(w["start"]-scene["start"]))/.16)); alpha=int(255*(1-(1-p)**3))
            layer=Image.new("RGBA",(ww+24,f.size+30),(0,0,0,0)); ld=ImageDraw.Draw(layer)
            ld.text((12,8),text,font=f,fill=color+(alpha,))
            im=Image.alpha_composite(im.convert("RGBA"),layer,(int(x-12),int(y+ri*gap))).convert("RGB")
            x+=ww+spacing
    return im


def render_video(audio,scenes,out,resolution,status):
    W,H=resolution; dur=duration(audio); frames=[]
    # Instead of rendering every video frame in Python, create one PNG per caption state.
    # FFmpeg then turns those PNGs into the video. This is dramatically lighter on Streamlit.
    tmp=Path(tempfile.mkdtemp(prefix="lyric_frames_")); idx=0
    try:
        states=[]
        for si,s in enumerate(scenes):
            starts=sorted(set([0.0]+[max(0,w["start"]-s["start"]) for w in s["words"]]))
            for k,st in enumerate(starts):
                en=starts[k+1] if k+1<len(starts) else s["end"]-s["start"]
                if en-st<.04:continue
                img=render_image(s,si,st,W,H); p=tmp/f"f{idx:04d}.png"; img.save(p,optimize=True); states.append((p,en-st));idx+=1
        if not states: raise RuntimeError("Nenhum quadro de legenda foi criado.")
        # Fill gaps and keep the last caption until the next phrase.
        concat=tmp/"concat.txt"
        with concat.open("w",encoding="utf-8") as f:
            for p,d in states:
                f.write(f"file '{p.as_posix()}'\nduration {d:.4f}\n")
            f.write(f"file '{states[-1][0].as_posix()}'\n")
        silent=tmp/"silent.mp4"
        status.write("🎬 Montando o vídeo…")
        cmd=[ffmpeg(),"-y","-f","concat","-safe","0","-i",str(concat),"-r",str(FPS),"-c:v","libx264","-preset","veryfast","-crf","20","-pix_fmt","yuv420p","-movflags","+faststart",str(silent)]
        cmdrun(cmd,timeout=max(180,int(dur*10)))
        cmdrun([ffmpeg(),"-y","-i",str(silent),"-i",audio,"-map","0:v:0","-map","1:a:0","-c:v","copy","-c:a","aac","-b:a","160k","-t",f"{dur:.3f}","-movflags","+faststart",out],timeout=max(120,int(dur*5)))
    finally:
        shutil.rmtree(tmp,ignore_errors=True)


st.set_page_config(page_title="Lyric AI Studio",page_icon="🎵")
st.title("🎵 Lyric AI Studio")
st.caption(f"Word-Sync Kinetic Engine · {APP_VERSION}")
audio=st.file_uploader("1. Música ou vídeo",type=["mp3","wav","m4a","mp4","mov","webm"])
lyrics=st.text_area("2. Letra oficial",height=220,placeholder="Uma frase por linha.\n\nOu:\n00:02.3 - 00:06.8\nMinha frase")
model=st.selectbox("Reconhecimento",["small","medium","large-v3-turbo"],index=0)
resolution=st.selectbox("Resolução",["720 × 1280","1080 × 1920"],index=0)
st.caption("Versão leve: sem downloads de fontes na inicialização, modelo em cache e renderização por quadros estáticos.")

if st.button("🚀 CRIAR LYRIC VIDEO",type="primary",use_container_width=True):
    if not audio: st.error("Envie a música primeiro."); st.stop()
    temp=Path(tempfile.mkdtemp(prefix="lyric_ai_")); ap=temp/"audio_input"+Path(audio.name).suffix; ap.write_bytes(audio.getbuffer())
    status=st.empty()
    try:
        dur=duration(str(ap)); status.write(f"⏱️ Duração: {dur:.2f}s")
        asr,lang=transcribe(str(ap),model,status)
        if not asr: raise RuntimeError("O reconhecimento não encontrou palavras.")
        scenes=build_scenes(lyrics,asr,dur) if lyrics.strip() else auto_scenes(asr,dur)
        if not scenes: scenes=auto_scenes(asr,dur)
        if not scenes: raise RuntimeError("Não foi possível criar as frases sincronizadas.")
        total=sum(len(s["words"]) for s in scenes)
        status.write(f"📝 {len(scenes)} frases · {total} palavras")
        size=(720,1280) if resolution.startswith("720") else (1080,1920)
        out=temp/"lyric_ai_final.mp4"; render_video(str(ap),scenes,str(out),size,status)
        status.success("✅ Vídeo criado.")
        st.video(str(out)); st.download_button("⬇️ BAIXAR MP4",out.read_bytes(),"lyric_ai_final.mp4","video/mp4",use_container_width=True)
        with st.expander("Diagnóstico"):
            st.write(f"Duração: **{dur:.2f}s**"); st.write(f"Frases: **{len(scenes)}**"); st.write(f"Palavras: **{total}**"); st.write(f"Idioma: **{lang}**")
            st.code("\n".join(f"{s['start']:.2f}-{s['end']:.2f} | {s['text']}" for s in scenes))
    except Exception as e:
        st.error("❌ A geração falhou."); st.code(str(e))