import os, re, math, shutil, subprocess, tempfile, urllib.request, difflib, unicodedata
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

APP_VERSION = "8.0-ROYAL-SYNC"
FPS = 30
W, H = 1080, 1920

CACHE = Path(".lyric_cache")
FONT_DIR = CACHE / "fonts"
FONT_DIR.mkdir(parents=True, exist_ok=True)

FONT_SOURCES = {
    "Anton": "https://raw.githubusercontent.com/google/fonts/main/ofl/anton/Anton-Regular.ttf",
    "Bebas Neue": "https://raw.githubusercontent.com/google/fonts/main/ofl/bebasneue/BebasNeue-Regular.ttf",
    "Montserrat": "https://raw.githubusercontent.com/google/fonts/main/ofl/montserrat/Montserrat%5Bwght%5D.ttf",
    "Oswald": "https://raw.githubusercontent.com/google/fonts/main/ofl/oswald/Oswald%5Bwght%5D.ttf",
    "Archivo Black": "https://raw.githubusercontent.com/google/fonts/main/ofl/archivoblack/ArchivoBlack-Regular.ttf",
    "DM Serif Display": "https://raw.githubusercontent.com/google/fonts/main/ofl/dmserifdisplay/DMSerifDisplay-Regular.ttf",
    "Playfair Display": "https://raw.githubusercontent.com/google/fonts/main/ofl/playfairdisplay/PlayfairDisplay%5Bwght%5D.ttf",
    "Libre Baskerville": "https://raw.githubusercontent.com/google/fonts/main/ofl/librebaskerville/LibreBaskerville-Regular.ttf",
}
SYSTEM_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
]
BLACK=(5,5,7); BLACK2=(12,12,15); WHITE=(248,248,246); ROYAL=(45,92,255)

def clamp(x,a,b): return max(a,min(b,x))
def ease(t): t=clamp(t,0,1); return 1-(1-t)**3
def smooth(t): t=clamp(t,0,1); return t*t*(3-2*t)
def safe(s): return re.sub(r"[^A-Za-z0-9_.-]+","_",s)[:100]
def norm(s):
    return re.sub(r"\s+"," ",s or "").strip()
def token(s):
    s=unicodedata.normalize("NFKD",s or "")
    s="".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-zA-Z0-9]","",s).lower()
def sim(a,b):
    a,b=token(a),token(b)
    if not a or not b:return 0
    if a==b:return 1
    if a in b or b in a:return .90
    return difflib.SequenceMatcher(None,a,b,autojunk=False).ratio()

def ffmpeg():
    try:
        import imageio_ffmpeg
        p=imageio_ffmpeg.get_ffmpeg_exe()
        if p:return p
    except Exception: pass
    p=shutil.which("ffmpeg")
    if p:return p
    raise RuntimeError("FFmpeg nÃ£o encontrado.")

def run(cmd, timeout=300):
    p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=timeout)
    if p.returncode: raise RuntimeError(p.stderr[-7000:])
    return p.stdout

def duration(path):
    ff=ffmpeg()
    p=subprocess.run([ff,"-hide_banner","-i",path,"-f","null","-"],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=60)
    m=re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)",p.stderr)
    return int(m.group(1))*3600+int(m.group(2))*60+float(m.group(3)) if m else 0.0

def audio_end(path,dur,latest_word=0):
    # Conservative ending: silence detection is only allowed to shorten the file
    # when a real final silence exists; the last recognized word gets a safety tail.
    end=dur
    ff=ffmpeg()
    try:
        p=subprocess.run([ff,"-hide_banner","-i",path,"-af","silencedetect=noise=-40dB:d=0.55","-f","null","-"],
                         stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=max(60,int(dur*2)))
        starts=[float(x) for x in re.findall(r"silence_start:\s*([\d.]+)",p.stderr)]
        ends=[float(x) for x in re.findall(r"silence_end:\s*([\d.]+)",p.stderr)]
        if starts and starts[-1]>dur*.72:
            end=min(end,starts[-1]+.25)
    except Exception: pass
    if latest_word>0:
        # Do not cut a genuine last syllable; 0.65s is a better safety margin for singing.
        end=min(end,max(latest_word+.65, dur if dur-latest_word<.35 else latest_word+.65))
    return max(.5,end)

@st.cache_resource(show_spinner=False)
def fonts():
    out={}
    for name,url in FONT_SOURCES.items():
        p=FONT_DIR/(safe(name)+".ttf")
        if not p.exists() or p.stat().st_size<10000:
            try:
                req=urllib.request.Request(url,headers={"User-Agent":"LyricAIStudio"})
                with urllib.request.urlopen(req,timeout=25) as r:p.write_bytes(r.read())
            except Exception: pass
        if p.exists() and p.stat().st_size>10000: out[name]=str(p)
    if not out:
        for p in SYSTEM_FONTS:
            if os.path.exists(p): out["System"]=p; break
    return out

def fit_font(text,maxw,size,path,minsize=34):
    while size>=minsize:
        f=ImageFont.truetype(path,size)
        b=f.getbbox(text)
        if b[2]-b[0]<=maxw:return f
        size-=2
    return ImageFont.truetype(path,minsize)

@st.cache_resource(show_spinner=False)
def whisper(model):
    from faster_whisper import WhisperModel
    return WhisperModel(model,device="cpu",compute_type="int8",
                        cpu_threads=max(2,min(8,os.cpu_count() or 4)),num_workers=1)

def transcribe(path,model,status=None):
    m=whisper(model)
    if status: status.write(f"ðï¸ Transcrevendo palavra por palavra com **{model}**â¦")
    segs,info=m.transcribe(
        path,language="pt",task="transcribe",
        beam_size=8,best_of=8,patience=1.2,
        temperature=0.0,compression_ratio_threshold=2.8,
        log_prob_threshold=-1.2,no_speech_threshold=0.20,
        condition_on_previous_text=True,vad_filter=False,
        word_timestamps=True,
        initial_prompt=("Letra de mÃºsica brasileira cantada em portuguÃªs. "
                        "ReconheÃ§a palavras curtas e rÃ¡pidas, repetiÃ§Ãµes e contraÃ§Ãµes. "
                        "NÃ£o resuma nem traduza.")
    )
    words=[]
    for seg in segs:
        if not seg.words: continue
        for w in seg.words:
            x=norm(w.word)
            if x:
                words.append({"word":x,"start":float(w.start),"end":float(w.end),
                              "prob":float(getattr(w,"probability",0) or 0)})
    return words,getattr(info,"language","pt"),float(getattr(info,"duration",0) or 0)

def parse_timed(text):
    out=[]
    for line in text.splitlines():
        line=line.strip()
        if not line: continue
        m=re.match(r"^(?:(\d+):)?(\d{1,2}):(\d{2})(?:[.,](\d{1,3}))?\s*\|\s*(.+)$",line)
        if m:
            h=int(m.group(1) or 0); mi=int(m.group(2)); se=int(m.group(3)); fr=m.group(4) or "0"
            out.append({"start":h*3600+mi*60+se+float("0."+fr),"text":norm(m.group(5))})
            continue
        m=re.match(r"^(\d+(?:[.,]\d+)?)\s*\|\s*(.+)$",line)
        if m: out.append({"start":float(m.group(1).replace(",",".")),"text":norm(m.group(2))})
    out.sort(key=lambda x:x["start"])
    return out

def plain_lines(text):
    return [norm(x) for x in text.splitlines() if norm(x)]

def align_phrase(text, start, end, asr):
    toks=re.findall(r"\S+",text)
    cand=[(i,w) for i,w in enumerate(asr) if w["end"]>=start-.25 and w["start"]<=end+.25]
    n=len(toks); m=len(cand)
    if not n:return []
    # Dynamic programming monotonic alignment. Skips ASR words when Whisper invents
    # extras and skips lyric tokens when Whisper misses them.
    dp=np.full((n+1,m+1),-1e9,float); back=np.zeros((n+1,m+1),np.int8)
    dp[0,:]=0
    for i in range(1,n+1):
        for j in range(1,m+1):
            sc=sim(toks[i-1],cand[j-1][1]["word"])
            match=dp[i-1,j-1]+(4.0*sc-0.20)
            skip_lyric=dp[i-1,j]-0.65
            skip_asr=dp[i,j-1]-0.18
            best=max(match,skip_lyric,skip_asr)
            dp[i,j]=best
            back[i,j]=1 if best==match else (2 if best==skip_lyric else 3)
    i,j=n,m; matches={}
    while i>0 and j>0:
        b=back[i,j]
        if b==1:
            score=sim(toks[i-1],cand[j-1][1]["word"])
            if score>=.38: matches[i-1]=cand[j-1][1]
            i-=1;j-=1
        elif b==2:i-=1
        else:j-=1
    # Interpolate missing lyric words only inside this phrase.
    result=[None]*n
    for i,w in matches.items(): result[i]={"word":toks[i],"start":w["start"],"end":w["end"],
                                           "prob":max(w.get("prob",0),sim(toks[i],w["word"])*.75)}
    known=[i for i,x in enumerate(result) if x]
    for i in range(n):
        if result[i]:continue
        prev=max([k for k in known if k<i],default=-1)
        nxt=min([k for k in known if k>i],default=n)
        a=result[prev]["end"] if prev>=0 else start
        b=result[nxt]["start"] if nxt<n else end
        count=max(1,nxt-prev)
        st=a+(b-a)*(i-prev)/count
        en=a+(b-a)*(i-prev+1)/count
        result[i]={"word":toks[i],"start":clamp(st,start,end),"end":clamp(max(st+.055,en),start+.055,end),"prob":.45}
    for w in result:
        w["start"]=clamp(w["start"],start,end)
        w["end"]=clamp(max(w["start"]+.055,w["end"]),w["start"]+.055,end)
    return result

def build_timed(lines,asr,aend):
    scenes=[]
    for i,line in enumerate(lines):
        st=line["start"]
        if st>=aend:break
        en=min(lines[i+1]["start"] if i+1<len(lines) else aend,aend)
        if en<=st+.05:continue
        words=align_phrase(line["text"],st,en,asr)
        if not words:continue
        for w in words:w["phrase_id"]=i;w["phrase_text"]=line["text"]
        scenes.append({"start":max(0,st-.025),"end":min(aend,en+.18),
                       "words":words,"phrase_text":line["text"],"instrumental":False})
    return scenes

def build_plain(text,asr,aend):
    lines=plain_lines(text)
    scenes=[]; cursor=0
    for pid,line in enumerate(lines):
        toks=re.findall(r"\S+",line)
        if not toks:continue
        # Find the first few tokens monotonically; phrase boundaries come from ASR.
        best=None; score=0
        for j in range(cursor,min(len(asr),cursor+45)):
            s=sim(toks[0],asr[j]["word"])
            if s>score:score=s;best=j
        if best is None or score<.35:continue
        st=asr[best]["start"]; idx=best
        # Estimate end by greedily matching the rest, but never allow a huge jump.
        for tok in toks[1:]:
            bi=None;bs=0
            for j in range(idx+1,min(len(asr),idx+25)):
                s=sim(tok,asr[j]["word"])
                if s>bs:bs=s;bi=j
                if s>=.98:break
            if bi is not None and bs>=.35:idx=bi
        en=min(aend,asr[idx]["end"]+.18)
        words=align_phrase(line,st,en,asr)
        if words:
            for w in words:w["phrase_id"]=pid;w["phrase_text"]=line
            scenes.append({"start":max(0,st-.025),"end":en,"words":words,
                           "phrase_text":line,"instrumental":False})
            cursor=idx+1
    return scenes

EMOTIONAL={"amor","saudade","coraÃ§Ã£o","coracao","beijo","vida","nunca","sempre","volta","voltar",
           "paixÃ£o","paixao","desejo","perfume","chora","chorar","quero","meu","minha","tudo","nada","vocÃª","voce"}

@dataclass
class Style:
    bg:Tuple[int,int,int]; fg:Tuple[int,int,int]; font:str; layout:str; blue:bool

def style_for(scene,idx,reg):
    s=sum(ord(c) for c in scene.get("phrase_text",""))+idx*37
    bg=[BLACK,BLACK,BLACK2,(20,20,22),(245,245,243)][s%5]
    fg=WHITE if sum(bg)<300 else BLACK
    available=[x for x in ["Anton","Bebas Neue","Archivo Black","Montserrat","Oswald","DM Serif Display","Playfair Display","Libre Baskerville"] if x in reg]
    font=available[s%len(available)] if available else next(iter(reg))
    # Stack more often for long phrases; otherwise varied but controlled.
    n=len(scene.get("words",[]))
    if n>=9: layout="stack" if s%2==0 else "center"
    elif n>=6: layout=["center","stack","editorial"][s%3]
    else: layout=["hero","center","editorial"][s%3]
    return Style(bg,fg,font,layout,(s%4==0))

def background(style,t):
    arr=np.zeros((H,W,3),np.float32);arr[:]=style.bg
    yy,xx=np.mgrid[0:H,0:W]
    # monochromatic moving light only; no colored background.
    cx=W*(.5+.10*math.sin(t*.23)); cy=H*(.48+.08*math.cos(t*.31))
    d=((xx-cx)/(W*.65))**2+((yy-cy)/(H*.65))**2
    glow=np.exp(-2.2*d)[...,None]
    if sum(style.bg)<300: arr+=glow*7
    else: arr-=glow*7
    rng=np.random.default_rng(int(t*997)%1000003)
    arr+=rng.normal(0,1.3,(H,W,1))
    return Image.fromarray(np.uint8(np.clip(arr,0,255)))

def choose_blue(words,enabled,seed):
    if not enabled or seed%3: return set()
    scored=[]
    for i,w in enumerate(words):
        x=token(w["word"]); sc=len(x)*.3+(4 if x in EMOTIONAL else 0)+(1.5 if len(x)>=8 else 0)
        scored.append((sc,i))
    scored.sort(reverse=True)
    return {scored[0][1]} if scored else set()

def draw_text_layer(text,font,color,alpha=255,scale=1.0,glow=False):
    b=font.getbbox(text);tw=b[2]-b[0];th=b[3]-b[1];pad=50
    layer=Image.new("RGBA",(tw+pad*2,th+pad*2),(0,0,0,0));d=ImageDraw.Draw(layer)
    if glow:
        for sw,a in ((12,18),(7,28),(3,42)):
            d.text((pad,pad),text,font=font,fill=(*color,a),stroke_width=sw,stroke_fill=(*color,a))
    d.text((pad,pad),text,font=font,fill=(*color,int(alpha)))
    if scale!=1:
        layer=layer.resize((int(layer.width*scale),int(layer.height*scale)),Image.Resampling.LANCZOS)
    return layer

def render_words(overlay,scene,font_path,style,local):
    words=scene["words"]; spoken=[(i,w,w["start"]-scene["start"]) for i,w in enumerate(words) if local>=w["start"]-scene["start"]-.01]
    if not spoken:return
    d=ImageDraw.Draw(overlay)
    blue=choose_blue(words,style.blue,hash(scene["phrase_text"])%1000)
    n=len(spoken)
    maxw=int(W*.88)
    # Large typography. Long phrases use stacked words deliberately.
    if style.layout=="stack":
        size=int(H*.078 if n<10 else H*.069)
        f=fit_font("WWWWWWWW",int(W*.72),size,font_path,34)
        visible=spoken[-9:]
        lh=int(f.size*1.02); total=lh*len(visible); y=H/2-total/2
        for i,w,rel in visible:
            word=w["word"].upper();age=local-rel;p=ease(age/.22)
            col=ROYAL if i in blue else style.fg
            layer=draw_text_layer(word,f,col,int(255*p),.94+.06*p,glow=(i in blue))
            x=int((W-layer.width)/2); yy=int(y+(1-p)*22)
            overlay.alpha_composite(layer,(x,yy)); y+=lh
        return
    # center/editorial/hero: wrap by actual pixels.
    size=int(H*.090 if n<=3 else H*.080 if n<=6 else H*.068)
    # Fit each token, not the entire phrase, preserving large words.
    f=fit_font("W"*max(5,max(len(w["word"]) for _,w,_ in spoken)),int(W*.82),size,font_path,32)
    rows=[];row=[];rw=0;space=d.textbbox((0,0)," ",font=f)[2]
    for item in spoken:
        ww=d.textbbox((0,0),item[1]["word"].upper(),font=f)[2]
        if row and rw+space+ww>maxw:
            rows.append((row,rw));row=[item];rw=ww
        else:
            rw += ww+(space if row else 0);row.append(item)
    if row:rows.append((row,rw))
    lh=int(f.size*1.08); y=H/2-(len(rows)*lh)/2
    for ri,(row,rw) in enumerate(rows):
        x=(W-rw)/2
        for i,w,rel in row:
            word=w["word"].upper();ww=d.textbbox((0,0),word,font=f)[2]
            age=local-rel;p=ease(age/.22);col=ROYAL if i in blue else style.fg
            # subtle, fluid entrance: fade + 10px rise + tiny scale.
            layer=draw_text_layer(word,f,col,int(255*p),.965+.035*p,glow=(i in blue))
            overlay.alpha_composite(layer,(int(x-(layer.width-ww)/2),int(y+(1-p)*10-42)))
            x+=ww+space
        y+=lh

def render_frame(scene,style,reg,local,t):
    base=background(style,t).convert("RGBA")
    overlay=Image.new("RGBA",(W,H),(0,0,0,0));d=ImageDraw.Draw(overlay)
    # monochrome decorative motion
    a=int(18+12*(.5+.5*math.sin(t*1.7)))
    d.ellipse((W*.12,H*.18,W*.88,H*.82),outline=style.fg+(a,),width=3)
    d.line((W*.10,H*.84,W*(.35+.08*math.sin(t*.8)),H*.84),fill=style.fg+(35,),width=4)
    font_path=reg[style.font]
    render_words(overlay,scene,font_path,style,local)
    dur=max(.1,scene["end"]-scene["start"])
    # phrase-level fade; no hard pop between phrases
    fade=.20
    if local<fade:
        aa=int(255*smooth(local/fade))
        overlay.putalpha(overlay.getchannel("A").point(lambda x:int(x*aa/255)))
    if dur-local<fade:
        aa=int(255*smooth(max(0,(dur-local)/fade)))
        overlay.putalpha(overlay.getchannel("A").point(lambda x:int(x*aa/255)))
    return Image.alpha_composite(base,overlay).convert("RGB")

def render_video(audio_path,scenes,reg,out_path,res,quality,progress):
    ww,hh=res
    global W,H; W,H=ww,hh
    dur=duration(audio_path)
    last=max((w["end"] for s in scenes for w in s.get("words",[])),default=0)
    end=audio_end(audio_path,dur,last)
    # If supplied timed lyrics contain lines after the last sung phrase, they are
    # automatically ignored because the audio/ASR never reaches them.
    end=min(end,last+.65) if last else end
    ff=ffmpeg(); silent=Path(out_path).with_name("silent.mp4")
    crf="13" if quality=="Alta qualidade" else "16"
    preset="slow" if quality=="Alta qualidade" else "medium"
    cmd=[ff,"-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{ww}x{hh}","-r",str(FPS),"-i","-",
         "-an","-c:v","libx264","-preset",preset,"-crf",crf,"-pix_fmt","yuv420p","-profile:v","high",
         "-movflags","+faststart",str(silent)]
    p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    total=max(1,int(math.ceil(end*FPS)));si=0
    try:
        for fi in range(total):
            t=fi/FPS
            while si+1<len(scenes) and t>=scenes[si]["end"]:si+=1
            if not scenes: scene={"start":0,"end":end,"words":[],"phrase_text":""}
            else: scene=scenes[min(si,len(scenes)-1)]
            style=style_for(scene,si,reg)
            local=clamp(t-scene["start"],0,max(.01,scene["end"]-scene["start"]))
            frame=np.asarray(render_frame(scene,style,reg,local,t),np.uint8)
            p.stdin.write(frame.tobytes())
            if progress and fi%FPS==0:progress.progress(min(.94,fi/total*.94),text=f"Renderizando {int(fi/total*100)}%")
        p.stdin.close()
        err=p.stderr.read().decode("utf8","replace");code=p.wait()
        if code:raise RuntimeError(err[-7000:])
    finally:
        if p.poll() is None:
            try:p.kill()
            except:pass
    run([ff,"-y","-i",str(silent),"-i",audio_path,"-map","0:v:0","-map","1:a:0",
         "-c:v","copy","-c:a","aac","-b:a","256k","-t",f"{end:.3f}","-movflags","+faststart",str(out_path)],
        timeout=max(180,int(end*10)))
    silent.unlink(missing_ok=True)
    if progress:progress.progress(1.0,text="â VÃ­deo concluÃ­do.")
    return end

st.set_page_config(page_title="Lyric AI Studio",page_icon="ðµ",layout="centered")
st.title("ðµ Lyric AI Studio")
st.caption(f"Royal Kinetic Word-Sync Â· {APP_VERSION}")
with st.expander("Como obter a melhor sincronizaÃ§Ã£o",expanded=False):
    st.markdown("""Cole a letra oficial. Para mÃ¡xima precisÃ£o, use:
`00:12.30 | primeira frase`
`00:16.80 | segunda frase`
`00:21.45 | terceira frase`

O tempo Ã© o inÃ­cio da frase. O Whisper procura as palavras dentro dessa janela.
Se a letra continuar depois do fim real da mÃºsica, ela serÃ¡ ignorada.""")
audio=st.file_uploader("1. MÃºsica ou vÃ­deo com a mÃºsica",type=["mp3","wav","m4a","mp4","mov","webm"])
lyrics=st.text_area("2. Letra oficial (recomendada)",height=230,
                    placeholder="00:12.30 | Eu sei que vou te amar\n00:16.80 | Por toda a minha vida\n00:21.45 | Eu vou te amar")
col1,col2=st.columns(2)
with col1:model=st.selectbox("Reconhecimento",["small","medium","large-v3-turbo","large-v3"],index=2)
with col2:quality=st.selectbox("Qualidade",["Alta qualidade","Equilibrado"],index=0)
resolution=st.selectbox("ResoluÃ§Ã£o",["1080Ã1920","720Ã1280"],index=0)
reg=fonts()
st.caption(f"Fontes disponÃ­veis: {len(reg)}")
if st.button("ð CRIAR LYRIC VIDEO",type="primary",use_container_width=True):
    if not audio:st.error("Envie a mÃºsica primeiro.");st.stop()
    tmp=Path(tempfile.mkdtemp(prefix="lyric_ai_"));status=st.empty();bar=st.progress(0)
    try:
        ap=tmp/safe(audio.name);ap.write_bytes(audio.getbuffer())
        dur=duration(str(ap))
        try: asr,lang,dd=transcribe(str(ap),model,status)
        except Exception:
            if model=="small":raise
            status.warning("Tentando o modelo small por seguranÃ§aâ¦")
            asr,lang,dd=transcribe(str(ap),"small",status)
        if not asr:raise RuntimeError("O reconhecimento nÃ£o encontrou palavras. Tente large-v3 ou forneÃ§a a letra com tempos.")
        timed=parse_timed(lyrics) if lyrics.strip() else []
        aend=audio_end(str(ap),dur,max((w["end"] for w in asr),default=0))
        if timed:
            status.write("ð§© Alinhando a letra oficial Ã s palavras reais do cantorâ¦")
            scenes=build_timed(timed,asr,aend)
        elif lyrics.strip():
            status.write("ð§© Alinhando a letra oficial ao Ã¡udioâ¦")
            scenes=build_plain(lyrics,asr,aend)
        else:
            scenes=[]
            cur=[]
            for w in asr:
                if w["start"]>=aend:break
                if cur and (w["start"]-cur[-1]["end"]>.58 or len(cur)>=15 or w["end"]-cur[0]["start"]>7):
                    scenes.append(cur);cur=[]
                cur.append(w)
            if cur:scenes.append(cur)
            scenes=[{"start":x[0]["start"],"end":min(aend,x[-1]["end"]+.18),"words":x,
                     "phrase_text":" ".join(w["word"] for w in x),"instrumental":False} for x in scenes]
        scenes=[s for s in scenes if s["start"]<aend and s["words"]]
        if not scenes:raise RuntimeError("NÃ£o foi possÃ­vel alinhar as frases ao Ã¡udio. Use a letra com timestamps.")
        bar.progress(.15,text="Preparando renderâ¦")
        res=(1080,1920) if resolution.startswith("1080") else (720,1280)
        out=tmp/"lyric_ai_final.mp4"
        final_end=render_video(str(ap),scenes,reg,str(out),res,quality,bar)
        status.success(f"VÃ­deo criado atÃ© {final_end:.2f}s â apenas atÃ© o trecho realmente cantado.")
        st.video(str(out))
        st.download_button("â¬ï¸ Baixar MP4",out.read_bytes(),"lyric_ai_final.mp4","video/mp4",use_container_width=True)
        with st.expander("DiagnÃ³stico"):
            st.write(f"Palavras ASR: **{len(asr)}**")
            st.write(f"Frases renderizadas: **{len(scenes)}**")
            st.write(f"Fim detectado: **{final_end:.2f}s**")
            st.write(f"Idioma: **{lang}**")
    except Exception as e:
        st.error("A geraÃ§Ã£o falhou.")
        st.code(str(e))
    # keep tmp alive for this Streamlit run
