#!/usr/bin/env python3
"""
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  CAT'S HQRIPPER 0.1 - SiIvaGunner High Quality Rip Generator                 ┃
┃  Team Flames / Samsoft / Flames Co.                                          ┃
┃  NO LIBROSA NEEDED - Works with scipy/pydub only!                            ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
"""

import subprocess, sys, os

def pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

def check_deps():
    deps = {'numpy':'numpy','scipy':'scipy','pydub':'pydub','soundfile':'soundfile','yt_dlp':'yt-dlp'}
    for mod, pkg in deps.items():
        try: __import__(mod)
        except ImportError:
            print(f"[CAT'S HQRIP] Installing {pkg}...")
            try: pip_install(pkg)
            except: pass

print("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
print("┃  CAT'S HQRIPPER 0.1 - Loading...         ┃")
print("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
check_deps()

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import tempfile, shutil, wave, struct, math, random, threading
from datetime import datetime

try:
    import numpy as np
    HAS_NP = True
except: HAS_NP = False

try:
    from scipy import signal
    from scipy.io import wavfile
    from scipy.ndimage import uniform_filter1d
    HAS_SP = True
except: HAS_SP = False

try:
    from pydub import AudioSegment
    HAS_PD = True
except: HAS_PD = False

try:
    import soundfile as sf
    HAS_SF = True
except: HAS_SF = False

try:
    import yt_dlp
    HAS_YT = True
except: HAS_YT = False

APP = "Cat's HQRIPPER 0.1"
SR = 44100
QUOTES = ["Please read the channel description.","GRAND DAD?! FLEENSTONES?!",
          "Todokete...","It's the Nutshack!","Now that's a high quality rip!"]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AUDIO ENGINE (NO LIBROSA)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AudioEngine:
    def __init__(self, log=print, prog=lambda x:None):
        self.log, self.prog = log, prog
        self.tmp = tempfile.mkdtemp(prefix="hqrip_")
    
    def cleanup(self):
        try: shutil.rmtree(self.tmp)
        except: pass
    
    def download(self, url, name="track"):
        if os.path.exists(url):
            out = os.path.join(self.tmp, f"{name}.wav")
            if url.endswith('.wav'): shutil.copy(url, out)
            elif HAS_PD: AudioSegment.from_file(url).export(out, format='wav')
            else: subprocess.run(['ffmpeg','-y','-i',url,out], capture_output=True)
            return out
        if not HAS_YT: raise Exception("yt-dlp not installed!")
        self.log(f"[DL] {url[:50]}...")
        opts = {'format':'bestaudio/best','outtmpl':os.path.join(self.tmp,f"{name}.%(ext)s"),
                'postprocessors':[{'key':'FFmpegExtractAudio','preferredcodec':'wav'}],'quiet':True}
        with yt_dlp.YoutubeDL(opts) as ydl: ydl.download([url])
        for f in os.listdir(self.tmp):
            if f.startswith(name) and f.endswith('.wav'): return os.path.join(self.tmp, f)
        raise Exception("Download failed")
    
    def load(self, path):
        if HAS_PD:
            a = AudioSegment.from_file(path).set_channels(1).set_frame_rate(SR)
            return np.array(a.get_array_of_samples()).astype(np.float32)/32768, SR
        elif HAS_SF:
            d, sr = sf.read(path)
            if len(d.shape)>1: d = d.mean(axis=1)
            if sr != SR: d = signal.resample(d, int(len(d)*SR/sr))
            return d.astype(np.float32), SR
        elif HAS_SP:
            sr, d = wavfile.read(path)
            if len(d.shape)>1: d = d.mean(axis=1)
            d = d.astype(np.float32)/32768
            if sr != SR: d = signal.resample(d, int(len(d)*SR/sr))
            return d, SR
        raise Exception("No audio library!")
    
    def detect_bpm(self, y, sr):
        if not HAS_NP or not HAS_SP: return 120.0
        hop, frame = 512, 2048
        n = (len(y)-frame)//hop+1
        if n < 10: return 120.0
        energy = np.array([np.sum(y[i*hop:i*hop+frame]**2) for i in range(n)])
        onset = np.diff(energy); onset[onset<0] = 0
        if len(onset)>10: onset = uniform_filter1d(onset, 5)
        th = np.mean(onset)+0.5*np.std(onset)
        peaks = [i for i in range(1,len(onset)-1) if onset[i]>th and onset[i]>onset[i-1] and onset[i]>onset[i+1]]
        if len(peaks)<4: return 120.0
        intervals = np.diff(peaks)*hop/sr
        intervals = intervals[(intervals>0.2)&(intervals<2.0)]
        if len(intervals)<2: return 120.0
        bpm = 60.0/np.median(intervals)
        while bpm<70: bpm*=2
        while bpm>180: bpm/=2
        return bpm
    
    def stretch(self, y, sr, tbpm, sbpm):
        if abs(tbpm-sbpm)<2: return y
        rate = sbpm/tbpm
        if HAS_SP: return signal.resample(y, int(len(y)/rate)).astype(np.float32)
        return np.interp(np.linspace(0,len(y)-1,int(len(y)/rate)), np.arange(len(y)), y).astype(np.float32)
    
    def pitch(self, y, sr, st):
        if st==0: return y
        f = 2**(st/12.0)
        if HAS_SP:
            sh = signal.resample(y, int(len(y)/f))
            return signal.resample(sh, len(y)).astype(np.float32)
        return y
    
    def norm(self, y, db=-3): 
        pk = np.max(np.abs(y))
        return y*(10**(db/20)/pk) if pk>0 else y
    
    def eq(self, y, sr, l=1, m=1, h=1):
        if not HAS_SP: return y
        nyq = sr/2
        bl,al = signal.butter(4,200/nyq,'low')
        bm,am = signal.butter(4,[200/nyq,2000/nyq],'band')
        bh,ah = signal.butter(4,2000/nyq,'high')
        return (signal.filtfilt(bl,al,y)*l+signal.filtfilt(bm,am,y)*m+signal.filtfilt(bh,ah,y)*h).astype(np.float32)
    
    def fade(self, y, fi=0.1, fo=2.0):
        env = np.ones(len(y))
        fis, fos = int(SR*fi), int(SR*fo)
        if fis>0 and fis<len(y): env[:fis] = np.linspace(0,1,fis)
        if fos>0 and fos<len(y): env[-fos:] = np.linspace(1,0,fos)
        return y*env
    
    def mix_blend(self, y1, y2, r=0.5): return (y1*r+y2*(1-r)).astype(np.float32)
    def mix_alt(self, y1, y2, sr, bpm, bars=4):
        spb = int((60/bpm)*sr*4)*bars
        res = np.zeros(len(y1),dtype=np.float32); use1 = True
        for i in range(0,len(y1),spb):
            res[i:min(i+spb,len(y1))] = y1[i:min(i+spb,len(y1))] if use1 else y2[i:min(i+spb,len(y1))]
            use1 = not use1
        return res
    def mix_layer(self, y1, y2, sr): return (self.eq(y1,sr,1.2,0.7,0.4)*0.6+self.eq(y2,sr,0.4,0.7,1.2)*0.6).astype(np.float32)
    def mix_sc(self, y1, y2, sr, bpm):
        beat = int((60/bpm)*sr)
        env = np.ones(len(y1),dtype=np.float32)
        for i in range(0,len(y1),beat):
            d = min(int(beat*0.3),len(y1)-i)
            if d>0: env[i:i+d] = np.concatenate([np.linspace(1,0.3,d//2),np.linspace(0.3,1,d-d//2)])[:d]
        return (y1*0.7+y2*env*0.5).astype(np.float32)
    
    def export(self, y, sr, path):
        ext = os.path.splitext(path)[1].lower()
        wav = path if ext=='.wav' else path.replace(ext,'.wav')
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        if HAS_SF: sf.write(wav, y, sr)
        elif HAS_SP: wavfile.write(wav, sr, (y*32767).astype(np.int16))
        else:
            with wave.open(wav,'w') as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
                for s in y: w.writeframesraw(struct.pack('<h',int(max(-1,min(1,s))*32767)))
        if ext=='.mp3':
            try:
                if HAS_PD: AudioSegment.from_wav(wav).export(path,format='mp3',bitrate='320k'); os.remove(wav)
                else: subprocess.run(['ffmpeg','-y','-i',wav,'-b:a','320k',path],capture_output=True); os.remove(wav)
            except: pass
        elif ext=='.mp4':
            try:
                subprocess.run(['ffmpeg','-y','-f','lavfi','-i',f'color=c=0x2d2d44:s=1920x1080:d={len(y)/sr}',
                               '-i',wav,'-c:v','libx264','-c:a','aac','-b:a','320k','-shortest',path],capture_output=True)
                os.remove(wav)
            except: pass
    
    def create_rip(self, url1, url2, out, mode='blend', bpm=None, p1=0, p2=0, eq1=(1,1,1), eq2=(1,1,1)):
        self.log("━"*50); self.log("  HIGH QUALITY RIP ENGINE"); self.log("━"*50)
        self.prog(5)
        try:
            self.log("\n[1/5] Downloading..."); self.prog(10)
            t1 = self.download(url1,"t1"); self.prog(25)
            t2 = self.download(url2,"t2"); self.prog(40)
            self.log("[2/5] Loading..."); y1,sr = self.load(t1); y2,_ = self.load(t2); self.prog(50)
            self.log("[3/5] Analyzing...")
            bpm1, bpm2 = self.detect_bpm(y1,sr), self.detect_bpm(y2,sr)
            self.log(f"  T1:{bpm1:.0f} T2:{bpm2:.0f} BPM")
            if bpm is None: bpm = round((bpm1+bpm2)/2)
            self.log(f"  Target:{bpm} BPM"); self.prog(55)
            self.log("[4/5] Processing...")
            y1, y2 = self.stretch(y1,sr,bpm,bpm1), self.stretch(y2,sr,bpm,bpm2)
            if p1: y1 = self.pitch(y1,sr,p1)
            if p2: y2 = self.pitch(y2,sr,p2)
            ml = max(len(y1),len(y2))
            if len(y1)<ml: y1 = np.tile(y1,int(np.ceil(ml/len(y1))))[:ml]
            if len(y2)<ml: y2 = np.tile(y2,int(np.ceil(ml/len(y2))))[:ml]
            y1, y2 = self.norm(y1,-6), self.norm(y2,-6)
            y1, y2 = self.eq(y1,sr,*eq1), self.eq(y2,sr,*eq2); self.prog(70)
            self.log(f"[5/5] Mixing ({mode})...")
            if mode=='blend': rip = self.mix_blend(y1,y2,0.5)
            elif mode=='alternate': rip = self.mix_alt(y1,y2,sr,bpm,4)
            elif mode=='layer': rip = self.mix_layer(y1,y2,sr)
            elif mode=='sidechain': rip = self.mix_sc(y1,y2,sr,bpm)
            else: rip = self.mix_blend(y1,y2,0.5)
            rip = self.fade(self.norm(rip,-1),0.1,2.0); self.prog(90)
            self.export(rip,sr,out); self.prog(100)
            self.log(""); self.log("━"*50)
            self.log(f"  ✓ DONE! {out}")
            self.log(f"  \"{random.choice(QUOTES)}\""); self.log("━"*50)
            return out
        finally: self.cleanup()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DAW TRACK (COMPACT)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DAWTrack(tk.Frame):
    def __init__(self, parent, name, color, **kw):
        super().__init__(parent, bg='#1e1e2e', **kw)
        self.color = color
        
        # Row 1: Name + URL
        r1 = tk.Frame(self, bg='#1e1e2e')
        r1.pack(fill='x', padx=3, pady=2)
        tk.Label(r1, text=name, font=('Consolas',9,'bold'), bg='#1e1e2e', fg=color, width=18, anchor='w').pack(side='left')
        self.url = tk.Entry(r1, bg='#0d0d1a', fg='#fff', insertbackground='#fff', font=('Consolas',9))
        self.url.pack(side='left', fill='x', expand=True, padx=2)
        tk.Button(r1, text="📁", bg='#3d3d5c', fg='#fff', width=2, font=('',8),
                 command=self.browse).pack(side='left', padx=1)
        tk.Button(r1, text="📋", bg='#3d3d5c', fg='#fff', width=2, font=('',8),
                 command=self.paste).pack(side='left')
        
        # Row 2: Controls
        r2 = tk.Frame(self, bg='#1e1e2e')
        r2.pack(fill='x', padx=3, pady=2)
        
        tk.Label(r2, text="Vol", bg='#1e1e2e', fg='#666', font=('Consolas',8)).pack(side='left')
        self.vol = tk.Scale(r2, from_=0, to=100, orient='horizontal', bg='#2d2d44', fg='#fff',
                           highlightthickness=0, length=50, width=10, showvalue=False)
        self.vol.set(80); self.vol.pack(side='left', padx=2)
        
        tk.Label(r2, text="Pitch", bg='#1e1e2e', fg='#666', font=('Consolas',8)).pack(side='left', padx=(5,0))
        self.pitch_sp = tk.Spinbox(r2, from_=-12, to=12, width=3, bg='#0d0d1a', fg='#fff', font=('Consolas',9))
        self.pitch_sp.delete(0,'end'); self.pitch_sp.insert(0,"0"); self.pitch_sp.pack(side='left', padx=2)
        
        tk.Label(r2, text="EQ:", bg='#1e1e2e', fg='#666', font=('Consolas',8)).pack(side='left', padx=(8,2))
        self.eq_l = tk.Scale(r2, from_=200, to=0, orient='vertical', bg='#2d2d44', fg='#fff',
                            highlightthickness=0, length=25, width=10, showvalue=False); self.eq_l.set(100)
        self.eq_l.pack(side='left', padx=1)
        self.eq_m = tk.Scale(r2, from_=200, to=0, orient='vertical', bg='#2d2d44', fg='#fff',
                            highlightthickness=0, length=25, width=10, showvalue=False); self.eq_m.set(100)
        self.eq_m.pack(side='left', padx=1)
        self.eq_h = tk.Scale(r2, from_=200, to=0, orient='vertical', bg='#2d2d44', fg='#fff',
                            highlightthickness=0, length=25, width=10, showvalue=False); self.eq_h.set(100)
        self.eq_h.pack(side='left', padx=1)
        tk.Label(r2, text="L M H", bg='#1e1e2e', fg='#444', font=('Consolas',7)).pack(side='left')
    
    def browse(self):
        p = filedialog.askopenfilename(filetypes=[("Audio","*.mp3 *.wav *.flac *.m4a"),("All","*.*")])
        if p: self.url.delete(0,'end'); self.url.insert(0,p)
    def paste(self):
        try: self.url.delete(0,'end'); self.url.insert(0,self.winfo_toplevel().clipboard_get().strip())
        except: pass
    def get_url(self): return self.url.get().strip()
    def get_pitch(self):
        try: return int(self.pitch_sp.get())
        except: return 0
    def get_eq(self): return (self.eq_l.get()/100, self.eq_m.get()/100, self.eq_h.get()/100)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN GUI (FIXED COMPACT SIZE)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CatsHQRipper:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(APP)
        self.root.geometry("580x620")  # FIXED SIZE
        self.root.resizable(False, False)
        self.root.configure(bg='#f0f0f0')
        self.processing = False
        self.build_ui()
    
    def build_ui(self):
        # Menu bar
        menu = tk.Frame(self.root, bg='#e0e0e0', height=24)
        menu.pack(fill='x')
        menu.pack_propagate(False)
        for txt, idx in [("🔴 Upload",0),("🎛 DAW",1),("📥 Console",2),("⚙ Settings",3)]:
            tk.Button(menu, text=txt, relief='flat', bg='#e0e0e0', font=('Segoe UI',8), padx=6,
                     command=lambda i=idx: self.nb.select(i)).pack(side='left', padx=1, pady=1)
        
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill='both', expand=True, padx=4, pady=4)
        
        self.build_upload(); self.build_daw(); self.build_console(); self.build_settings()
    
    def build_upload(self):
        tab = ttk.Frame(self.nb, padding=5)
        self.nb.add(tab, text="Upload")
        
        # Quick Mashup
        qm = ttk.LabelFrame(tab, text="🎵 Quick Mashup", padding=4)
        qm.pack(fill='x', pady=(0,5))
        
        for lbl, attr in [("Track 1:","t1_e"),("Track 2:","t2_e")]:
            r = ttk.Frame(qm); r.pack(fill='x', pady=1)
            ttk.Label(r, text=lbl, width=8).pack(side='left')
            e = ttk.Entry(r, width=45); e.pack(side='left', padx=2); setattr(self, attr, e)
            ttk.Button(r, text="📁", width=3, command=lambda en=e: self.browse(en)).pack(side='left')
        
        mr = ttk.Frame(qm); mr.pack(fill='x', pady=3)
        ttk.Label(mr, text="Mode:", width=8).pack(side='left')
        self.mode_v = tk.StringVar(value='blend')
        for txt, val in [('Blend','blend'),('Alternate','alternate'),('Layer','layer'),('Sidechain','sidechain')]:
            ttk.Radiobutton(mr, text=txt, value=val, variable=self.mode_v).pack(side='left', padx=3)
        
        # Output
        of = ttk.LabelFrame(tab, text="Output", padding=4)
        of.pack(fill='x', pady=(0,5))
        
        or1 = ttk.Frame(of); or1.pack(fill='x', pady=1)
        ttk.Label(or1, text="Directory:", width=8).pack(side='left')
        self.out_dir = ttk.Entry(or1, width=38); self.out_dir.pack(side='left', padx=2)
        self.out_dir.insert(0, os.getcwd())
        ttk.Button(or1, text="Browse", width=7, command=self.browse_dir).pack(side='left')
        
        or2 = ttk.Frame(of); or2.pack(fill='x', pady=1)
        ttk.Label(or2, text="Format:", width=8).pack(side='left')
        self.fmt_cb = ttk.Combobox(or2, values=['.mp4','.mp3','.wav'], width=8)
        self.fmt_cb.set('.mp4'); self.fmt_cb.pack(side='left', padx=2)
        
        # Create button
        rf = ttk.Frame(tab); rf.pack(fill='x', pady=8)
        self.rip_btn = tk.Button(rf, text="🎮 CREATE HIGH QUALITY RIP™", font=('Segoe UI',11,'bold'),
                                bg='#4a86c7', fg='white', padx=10, pady=6, command=self.quick_rip)
        self.rip_btn.pack(fill='x', padx=10)
        
        self.prog_v = tk.DoubleVar()
        ttk.Progressbar(rf, variable=self.prog_v, maximum=100).pack(fill='x', padx=10, pady=4)
        ttk.Label(rf, text='Output: HIGH_QUALITY_RIP.mp4', font=('Segoe UI',8), foreground='gray').pack()
    
    def build_daw(self):
        tab = tk.Frame(self.nb, bg='#1e1e2e')
        self.nb.add(tab, text="🎛 DAW")
        
        # Header
        hdr = tk.Frame(tab, bg='#0d0d1a', height=30)
        hdr.pack(fill='x'); hdr.pack_propagate(False)
        tk.Label(hdr, text="🎛 CAT'S DAW", font=('Consolas',11,'bold'),
                bg='#0d0d1a', fg='#00ff88').pack(side='left', padx=8, pady=4)
        
        tr = tk.Frame(hdr, bg='#0d0d1a'); tr.pack(side='right', padx=8)
        tk.Label(tr, text="BPM:", bg='#0d0d1a', fg='#888', font=('Consolas',9)).pack(side='left')
        self.bpm_e = tk.Entry(tr, width=5, bg='#1e1e2e', fg='#fff', font=('Consolas',9))
        self.bpm_e.insert(0,"auto"); self.bpm_e.pack(side='left', padx=3)
        
        # Tracks
        tf = tk.Frame(tab, bg='#1e1e2e'); tf.pack(fill='x', padx=6, pady=6)
        self.daw_t1 = DAWTrack(tf, "TRACK 1 (Source)", "#ff6b6b"); self.daw_t1.pack(fill='x', pady=(0,4))
        self.daw_t2 = DAWTrack(tf, "TRACK 2 (Mashup)", "#4ecdc4"); self.daw_t2.pack(fill='x')
        
        # Master
        mf = tk.Frame(tab, bg='#2d2d44'); mf.pack(fill='x', padx=6, pady=4)
        
        mc = tk.Frame(mf, bg='#2d2d44'); mc.pack(pady=4)
        tk.Label(mc, text="Mode:", bg='#2d2d44', fg='#888', font=('Consolas',9)).pack(side='left', padx=4)
        self.daw_mode = tk.StringVar(value='blend')
        for txt, val in [('Blend','blend'),('Alternate','alternate'),('Layer','layer'),('Sidechain','sidechain')]:
            tk.Radiobutton(mc, text=txt, value=val, variable=self.daw_mode, bg='#2d2d44', fg='#fff',
                          selectcolor='#4a4a6a', font=('Consolas',9)).pack(side='left', padx=4)
        
        oc = tk.Frame(mf, bg='#2d2d44'); oc.pack(pady=4)
        tk.Label(oc, text="Output:", bg='#2d2d44', fg='#888', font=('Consolas',9)).pack(side='left', padx=4)
        self.daw_out = tk.Entry(oc, width=25, bg='#1e1e2e', fg='#fff', font=('Consolas',9))
        self.daw_out.insert(0,"HIGH_QUALITY_RIP"); self.daw_out.pack(side='left', padx=3)
        self.daw_fmt = ttk.Combobox(oc, values=['.mp4','.mp3','.wav'], width=5)
        self.daw_fmt.set('.mp4'); self.daw_fmt.pack(side='left', padx=3)
        
        # Render
        self.daw_btn = tk.Button(tab, text="🔥 RENDER HIGH QUALITY RIP", font=('Consolas',12,'bold'),
                                bg='#9933ff', fg='white', padx=15, pady=8, command=self.daw_render)
        self.daw_btn.pack(fill='x', padx=6, pady=6)
        
        self.daw_prog_v = tk.DoubleVar()
        ttk.Progressbar(tab, variable=self.daw_prog_v, maximum=100).pack(fill='x', padx=6, pady=(0,4))
    
    def build_console(self):
        tab = ttk.Frame(self.nb, padding=4)
        self.nb.add(tab, text="Console")
        self.console = scrolledtext.ScrolledText(tab, width=65, height=25, font=('Consolas',9), bg='#0d0d1a', fg='#00ff88')
        self.console.pack(fill='both', expand=True)
        self.log("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
        self.log("┃  CAT'S HQRIPPER 0.1 - Console              ┃")
        self.log("┃  NO LIBROSA NEEDED!                        ┃")
        self.log("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
        self.log(f"\n[SYS] NumPy:{'✓' if HAS_NP else '✗'} SciPy:{'✓' if HAS_SP else '✗'} PyDub:{'✓' if HAS_PD else '✗'} yt-dlp:{'✓' if HAS_YT else '✗'}")
        self.log(f"\n[READY] {random.choice(QUOTES)}")
    
    def build_settings(self):
        tab = ttk.Frame(self.nb, padding=8)
        self.nb.add(tab, text="Settings")
        af = ttk.LabelFrame(tab, text="About", padding=8); af.pack(fill='x', pady=5)
        ttk.Label(af, text=f"{APP}\nSiIvaGunner High Quality Rip Generator\nTeam Flames / Samsoft\n\nNO LIBROSA NEEDED!\nUses scipy + pydub only.\n\n\"Please read the channel description.\"",
                 font=('Segoe UI',9), justify='left').pack(anchor='w')
    
    def log(self, msg):
        self.console.insert('end', msg+'\n'); self.console.see('end'); self.root.update_idletasks()
    def set_prog(self, v):
        self.prog_v.set(v); self.daw_prog_v.set(v); self.root.update_idletasks()
    def browse(self, e):
        p = filedialog.askopenfilename(filetypes=[("Audio","*.mp3 *.wav *.flac *.m4a"),("All","*.*")])
        if p: e.delete(0,'end'); e.insert(0,p)
    def browse_dir(self):
        p = filedialog.askdirectory()
        if p: self.out_dir.delete(0,'end'); self.out_dir.insert(0,p)
    
    def quick_rip(self):
        if self.processing: messagebox.showwarning("Busy","Processing!"); return
        t1, t2 = self.t1_e.get().strip(), self.t2_e.get().strip()
        if not t1 or not t2: messagebox.showerror("Error","Enter both tracks!"); return
        self.processing = True; self.rip_btn.config(state='disabled', text="⏳ Processing...")
        self.nb.select(2)
        out = os.path.join(self.out_dir.get(), "HIGH_QUALITY_RIP"+self.fmt_cb.get())
        threading.Thread(target=self._rip, args=(t1,t2,out,self.mode_v.get(),None,0,0,(1,1,1),(1,1,1)), daemon=True).start()
    
    def daw_render(self):
        if self.processing: messagebox.showwarning("Busy","Processing!"); return
        t1, t2 = self.daw_t1.get_url(), self.daw_t2.get_url()
        if not t1 or not t2: messagebox.showerror("Error","Enter both tracks!"); return
        self.processing = True; self.daw_btn.config(state='disabled', text="⏳ Rendering...")
        self.nb.select(2)
        out = os.path.join(self.out_dir.get(), self.daw_out.get()+self.daw_fmt.get())
        bpm_t = self.bpm_e.get().strip().lower()
        bpm = None if bpm_t=='auto' else int(bpm_t)
        threading.Thread(target=self._rip, args=(t1,t2,out,self.daw_mode.get(),bpm,
                        self.daw_t1.get_pitch(),self.daw_t2.get_pitch(),
                        self.daw_t1.get_eq(),self.daw_t2.get_eq()), daemon=True).start()
    
    def _rip(self, t1, t2, out, mode, bpm, p1, p2, eq1, eq2):
        try:
            AudioEngine(self.log, self.set_prog).create_rip(t1,t2,out,mode,bpm,p1,p2,eq1,eq2)
            self.root.after(0, lambda: messagebox.showinfo("✓ Done!", f"High Quality Rip Complete!\n\n{out}"))
        except Exception as e:
            self.log(f"[ERROR] {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.processing = False
            self.root.after(0, lambda: self.rip_btn.config(state='normal', text="🎮 CREATE HIGH QUALITY RIP™"))
            self.root.after(0, lambda: self.daw_btn.config(state='normal', text="🔥 RENDER HIGH QUALITY RIP"))
            self.root.after(0, lambda: self.set_prog(0))
    
    def run(self): self.root.mainloop()

if __name__ == "__main__":
    CatsHQRipper().run()
