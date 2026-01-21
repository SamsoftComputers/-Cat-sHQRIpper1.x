#!/usr/bin/env python3
"""
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  ██████╗ █████╗ ████████╗███████╗    ██╗  ██╗ ██████╗ ██████╗ ██╗██████╗     ┃
┃ ██╔════╝██╔══██╗╚══██╔══╝██╔════╝    ██║  ██║██╔═══██╗██╔══██╗██║██╔══██╗    ┃
┃ ██║     ███████║   ██║   ███████╗    ███████║██║   ██║██████╔╝██║██████╔╝    ┃
┃ ██║     ██╔══██║   ██║   ╚════██║    ██╔══██║██║▄▄ ██║██╔══██╗██║██╔═══╝     ┃
┃ ╚██████╗██║  ██║   ██║   ███████║    ██║  ██║╚██████╔╝██║  ██║██║██║         ┃
┃  ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝    ╚═╝  ╚═╝ ╚══▀▀═╝ ╚═╝  ╚═╝╚═╝╚═╝         ┃
┃                         Version 0.1 - "First Rip"                             ┃
┃               SiIvaGunner-Style High Quality Rip Generator                    ┃
┃                    Team Flames / Samsoft / Flames Co.                         ┃
┃                   "Please read the channel description."                      ┃
┃                                                                               ┃
┃  NO LIBROSA NEEDED - Works with scipy/pydub only! (No LLVM hassle!)          ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
"""

import subprocess, sys, os

def pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

def check_deps():
    # NO LIBROSA - it requires LLVM which is a pain to install!
    # We use scipy + pydub instead for everything
    deps = {
        'requests': 'requests',
        'numpy': 'numpy', 
        'scipy': 'scipy',
        'pydub': 'pydub',
        'soundfile': 'soundfile',
        'yt_dlp': 'yt-dlp'
    }
    for mod, pkg in deps.items():
        try: 
            __import__(mod)
        except ImportError:
            print(f"[CAT'S HQRIP] Installing {pkg}...")
            try: 
                pip_install(pkg)
                print(f"[CAT'S HQRIP] ✓ {pkg} installed!")
            except Exception as e: 
                print(f"[CAT'S HQRIP] ⚠ Could not install {pkg}: {e}")

print("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
print("┃  CAT'S HQRIPPER 0.1 - Loading...         ┃")
print("┃  (No LLVM/librosa needed!)               ┃")
print("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
check_deps()

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext, Canvas
import re, time, threading, tempfile, shutil, wave, struct, math, random
from datetime import datetime

try:
    import numpy as np
    HAS_NP = True
except: 
    HAS_NP = False

try:
    from scipy import signal
    from scipy.io import wavfile
    from scipy.ndimage import uniform_filter1d
    HAS_SP = True
except: 
    HAS_SP = False

try:
    from pydub import AudioSegment
    HAS_PD = True
except: 
    HAS_PD = False

try:
    import soundfile as sf
    HAS_SF = True
except: 
    HAS_SF = False

try:
    import yt_dlp
    HAS_YT = True
except: 
    HAS_YT = False

APP = "Cat's HQRIPPER 0.1"
SR = 44100
QUOTES = [
    "Please read the channel description.",
    "GRAND DAD?! FLEENSTONES?!",
    "Todokete... setsunasa ni wa~",
    "It's the Nutshack!",
    "Now that's a high quality rip!",
    "Wood Man would be proud.",
    "The Voice Inside Your Head approves.",
    "Chad Warden certified.",
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AUDIO ENGINE - NO LIBROSA NEEDED!
# Uses scipy for signal processing, pydub for loading
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AudioEngine:
    """High Quality Audio Engine - Works WITHOUT librosa!"""
    
    def __init__(self, log=print, prog=lambda x:None):
        self.log = log
        self.prog = prog
        self.tmp = tempfile.mkdtemp(prefix="hqrip_")
    
    def cleanup(self):
        try: 
            shutil.rmtree(self.tmp)
        except: 
            pass
    
    def download(self, url, name="track"):
        """Download audio from URL or copy local file"""
        # Handle local files
        if os.path.exists(url):
            out = os.path.join(self.tmp, f"{name}.wav")
            ext = os.path.splitext(url)[1].lower()
            
            if ext == '.wav':
                shutil.copy(url, out)
            elif HAS_PD:
                self.log(f"[LOCAL] Converting {os.path.basename(url)}...")
                audio = AudioSegment.from_file(url)
                audio.export(out, format='wav')
            else:
                # Try ffmpeg
                subprocess.run(['ffmpeg', '-y', '-i', url, out], 
                              capture_output=True, check=True)
            return out
        
        # Download from URL
        if not HAS_YT:
            raise Exception("yt-dlp not installed! Run: brew install yt-dlp")
        
        self.log(f"[DOWNLOAD] {url[:60]}...")
        
        out_template = os.path.join(self.tmp, f"{name}.%(ext)s")
        
        opts = {
            'format': 'bestaudio/best',
            'outtmpl': out_template,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                title = info.get('title', name)
                self.log(f"[DOWNLOAD] ✓ {title[:45]}")
        except Exception as e:
            raise Exception(f"Download failed: {str(e)[:80]}")
        
        # Find the WAV file
        for f in os.listdir(self.tmp):
            if f.startswith(name) and f.endswith('.wav'):
                return os.path.join(self.tmp, f)
        
        raise Exception("Download completed but WAV file not found")
    
    def load(self, path):
        """Load audio file to numpy array - NO LIBROSA!"""
        self.log(f"[LOAD] {os.path.basename(path)}")
        
        if HAS_PD:
            # Use pydub - most reliable
            audio = AudioSegment.from_file(path)
            audio = audio.set_channels(1).set_frame_rate(SR)
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            samples = samples / 32768.0  # Normalize to -1 to 1
            return samples, SR
        
        elif HAS_SF:
            # Use soundfile
            data, sr = sf.read(path)
            if len(data.shape) > 1:
                data = data.mean(axis=1)  # Convert to mono
            if sr != SR:
                # Resample
                data = signal.resample(data, int(len(data) * SR / sr))
            return data.astype(np.float32), SR
        
        elif HAS_SP:
            # Use scipy.io.wavfile
            sr, data = wavfile.read(path)
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            data = data.astype(np.float32) / 32768.0
            if sr != SR:
                data = signal.resample(data, int(len(data) * SR / sr))
            return data, SR
        
        else:
            raise Exception("No audio loading library available! Install pydub or scipy.")
    
    def detect_bpm(self, y, sr):
        """
        Detect BPM using onset detection - NO LIBROSA!
        Uses scipy for signal processing.
        """
        self.log("[BPM] Analyzing tempo...")
        
        if not HAS_NP or not HAS_SP:
            self.log("[BPM] ⚠ Using default 120 BPM (numpy/scipy not available)")
            return 120.0
        
        # Parameters
        hop_length = 512
        frame_length = 2048
        
        # Calculate spectral flux (onset strength)
        # Split into frames
        n_frames = (len(y) - frame_length) // hop_length + 1
        
        if n_frames < 10:
            return 120.0
        
        # Calculate energy in each frame
        energy = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * hop_length
            frame = y[start:start + frame_length]
            energy[i] = np.sum(frame ** 2)
        
        # Calculate onset strength (difference in energy)
        onset_env = np.diff(energy)
        onset_env[onset_env < 0] = 0  # Half-wave rectify
        
        # Smooth the onset envelope
        if len(onset_env) > 10:
            onset_env = uniform_filter1d(onset_env, size=5)
        
        # Find peaks (onsets)
        threshold = np.mean(onset_env) + 0.5 * np.std(onset_env)
        peaks = []
        
        for i in range(1, len(onset_env) - 1):
            if onset_env[i] > threshold:
                if onset_env[i] > onset_env[i-1] and onset_env[i] > onset_env[i+1]:
                    # Minimum distance between peaks
                    if len(peaks) == 0 or i - peaks[-1] > 5:
                        peaks.append(i)
        
        if len(peaks) < 4:
            self.log("[BPM] ⚠ Not enough beats detected, using 120 BPM")
            return 120.0
        
        # Calculate intervals between peaks
        intervals = np.diff(peaks) * hop_length / sr
        
        # Filter out unreasonable intervals
        intervals = intervals[(intervals > 0.2) & (intervals < 2.0)]
        
        if len(intervals) < 2:
            return 120.0
        
        # Calculate BPM from median interval
        median_interval = np.median(intervals)
        bpm = 60.0 / median_interval
        
        # Normalize to reasonable range (70-180 BPM)
        while bpm < 70:
            bpm *= 2
        while bpm > 180:
            bpm /= 2
        
        self.log(f"[BPM] ✓ Detected: {bpm:.1f} BPM")
        return bpm
    
    def time_stretch(self, y, sr, target_bpm, source_bpm):
        """
        Time stretch audio to match target BPM - NO LIBROSA!
        Uses scipy resampling.
        """
        if abs(target_bpm - source_bpm) < 2:
            return y  # Close enough
        
        rate = source_bpm / target_bpm
        self.log(f"[STRETCH] {source_bpm:.0f} → {target_bpm:.0f} BPM (rate: {rate:.3f})")
        
        if HAS_SP:
            # Calculate new length
            new_length = int(len(y) / rate)
            # Use scipy resample for high quality
            stretched = signal.resample(y, new_length)
            return stretched.astype(np.float32)
        else:
            # Simple linear interpolation fallback
            new_length = int(len(y) / rate)
            indices = np.linspace(0, len(y) - 1, new_length)
            return np.interp(indices, np.arange(len(y)), y).astype(np.float32)
    
    def pitch_shift(self, y, sr, semitones):
        """
        Pitch shift audio - NO LIBROSA!
        Uses resample + time stretch method.
        """
        if semitones == 0:
            return y
        
        self.log(f"[PITCH] Shifting {semitones:+d} semitones")
        
        # Pitch shift factor
        factor = 2 ** (semitones / 12.0)
        
        if HAS_SP:
            # Resample to change pitch
            shifted_length = int(len(y) / factor)
            shifted = signal.resample(y, shifted_length)
            
            # Resample back to original length to preserve duration
            result = signal.resample(shifted, len(y))
            return result.astype(np.float32)
        else:
            # Simple interpolation fallback
            shifted_length = int(len(y) / factor)
            indices = np.linspace(0, len(y) - 1, shifted_length)
            shifted = np.interp(indices, np.arange(len(y)), y)
            
            # Resample back
            indices2 = np.linspace(0, len(shifted) - 1, len(y))
            result = np.interp(indices2, np.arange(len(shifted)), shifted)
            return result.astype(np.float32)
    
    def normalize(self, y, target_db=-3.0):
        """Normalize audio to target dB level"""
        peak = np.max(np.abs(y))
        if peak == 0:
            return y
        target_amp = 10 ** (target_db / 20.0)
        return y * (target_amp / peak)
    
    def eq_3band(self, y, sr, low_gain=1.0, mid_gain=1.0, high_gain=1.0):
        """
        3-band equalizer - NO LIBROSA!
        Uses scipy butterworth filters.
        """
        if not HAS_SP:
            return y * ((low_gain + mid_gain + high_gain) / 3)
        
        nyq = sr / 2
        
        # Design filters
        # Low: 0-200 Hz
        b_low, a_low = signal.butter(4, 200 / nyq, btype='low')
        # Mid: 200-2000 Hz  
        b_mid, a_mid = signal.butter(4, [200 / nyq, 2000 / nyq], btype='band')
        # High: 2000+ Hz
        b_high, a_high = signal.butter(4, 2000 / nyq, btype='high')
        
        # Apply filters
        low = signal.filtfilt(b_low, a_low, y) * low_gain
        mid = signal.filtfilt(b_mid, a_mid, y) * mid_gain
        high = signal.filtfilt(b_high, a_high, y) * high_gain
        
        return (low + mid + high).astype(np.float32)
    
    def fade(self, y, fade_in_sec=0.1, fade_out_sec=2.0):
        """Apply fade in/out envelope"""
        envelope = np.ones(len(y))
        
        fade_in_samples = int(SR * fade_in_sec)
        fade_out_samples = int(SR * fade_out_sec)
        
        if fade_in_samples > 0 and fade_in_samples < len(y):
            envelope[:fade_in_samples] = np.linspace(0, 1, fade_in_samples)
        
        if fade_out_samples > 0 and fade_out_samples < len(y):
            envelope[-fade_out_samples:] = np.linspace(1, 0, fade_out_samples)
        
        return y * envelope
    
    # ─── MIXING MODES ───
    
    def mix_blend(self, y1, y2, ratio=0.5):
        """Simple blend of two tracks"""
        return (y1 * ratio + y2 * (1 - ratio)).astype(np.float32)
    
    def mix_alternate(self, y1, y2, sr, bpm, bars=4):
        """Alternate between tracks every N bars"""
        samples_per_beat = int((60 / bpm) * sr)
        samples_per_bar = samples_per_beat * 4
        samples_per_switch = samples_per_bar * bars
        
        result = np.zeros(len(y1), dtype=np.float32)
        use_track1 = True
        
        for i in range(0, len(y1), samples_per_switch):
            end = min(i + samples_per_switch, len(y1))
            
            if use_track1:
                result[i:end] = y1[i:end]
            else:
                result[i:end] = y2[i:end]
            
            # Crossfade at switch points
            crossfade_len = min(int(sr * 0.25), samples_per_switch // 4)
            if i > 0 and crossfade_len > 0:
                fade = np.linspace(0, 1, crossfade_len)
                start = max(0, i - crossfade_len // 2)
                end_fade = min(len(y1), i + crossfade_len // 2)
                actual_len = end_fade - start
                
                if actual_len > 0:
                    fade_slice = fade[:actual_len] if use_track1 else (1 - fade[:actual_len])
                    if use_track1:
                        result[start:end_fade] = y2[start:end_fade] * (1 - fade_slice) + y1[start:end_fade] * fade_slice
            
            use_track1 = not use_track1
        
        return result
    
    def mix_layer(self, y1, y2, sr):
        """Layer tracks with EQ separation"""
        # Track 1: Keep bass/low-mids
        y1_eq = self.eq_3band(y1, sr, low_gain=1.2, mid_gain=0.7, high_gain=0.4)
        # Track 2: Keep highs/high-mids
        y2_eq = self.eq_3band(y2, sr, low_gain=0.4, mid_gain=0.7, high_gain=1.2)
        
        return (y1_eq * 0.6 + y2_eq * 0.6).astype(np.float32)
    
    def mix_sidechain(self, y1, y2, sr, bpm):
        """Sidechain compression effect (pumping)"""
        beat_samples = int((60 / bpm) * sr)
        num_beats = len(y1) // beat_samples + 1
        
        envelope = np.ones(len(y1), dtype=np.float32)
        
        for i in range(num_beats):
            start = i * beat_samples
            duck_length = min(int(beat_samples * 0.3), len(y1) - start)
            
            if duck_length > 0:
                # Create duck envelope (quick drop, slow release)
                duck = np.concatenate([
                    np.linspace(1.0, 0.3, duck_length // 2),
                    np.linspace(0.3, 1.0, duck_length - duck_length // 2)
                ])
                end = min(start + duck_length, len(y1))
                envelope[start:end] = duck[:end - start]
        
        return (y1 * 0.7 + y2 * envelope * 0.5).astype(np.float32)
    
    # ─── EXPORT ───
    
    def export(self, y, sr, output_path):
        """Export audio to file"""
        self.log(f"[EXPORT] Saving: {os.path.basename(output_path)}")
        
        ext = os.path.splitext(output_path)[1].lower()
        wav_path = output_path if ext == '.wav' else output_path.replace(ext, '.wav')
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        # Save WAV first
        if HAS_SF:
            sf.write(wav_path, y, sr)
        elif HAS_SP:
            y_int = (y * 32767).astype(np.int16)
            wavfile.write(wav_path, sr, y_int)
        else:
            # Manual WAV export
            with wave.open(wav_path, 'w') as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                for sample in y:
                    clamped = max(-1.0, min(1.0, sample))
                    packed = struct.pack('<h', int(clamped * 32767))
                    w.writeframesraw(packed)
        
        # Convert to MP3 if needed
        if ext == '.mp3':
            try:
                if HAS_PD:
                    audio = AudioSegment.from_wav(wav_path)
                    audio.export(output_path, format='mp3', bitrate='320k')
                    os.remove(wav_path)
                    self.log("[EXPORT] ✓ Saved as MP3 (320kbps)")
                else:
                    subprocess.run(['ffmpeg', '-y', '-i', wav_path, '-b:a', '320k', output_path],
                                  capture_output=True, check=True)
                    os.remove(wav_path)
                    self.log("[EXPORT] ✓ Saved as MP3 (ffmpeg)")
            except Exception as e:
                self.log(f"[EXPORT] ⚠ Keeping as WAV: {e}")
        
        # Convert to MP4 if needed
        elif ext == '.mp4':
            try:
                duration = len(y) / sr
                subprocess.run([
                    'ffmpeg', '-y',
                    '-f', 'lavfi', '-i', f'color=c=0x2d2d44:s=1920x1080:d={duration}',
                    '-i', wav_path,
                    '-c:v', 'libx264', '-preset', 'fast',
                    '-c:a', 'aac', '-b:a', '320k',
                    '-shortest', output_path
                ], capture_output=True, check=True)
                os.remove(wav_path)
                self.log("[EXPORT] ✓ Saved as MP4")
            except Exception as e:
                self.log(f"[EXPORT] ⚠ Keeping as WAV (ffmpeg needed for MP4)")
                final_wav = output_path.replace('.mp4', '.wav')
                if wav_path != final_wav:
                    os.rename(wav_path, final_wav)
        
        return output_path
    
    # ─── MAIN RIP FUNCTION ───
    
    def create_rip(self, url1, url2, output, mode='blend', target_bpm=None, 
                   pitch1=0, pitch2=0, eq1=(1,1,1), eq2=(1,1,1)):
        """Create a High Quality Rip™"""
        
        self.log("━" * 55)
        self.log("  HIGH QUALITY RIP ENGINE (No librosa needed!)")
        self.log("━" * 55)
        self.prog(5)
        
        try:
            # Step 1: Download tracks
            self.log("\n[1/5] Downloading tracks...")
            self.prog(10)
            track1_path = self.download(url1, "track1")
            self.prog(25)
            track2_path = self.download(url2, "track2")
            self.prog(40)
            
            # Step 2: Load audio
            self.log("\n[2/5] Loading audio...")
            y1, sr = self.load(track1_path)
            y2, _ = self.load(track2_path)
            self.prog(50)
            
            # Step 3: Detect BPM
            self.log("\n[3/5] Analyzing tempo...")
            bpm1 = self.detect_bpm(y1, sr)
            bpm2 = self.detect_bpm(y2, sr)
            self.log(f"  Track 1: {bpm1:.0f} BPM")
            self.log(f"  Track 2: {bpm2:.0f} BPM")
            
            if target_bpm is None:
                target_bpm = round((bpm1 + bpm2) / 2)
            self.log(f"  Target:  {target_bpm} BPM")
            self.prog(55)
            
            # Step 4: Process audio
            self.log("\n[4/5] Processing audio...")
            
            # Time stretch to match BPM
            y1 = self.time_stretch(y1, sr, target_bpm, bpm1)
            y2 = self.time_stretch(y2, sr, target_bpm, bpm2)
            
            # Pitch shift
            if pitch1 != 0:
                y1 = self.pitch_shift(y1, sr, pitch1)
            if pitch2 != 0:
                y2 = self.pitch_shift(y2, sr, pitch2)
            
            # Match lengths (loop shorter track)
            max_len = max(len(y1), len(y2))
            if len(y1) < max_len:
                repeats = int(np.ceil(max_len / len(y1)))
                y1 = np.tile(y1, repeats)[:max_len]
            if len(y2) < max_len:
                repeats = int(np.ceil(max_len / len(y2)))
                y2 = np.tile(y2, repeats)[:max_len]
            
            # Normalize
            y1 = self.normalize(y1, -6.0)
            y2 = self.normalize(y2, -6.0)
            
            # Apply EQ
            y1 = self.eq_3band(y1, sr, *eq1)
            y2 = self.eq_3band(y2, sr, *eq2)
            self.prog(70)
            
            # Step 5: Mix
            self.log(f"\n[5/5] Mixing ({mode})...")
            
            if mode == 'blend':
                rip = self.mix_blend(y1, y2, 0.5)
            elif mode == 'alternate':
                rip = self.mix_alternate(y1, y2, sr, target_bpm, 4)
            elif mode == 'layer':
                rip = self.mix_layer(y1, y2, sr)
            elif mode == 'sidechain':
                rip = self.mix_sidechain(y1, y2, sr, target_bpm)
            else:
                rip = self.mix_blend(y1, y2, 0.5)
            
            # Final processing
            rip = self.normalize(rip, -1.0)
            rip = self.fade(rip, 0.1, 2.0)
            self.prog(90)
            
            # Export
            self.export(rip, sr, output)
            self.prog(100)
            
            self.log("")
            self.log("━" * 55)
            self.log("  ✓ HIGH QUALITY RIP COMPLETE!")
            self.log(f"  {output}")
            self.log(f"  \"{random.choice(QUOTES)}\"")
            self.log("━" * 55)
            
            return output
            
        finally:
            self.cleanup()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DAW TRACK WIDGET
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DAWTrack(tk.Frame):
    """Single track in the DAW with controls"""
    
    def __init__(self, parent, name, color, **kw):
        super().__init__(parent, bg='#1e1e2e', **kw)
        self.name = name
        self.color = color
        
        # Header
        hdr = tk.Frame(self, bg='#2d2d44', height=28)
        hdr.pack(fill='x')
        hdr.pack_propagate(False)
        
        tk.Label(hdr, text=name, font=('Consolas', 10, 'bold'), 
                bg='#2d2d44', fg=color).pack(side='left', padx=5)
        
        self.bpm_lbl = tk.Label(hdr, text="-- BPM", font=('Consolas', 9), 
                               bg='#2d2d44', fg='#888')
        self.bpm_lbl.pack(side='right', padx=10)
        
        # URL Entry
        url_frame = tk.Frame(self, bg='#1e1e2e')
        url_frame.pack(fill='x', padx=5, pady=3)
        
        self.url_entry = tk.Entry(url_frame, bg='#0d0d1a', fg='#fff',
                                  insertbackground='#fff', font=('Consolas', 9))
        self.url_entry.pack(side='left', fill='x', expand=True)
        
        tk.Button(url_frame, text="📁", bg='#3d3d5c', fg='#fff', width=3,
                 command=self.browse).pack(side='left', padx=2)
        tk.Button(url_frame, text="📋", bg='#3d3d5c', fg='#fff', width=3,
                 command=self.paste).pack(side='left')
        
        # Waveform canvas
        self.canvas = Canvas(self, bg='#0d0d1a', height=50, highlightthickness=0)
        self.canvas.pack(fill='x', padx=5, pady=3)
        
        # Controls
        ctrl = tk.Frame(self, bg='#1e1e2e')
        ctrl.pack(fill='x', padx=5, pady=3)
        
        # Volume
        tk.Label(ctrl, text="Vol", bg='#1e1e2e', fg='#888', 
                font=('Consolas', 8)).pack(side='left')
        self.vol = tk.Scale(ctrl, from_=0, to=100, orient='horizontal',
                           bg='#2d2d44', fg='#fff', highlightthickness=0,
                           length=70, showvalue=False)
        self.vol.set(80)
        self.vol.pack(side='left', padx=2)
        
        # Pitch
        tk.Label(ctrl, text="Pitch", bg='#1e1e2e', fg='#888',
                font=('Consolas', 8)).pack(side='left', padx=(8, 0))
        self.pitch = tk.Spinbox(ctrl, from_=-12, to=12, width=4,
                               bg='#0d0d1a', fg='#fff')
        self.pitch.delete(0, 'end')
        self.pitch.insert(0, "0")
        self.pitch.pack(side='left', padx=2)
        
        # EQ
        tk.Label(ctrl, text="EQ:", bg='#1e1e2e', fg='#888',
                font=('Consolas', 8)).pack(side='left', padx=(12, 2))
        
        self.eq_low = tk.Scale(ctrl, from_=200, to=0, orient='vertical',
                              bg='#2d2d44', fg='#fff', highlightthickness=0,
                              length=35, width=12, showvalue=False)
        self.eq_low.set(100)
        self.eq_low.pack(side='left', padx=1)
        
        self.eq_mid = tk.Scale(ctrl, from_=200, to=0, orient='vertical',
                              bg='#2d2d44', fg='#fff', highlightthickness=0,
                              length=35, width=12, showvalue=False)
        self.eq_mid.set(100)
        self.eq_mid.pack(side='left', padx=1)
        
        self.eq_high = tk.Scale(ctrl, from_=200, to=0, orient='vertical',
                               bg='#2d2d44', fg='#fff', highlightthickness=0,
                               length=35, width=12, showvalue=False)
        self.eq_high.set(100)
        self.eq_high.pack(side='left', padx=1)
        
        tk.Label(ctrl, text="L M H", bg='#1e1e2e', fg='#555',
                font=('Consolas', 7)).pack(side='left')
    
    def browse(self):
        path = filedialog.askopenfilename(
            filetypes=[("Audio", "*.mp3 *.wav *.flac *.m4a *.ogg"), ("All", "*.*")]
        )
        if path:
            self.url_entry.delete(0, 'end')
            self.url_entry.insert(0, path)
    
    def paste(self):
        try:
            text = self.winfo_toplevel().clipboard_get()
            self.url_entry.delete(0, 'end')
            self.url_entry.insert(0, text.strip())
        except:
            pass
    
    def get_url(self):
        return self.url_entry.get().strip()
    
    def get_pitch(self):
        try:
            return int(self.pitch.get())
        except:
            return 0
    
    def get_eq(self):
        return (self.eq_low.get() / 100, 
                self.eq_mid.get() / 100, 
                self.eq_high.get() / 100)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN GUI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CatsHQRipper:
    """Cat's HQRIPPER 0.1 - Main Application"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(APP)
        self.root.geometry("680x800")
        self.root.configure(bg='#f0f0f0')
        self.root.resizable(False, False)
        
        self.processing = False
        
        self.build_menu()
        self.build_notebook()
        self.build_upload_tab()
        self.build_daw_tab()
        self.build_console_tab()
        self.build_settings_tab()
        self.build_statusbar()
    
    def build_menu(self):
        menu = tk.Frame(self.root, bg='#e0e0e0', height=28)
        menu.pack(fill='x')
        menu.pack_propagate(False)
        
        btns = [("🔴 Upload", 0), ("🎛 DAW", 1), ("📥 Console", 2), ("⚙ Settings", 3)]
        for txt, idx in btns:
            tk.Button(menu, text=txt, relief='flat', bg='#e0e0e0',
                     font=('Segoe UI', 9), padx=8,
                     command=lambda i=idx: self.nb.select(i)).pack(side='left', padx=1, pady=2)
    
    def build_notebook(self):
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill='both', expand=True, padx=5, pady=5)
    
    def build_upload_tab(self):
        tab = ttk.Frame(self.nb, padding=8)
        self.nb.add(tab, text="Upload")
        
        # File Downloader
        dl = ttk.LabelFrame(tab, text="File Downloader", padding=5)
        dl.pack(fill='x', pady=(0, 8))
        
        for lbl, attr in [("Logo link", "logo_e"), ("Audio/Video", "video_e")]:
            r = ttk.Frame(dl)
            r.pack(fill='x', pady=2)
            ttk.Label(r, text=lbl, width=11).pack(side='left')
            e = ttk.Entry(r, width=52)
            e.pack(side='left', padx=3)
            setattr(self, attr, e)
        
        r = ttk.Frame(dl)
        r.pack(fill='x', pady=2)
        ttk.Label(r, text="Title:", width=11).pack(side='left')
        self.title_e = ttk.Entry(r, width=38)
        self.title_e.pack(side='left', padx=3)
        self.title_e.insert(0, "Title Theme - 7 GRAND DAD")
        ttk.Button(r, text="📋 Parse", width=10, command=self.parse_clip).pack(side='left', padx=3)
        
        # Quick Mashup
        qm = ttk.LabelFrame(tab, text="🎵 Quick Mashup", padding=5)
        qm.pack(fill='x', pady=(0, 8))
        
        for lbl, attr in [("Track 1:", "t1_e"), ("Track 2:", "t2_e")]:
            r = ttk.Frame(qm)
            r.pack(fill='x', pady=2)
            ttk.Label(r, text=lbl, width=10).pack(side='left')
            e = ttk.Entry(r, width=48)
            e.pack(side='left', padx=3)
            setattr(self, attr, e)
            ttk.Button(r, text="📁", width=3, command=lambda en=e: self.browse(en)).pack(side='left')
        
        mr = ttk.Frame(qm)
        mr.pack(fill='x', pady=5)
        ttk.Label(mr, text="Mode:", width=10).pack(side='left')
        self.mode_v = tk.StringVar(value='blend')
        for txt, val in [('🔀 Blend', 'blend'), ('🔄 Alternate', 'alternate'),
                        ('📊 Layer', 'layer'), ('💓 Sidechain', 'sidechain')]:
            ttk.Radiobutton(mr, text=txt, value=val, variable=self.mode_v).pack(side='left', padx=5)
        
        # Description + Scheduler
        ds = ttk.Frame(tab)
        ds.pack(fill='x', pady=(0, 8))
        
        self.desc = scrolledtext.ScrolledText(ds, width=35, height=5, font=('Segoe UI', 9))
        self.desc.pack(side='left', fill='both', expand=True)
        self.desc.insert('1.0', "High quality rip.\n\nPlease read the channel description.")
        
        sf = ttk.LabelFrame(ds, text="Scheduler", padding=5)
        sf.pack(side='right', fill='y', padx=(8, 0))
        
        now = datetime.now()
        self.sched = {}
        for l, v, rng in [("Year", now.year, (2020, 2030)), ("Month", now.month, (1, 12)),
                         ("Day", now.day, (1, 31)), ("Hour", now.hour, (0, 23)), ("Min", now.minute, (0, 59))]:
            r = ttk.Frame(sf)
            r.pack(fill='x')
            ttk.Label(r, text=l, width=5).pack(side='left')
            s = ttk.Spinbox(r, from_=rng[0], to=rng[1], width=5)
            s.set(v)
            s.pack(side='left')
            self.sched[l] = s
        
        ttk.Button(sf, text="⏱ Now", command=self.set_now).pack(pady=3)
        
        # Tags
        tr = ttk.Frame(tab)
        tr.pack(fill='x', pady=(0, 8))
        ttk.Label(tr, text="Tags:").pack(side='left')
        self.tags_e = ttk.Entry(tr, width=55)
        self.tags_e.pack(side='left', padx=5)
        
        # Playlist
        pl = ttk.LabelFrame(tab, text="Playlist", padding=5)
        pl.pack(fill='x', pady=(0, 8))
        
        pr = ttk.Frame(pl)
        pr.pack(fill='x', pady=2)
        ttk.Label(pr, text="Playlist:").pack(side='left')
        self.pl_e = ttk.Entry(pr, width=40)
        self.pl_e.pack(side='left', padx=5)
        
        pr2 = ttk.Frame(pl)
        pr2.pack(fill='x', pady=2)
        self.game_e = ttk.Entry(pr2, width=25)
        self.game_e.pack(side='left', padx=5)
        self.game_e.insert(0, "7 GRAND DAD")
        ttk.Button(pr2, text="Autocomplete", command=lambda: self.pl_e.insert(0, self.game_e.get())).pack(side='left')
        
        # Upload buttons
        uf = ttk.LabelFrame(tab, text="Uploading", padding=5)
        uf.pack(fill='x', pady=(0, 8))
        
        ub = ttk.Frame(uf)
        ub.pack(fill='x', pady=3)
        tk.Button(ub, text="🔴 Upload & render", bg='#ff4444', fg='white',
                 font=('Segoe UI', 9, 'bold'), padx=6).pack(side='left', padx=3)
        ttk.Button(ub, text="Upload").pack(side='left', padx=3)
        tk.Button(ub, text="▶ External", bg='#333', fg='white', padx=6).pack(side='right', padx=3)
        ttk.Button(ub, text="Thumbnail").pack(pady=3)
        
        # Create Rip button
        rf = ttk.Frame(tab)
        rf.pack(fill='x', pady=8)
        
        self.rip_btn = tk.Button(rf, text="🎮 CREATE HIGH QUALITY RIP™",
                                font=('Segoe UI', 12, 'bold'), bg='#4a86c7', fg='white',
                                padx=15, pady=8, command=self.quick_rip)
        self.rip_btn.pack(fill='x', padx=15)
        
        self.prog_v = tk.DoubleVar()
        ttk.Progressbar(rf, variable=self.prog_v, maximum=100).pack(fill='x', padx=15, pady=5)
        
        ttk.Label(rf, text='Output: HIGH_QUALITY_RIP.mp4', font=('Segoe UI', 8),
                 foreground='gray').pack()
    
    def build_daw_tab(self):
        tab = tk.Frame(self.nb, bg='#1e1e2e')
        self.nb.add(tab, text="🎛 DAW")
        
        # Header
        hdr = tk.Frame(tab, bg='#0d0d1a', height=40)
        hdr.pack(fill='x')
        hdr.pack_propagate(False)
        
        tk.Label(hdr, text="🎛 CAT'S DAW - High Quality Mixing",
                font=('Consolas', 12, 'bold'), bg='#0d0d1a', fg='#00ff88').pack(side='left', padx=10, pady=8)
        
        # BPM
        tr = tk.Frame(hdr, bg='#0d0d1a')
        tr.pack(side='right', padx=10)
        tk.Label(tr, text="BPM:", bg='#0d0d1a', fg='#888', font=('Consolas', 9)).pack(side='left')
        self.bpm_e = tk.Entry(tr, width=5, bg='#1e1e2e', fg='#fff', font=('Consolas', 10))
        self.bpm_e.insert(0, "auto")
        self.bpm_e.pack(side='left', padx=5)
        
        # Tracks
        tf = tk.Frame(tab, bg='#1e1e2e')
        tf.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.daw_t1 = DAWTrack(tf, "TRACK 1 (Source)", "#ff6b6b")
        self.daw_t1.pack(fill='x', pady=(0, 10))
        
        self.daw_t2 = DAWTrack(tf, "TRACK 2 (Mashup)", "#4ecdc4")
        self.daw_t2.pack(fill='x', pady=(0, 10))
        
        # Master
        mf = tk.Frame(tab, bg='#2d2d44')
        mf.pack(fill='x', padx=10, pady=(0, 10))
        
        tk.Label(mf, text="MASTER", font=('Consolas', 10, 'bold'), bg='#2d2d44', fg='#fff').pack(pady=5)
        
        mc = tk.Frame(mf, bg='#2d2d44')
        mc.pack(pady=5)
        tk.Label(mc, text="Mode:", bg='#2d2d44', fg='#888', font=('Consolas', 9)).pack(side='left', padx=5)
        
        self.daw_mode = tk.StringVar(value='blend')
        for txt, val in [('Blend', 'blend'), ('Alternate', 'alternate'),
                        ('Layer', 'layer'), ('Sidechain', 'sidechain')]:
            tk.Radiobutton(mc, text=txt, value=val, variable=self.daw_mode,
                          bg='#2d2d44', fg='#fff', selectcolor='#4a4a6a',
                          font=('Consolas', 9)).pack(side='left', padx=6)
        
        # Output
        of = tk.Frame(mf, bg='#2d2d44')
        of.pack(pady=5)
        tk.Label(of, text="Output:", bg='#2d2d44', fg='#888', font=('Consolas', 9)).pack(side='left', padx=5)
        self.daw_out = tk.Entry(of, width=30, bg='#1e1e2e', fg='#fff', font=('Consolas', 9))
        self.daw_out.insert(0, "HIGH_QUALITY_RIP")
        self.daw_out.pack(side='left', padx=5)
        self.daw_fmt = ttk.Combobox(of, values=['.mp4', '.mp3', '.wav'], width=6)
        self.daw_fmt.set('.mp4')
        self.daw_fmt.pack(side='left', padx=5)
        
        # Render button
        self.daw_btn = tk.Button(tab, text="🔥 RENDER HIGH QUALITY RIP",
                                font=('Consolas', 14, 'bold'), bg='#9933ff', fg='white',
                                padx=20, pady=12, command=self.daw_render)
        self.daw_btn.pack(fill='x', padx=10, pady=(0, 10))
        
        self.daw_prog_v = tk.DoubleVar()
        ttk.Progressbar(tab, variable=self.daw_prog_v, maximum=100).pack(fill='x', padx=10, pady=(0, 5))
        
        tk.Label(tab, text="Ready - No librosa needed!", bg='#1e1e2e', fg='#888',
                font=('Consolas', 9)).pack(pady=(0, 10))
    
    def build_console_tab(self):
        tab = ttk.Frame(self.nb, padding=10)
        self.nb.add(tab, text="Console")
        
        self.console = scrolledtext.ScrolledText(tab, width=70, height=32,
                                                 font=('Consolas', 9),
                                                 bg='#0d0d1a', fg='#00ff88')
        self.console.pack(fill='both', expand=True)
        
        self.log("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
        self.log("┃  CAT'S HQRIPPER 0.1 - Console                 ┃")
        self.log("┃  NO LIBROSA NEEDED! (No LLVM hassle!)         ┃")
        self.log("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
        self.log("")
        self.log(f"[SYS] NumPy:     {'✓' if HAS_NP else '✗'}")
        self.log(f"[SYS] SciPy:     {'✓' if HAS_SP else '✗'}")
        self.log(f"[SYS] PyDub:     {'✓' if HAS_PD else '✗'}")
        self.log(f"[SYS] SoundFile: {'✓' if HAS_SF else '✗'}")
        self.log(f"[SYS] yt-dlp:    {'✓' if HAS_YT else '✗'}")
        self.log("")
        self.log(f"[READY] {random.choice(QUOTES)}")
    
    def build_settings_tab(self):
        tab = ttk.Frame(self.nb, padding=10)
        self.nb.add(tab, text="Settings")
        
        # Output
        of = ttk.LabelFrame(tab, text="Output", padding=10)
        of.pack(fill='x', pady=5)
        
        or1 = ttk.Frame(of)
        or1.pack(fill='x', pady=3)
        ttk.Label(or1, text="Directory:").pack(side='left')
        self.out_dir = ttk.Entry(or1, width=40)
        self.out_dir.pack(side='left', padx=5)
        self.out_dir.insert(0, os.getcwd())
        ttk.Button(or1, text="Browse", command=self.browse_dir).pack(side='left')
        
        # Quality
        qf = ttk.LabelFrame(tab, text="Quality", padding=10)
        qf.pack(fill='x', pady=5)
        
        ttk.Label(qf, text="Format:").pack(anchor='w')
        self.fmt_cb = ttk.Combobox(qf, values=['.mp4', '.mp3', '.wav'], width=10)
        self.fmt_cb.set('.mp4')
        self.fmt_cb.pack(anchor='w', pady=3)
        
        # About
        af = ttk.LabelFrame(tab, text="About", padding=10)
        af.pack(fill='x', pady=5)
        
        about = f"""{APP}
SiIvaGunner High Quality Rip Generator
Team Flames / Samsoft / Flames Co.

NO LIBROSA REQUIRED!
Uses scipy + pydub for all audio processing.
No LLVM installation needed!

"Please read the channel description."
"""
        ttk.Label(af, text=about, font=('Segoe UI', 9), justify='left').pack(anchor='w')
    
    def build_statusbar(self):
        ttk.Label(self.root, text="Ready - No librosa needed!", relief='sunken', anchor='w').pack(fill='x', side='bottom')
    
    # ─── HELPERS ───
    
    def log(self, msg):
        self.console.insert('end', msg + '\n')
        self.console.see('end')
        self.root.update_idletasks()
    
    def set_prog(self, v):
        self.prog_v.set(v)
        self.daw_prog_v.set(v)
        self.root.update_idletasks()
    
    def parse_clip(self):
        try:
            t = self.root.clipboard_get()
            if 'youtube' in t.lower() or 'soundcloud' in t.lower():
                self.video_e.delete(0, 'end')
                self.video_e.insert(0, t.strip())
        except:
            pass
    
    def browse(self, e):
        p = filedialog.askopenfilename(filetypes=[("Audio", "*.mp3 *.wav *.flac *.m4a"), ("All", "*.*")])
        if p:
            e.delete(0, 'end')
            e.insert(0, p)
    
    def browse_dir(self):
        p = filedialog.askdirectory()
        if p:
            self.out_dir.delete(0, 'end')
            self.out_dir.insert(0, p)
    
    def set_now(self):
        now = datetime.now()
        self.sched['Year'].set(now.year)
        self.sched['Month'].set(now.month)
        self.sched['Day'].set(now.day)
        self.sched['Hour'].set(now.hour)
        self.sched['Min'].set(now.minute)
    
    # ─── RIP FUNCTIONS ───
    
    def quick_rip(self):
        if self.processing:
            messagebox.showwarning("Busy", "Already processing!")
            return
        
        t1 = self.t1_e.get().strip()
        t2 = self.t2_e.get().strip()
        
        if not t1 or not t2:
            messagebox.showerror("Error", "Enter both Track 1 and Track 2!")
            return
        
        self.processing = True
        self.rip_btn.config(state='disabled', text="⏳ Processing...")
        self.nb.select(2)  # Console
        
        out = os.path.join(self.out_dir.get(), "HIGH_QUALITY_RIP" + self.fmt_cb.get())
        mode = self.mode_v.get()
        
        thread = threading.Thread(target=self._do_rip,
                                 args=(t1, t2, out, mode, None, 0, 0, (1,1,1), (1,1,1)),
                                 daemon=True)
        thread.start()
    
    def daw_render(self):
        if self.processing:
            messagebox.showwarning("Busy", "Already processing!")
            return
        
        t1 = self.daw_t1.get_url()
        t2 = self.daw_t2.get_url()
        
        if not t1 or not t2:
            messagebox.showerror("Error", "Enter URLs for both tracks!")
            return
        
        self.processing = True
        self.daw_btn.config(state='disabled', text="⏳ Rendering...")
        self.nb.select(2)  # Console
        
        out = os.path.join(self.out_dir.get(), self.daw_out.get() + self.daw_fmt.get())
        
        bpm_txt = self.bpm_e.get().strip().lower()
        bpm = None if bpm_txt == 'auto' else int(bpm_txt)
        
        mode = self.daw_mode.get()
        p1 = self.daw_t1.get_pitch()
        p2 = self.daw_t2.get_pitch()
        eq1 = self.daw_t1.get_eq()
        eq2 = self.daw_t2.get_eq()
        
        thread = threading.Thread(target=self._do_rip,
                                 args=(t1, t2, out, mode, bpm, p1, p2, eq1, eq2),
                                 daemon=True)
        thread.start()
    
    def _do_rip(self, t1, t2, out, mode, bpm, p1, p2, eq1, eq2):
        try:
            engine = AudioEngine(self.log, self.set_prog)
            engine.create_rip(t1, t2, out, mode, bpm, p1, p2, eq1, eq2)
            
            self.root.after(0, lambda: messagebox.showinfo(
                "✓ High Quality Rip Complete!",
                f"Your rip is ready!\n\n{out}\n\n\"{random.choice(QUOTES)}\""
            ))
        except Exception as e:
            self.log(f"[ERROR] {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.processing = False
            self.root.after(0, lambda: self.rip_btn.config(state='normal', text="🎮 CREATE HIGH QUALITY RIP™"))
            self.root.after(0, lambda: self.daw_btn.config(state='normal', text="🔥 RENDER HIGH QUALITY RIP"))
            self.root.after(0, lambda: self.set_prog(0))
    
    def run(self):
        self.root.mainloop()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    CatsHQRipper().run()
