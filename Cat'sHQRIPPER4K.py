#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║  CAT'S EZGUNNER HQ RIPPER 7.2 — SiIvaGunner Style Mashup Tool                        ║
║  Rave.DJ Unlimited Remix → Premiere Pro Ready                                         ║
║  Team Flames / Samsoft / Flames Co.                                                   ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
"""

import subprocess
import sys
import os

# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-INSTALL DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════════════

def install_package(package):
    """Install a package via pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

def ensure_dependencies():
    """Auto-install all required packages."""
    packages = {
        'requests': 'requests',
        'selenium': 'selenium',
        'webdriver_manager': 'webdriver-manager',
    }
    
    for module, pip_name in packages.items():
        try:
            __import__(module)
        except ImportError:
            print(f"[HQ RIPPER 7.2] Installing {pip_name}...")
            try:
                install_package(pip_name)
                print(f"[HQ RIPPER 7.2] ✓ {pip_name} installed!")
            except Exception as e:
                print(f"[HQ RIPPER 7.2] ⚠ Could not install {pip_name}: {e}")

print("╔════════════════════════════════════════════╗")
print("║  CAT'S EZGUNNER HQ RIPPER 7.2              ║")
print("║  Checking dependencies...                  ║")
print("╚════════════════════════════════════════════╝")
ensure_dependencies()

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import re
import time
import threading
import json
from datetime import datetime, timedelta
import wave
import struct
import math

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from selenium import webdriver
    from selenium.common.exceptions import TimeoutException
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.expected_conditions import url_changes
    from selenium.webdriver.chrome.service import Service
    HAS_SELENIUM = True
except ImportError:
    HAS_SELENIUM = False

try:
    from webdriver_manager.chrome import ChromeDriverManager
    HAS_WEBDRIVER_MANAGER = True
except ImportError:
    HAS_WEBDRIVER_MANAGER = False

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

APP_NAME = "Cat's EZGUNNER HQ Ripper 7.2"
VERSION = "7.2.0"
ATTEMPTS = 3
SHORT_TIME_OUT = 30
TIME_OUT = 720
POLL_FREQ = 5

# ═══════════════════════════════════════════════════════════════════════════════
# RAVE.DJ ENGINE (ALL BUGS FIXED)
# ═══════════════════════════════════════════════════════════════════════════════

class RaveDJEngine:
    """Core Rave.DJ mashup engine - all original bugs fixed."""
    
    def __init__(self, log_callback=None, progress_callback=None):
        self.driver = None
        self.log = log_callback or print
        self.progress = progress_callback or (lambda x: None)
    
    @staticmethod
    def is_valid_youtube_url(url):
        """Validate YouTube URLs."""
        patterns = [
            re.compile(r'^(https?://)?(www\.)?(youtube\.com)/watch\?v=[\w-]{11}'),
            re.compile(r'^(https?://)?(www\.)?youtu\.be/[\w-]{11}'),
            re.compile(r'^(https?://)?(www\.)?(youtube\.com)/shorts/[\w-]{11}'),
        ]
        return any(p.match(url) for p in patterns)
    
    @staticmethod
    def is_valid_spotify_url(url):
        """Validate Spotify URLs."""
        pattern = re.compile(r'^https://open\.spotify\.com/(track|album|playlist)/[a-zA-Z0-9]{22}')
        return bool(pattern.match(url))
    
    @staticmethod
    def verify_link(url):
        """Check if URL is valid for Rave.DJ."""
        return RaveDJEngine.is_valid_spotify_url(url) or RaveDJEngine.is_valid_youtube_url(url)
    
    @staticmethod
    def clean_url(url):
        """Clean and normalize URL - FIXED regex syntax error from original."""
        url = url.strip()
        if "youtube" in url.lower():
            # FIXED: Original had broken regex: r[](https://...
            match = re.match(r'(https?://)?(www\.)?(youtube\.com/watch\?v=[\w-]{11})', url)
            if match:
                return f"https://www.{match.group(3)}"
            # Handle youtu.be short links
            match = re.match(r'https?://youtu\.be/([\w-]{11})', url)
            if match:
                return f"https://www.youtube.com/watch?v={match.group(1)}"
            # Handle YouTube Shorts
            match = re.match(r'https?://(www\.)?youtube\.com/shorts/([\w-]{11})', url)
            if match:
                return f"https://www.youtube.com/watch?v={match.group(2)}"
        return url
    
    def download_video(self, url, output_filename):
        """Download mashup from Rave.DJ URL."""
        if not HAS_REQUESTS:
            raise Exception("requests library not available")
        
        self.log("[DOWNLOAD] Fetching mashup data...")
        self.progress(60)
        
        # Extract mashup ID from URL
        url_parts = url.rstrip('/').split("/")
        mashup_id = url_parts[-1]
        api_url = f"https://api.red.wemesh.ca/mashups/{mashup_id}"
        
        attempts = 0
        max_attempts = 10
        video_url = None
        
        while attempts < max_attempts:
            self.log(f"[DOWNLOAD] Attempt {attempts + 1}/{max_attempts}...")
            try:
                response = requests.get(api_url, timeout=30)
                data = response.json()
                
                if 'data' in data and 'videos' in data['data']:
                    video_url = data['data']['videos'].get('max')
                    if video_url:
                        break
                
                # Check if still processing
                if 'data' in data and data['data'].get('status') == 'PENDING':
                    self.log("[DOWNLOAD] Mashup still processing...")
            except Exception as e:
                self.log(f"[DOWNLOAD] API error: {e}")
            
            time.sleep(5)
            attempts += 1
            self.progress(60 + attempts * 2)
        
        if not video_url:
            raise Exception("Could not get video URL after multiple attempts")
        
        self.log("[DOWNLOAD] Downloading video file...")
        self.progress(80)
        
        video_response = requests.get(video_url, timeout=180, stream=True)
        
        if os.path.exists(output_filename):
            os.remove(output_filename)
        
        total_size = int(video_response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_filename, 'wb') as f:
            for chunk in video_response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = 80 + int((downloaded / total_size) * 20)
                        self.progress(min(pct, 99))
        
        self.progress(100)
        self.log(f"[DOWNLOAD] ✓ Saved: {output_filename}")
        return output_filename
    
    def init_browser(self):
        """Initialize Chrome with auto driver management."""
        if not HAS_SELENIUM:
            raise Exception("Selenium not installed!")
        
        self.log("[BROWSER] Starting Chrome...")
        
        options = webdriver.ChromeOptions()
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        try:
            if HAS_WEBDRIVER_MANAGER:
                service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=options)
            else:
                self.driver = webdriver.Chrome(options=options)
        except Exception as e:
            self.log(f"[BROWSER] Chrome error: {e}")
            self.log("[BROWSER] Trying Firefox...")
            try:
                from selenium.webdriver.firefox.service import Service as FFService
                from webdriver_manager.firefox import GeckoDriverManager
                service = FFService(GeckoDriverManager().install())
                self.driver = webdriver.Firefox(service=service)
            except:
                raise Exception("Could not start any browser!")
        
        self.driver.set_window_size(1200, 800)
        self.log("[BROWSER] ✓ Browser ready")
    
    def get_site(self):
        """Navigate to Rave.DJ."""
        self.log("[RAVE.DJ] Loading site...")
        self.driver.get('https://rave.dj/mix')
        time.sleep(2)
        self.progress(10)
    
    def dismiss_cookies(self):
        """Handle cookie consent popup."""
        try:
            popup = WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located((By.ID, 'qc-cmp2-ui'))
            )
            btn = popup.find_element(By.CSS_SELECTOR, 'button[mode="primary"]')
            btn.click()
            self.log("[RAVE.DJ] ✓ Dismissed cookies")
        except:
            pass
    
    def open_spotify_tab(self):
        """Open Spotify login tab for auth."""
        original = self.driver.current_window_handle
        self.driver.execute_script("window.open('');")
        self.driver.switch_to.window(self.driver.window_handles[-1])
        self.driver.get("https://accounts.spotify.com/en/login")
        self.log("[SPOTIFY] Login tab opened (use if needed)")
        self.driver.switch_to.window(original)
    
    def add_track(self, url):
        """Add a track to the mix."""
        driver = self.driver
        
        current = driver.find_elements(By.XPATH, "//div[contains(@class, 'track-title')]")
        initial_count = len(current)
        
        search = driver.find_element(By.CLASS_NAME, 'search-input')
        search.clear()
        search.send_keys(url)
        search.send_keys(Keys.RETURN)
        
        self.log(f"[TRACK] Adding: {url[:50]}...")
        
        try:
            WebDriverWait(driver, 15).until(
                lambda d: len(d.find_elements(By.XPATH, "//div[contains(@class, 'track-title')]")) > initial_count
            )
            self.log("[TRACK] ✓ Track added!")
            return True
        except Exception as e:
            self.log(f"[TRACK] ✗ Failed: {str(e)[:40]}")
            return False
    
    def create_mashup(self, output_file):
        """Click create and wait for mashup."""
        driver = self.driver
        initial_url = driver.current_url
        
        self.log("[MASHUP] Starting creation...")
        self.progress(25)
        
        retries = ATTEMPTS
        while retries >= 0:
            try:
                # Find and click create button
                btn = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "button.mix-button.mix-floating-footer"))
                )
                btn.click()
                
                self.log("[MASHUP] Waiting for Rave.DJ to process...")
                self.progress(30)
                
                # Wait for redirect to mashup page
                WebDriverWait(driver, SHORT_TIME_OUT).until(url_changes(initial_url))
                mashup_url = driver.current_url
                self.log(f"[MASHUP] URL: {mashup_url}")
                self.progress(40)
                
                # Wait for player to load (mashup complete)
                self.log("[MASHUP] Waiting for render (may take a few minutes)...")
                WebDriverWait(driver, TIME_OUT, poll_frequency=POLL_FREQ).until(
                    EC.presence_of_element_located((By.ID, 'ForegroundPlayer'))
                )
                
                self.progress(55)
                self.log("[MASHUP] ✓ Mashup ready!")
                
                # Download it
                return self.download_video(mashup_url, output_file)
                
            except TimeoutException:
                if driver.current_url == initial_url:
                    retries -= 1
                    self.log(f"[MASHUP] Retry... ({retries} left)")
                else:
                    # Save failed URL for later
                    with open('failed_mashups.txt', 'a') as f:
                        f.write(driver.current_url + '\n')
                    raise Exception("Mashup timed out - URL saved to failed_mashups.txt")
            except Exception as e:
                raise
        
        raise Exception("Failed to create mashup after retries")
    
    def close(self):
        """Clean up browser."""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
            self.driver = None


# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM DAW ENGINE (FALLBACK IF NO RAVE.DJ)
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleDAW:
    """Basic audio mixing when Rave.DJ is unavailable."""
    
    @staticmethod
    def generate_tone(freq, duration, sample_rate=44100, volume=0.5):
        """Generate a sine wave tone."""
        samples = []
        for i in range(int(sample_rate * duration)):
            t = i / sample_rate
            sample = volume * math.sin(2 * math.pi * freq * t)
            samples.append(sample)
        return samples
    
    @staticmethod
    def mix_samples(samples1, samples2, ratio=0.5):
        """Mix two sample arrays."""
        max_len = max(len(samples1), len(samples2))
        while len(samples1) < max_len:
            samples1.append(0)
        while len(samples2) < max_len:
            samples2.append(0)
        
        mixed = []
        for s1, s2 in zip(samples1, samples2):
            mixed.append(s1 * ratio + s2 * (1 - ratio))
        return mixed
    
    @staticmethod
    def save_wav(samples, filename, sample_rate=44100):
        """Save samples to WAV file."""
        with wave.open(filename, 'w') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            
            for sample in samples:
                clamped = max(-1, min(1, sample))
                packed = struct.pack('<h', int(clamped * 32767))
                wav.writeframesraw(packed)
        
        return filename
    
    @staticmethod
    def create_placeholder_mix(output_file, duration=10):
        """Create a placeholder mix file."""
        tone1 = SimpleDAW.generate_tone(440, duration, volume=0.3)
        tone2 = SimpleDAW.generate_tone(554.37, duration, volume=0.3)
        mixed = SimpleDAW.mix_samples(tone1, tone2, 0.5)
        
        wav_file = output_file.replace('.mp4', '.wav')
        SimpleDAW.save_wav(mixed, wav_file)
        return wav_file


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN GUI - SILVAGUNNER HQ RIPPER STYLE
# ═══════════════════════════════════════════════════════════════════════════════

class HQRipperGUI:
    """SiIvaGunner-style High Quality Ripper interface."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(f"{APP_NAME}")
        self.root.geometry("580x700")
        self.root.resizable(False, False)
        self.root.configure(bg='#f0f0f0')
        
        self.engine = None
        self.is_processing = False
        
        self.create_menu()
        self.create_notebook()
        self.create_upload_tab()
        self.create_schedule_tab()
        self.create_downloads_tab()
        self.create_settings_tab()
        self.create_status_bar()
    
    def create_menu(self):
        """Create top menu bar with tabs."""
        menu_frame = tk.Frame(self.root, bg='#e0e0e0', height=30)
        menu_frame.pack(fill='x', padx=0, pady=0)
        menu_frame.pack_propagate(False)
        
        menus = [
            ("🔴 Upload", "upload"),
            ("⏰ Schedule", "schedule"),
            ("📥 Downloads", "downloads"),
            ("⚙ Settings", "settings"),
            ("📁 Paths", "paths"),
        ]
        
        for text, cmd in menus:
            btn = tk.Button(
                menu_frame, 
                text=text, 
                relief='flat',
                bg='#e0e0e0',
                activebackground='#d0d0d0',
                font=('Segoe UI', 9),
                padx=8,
                pady=2,
                command=lambda c=cmd: self.menu_click(c)
            )
            btn.pack(side='left', padx=1, pady=2)
    
    def menu_click(self, cmd):
        """Handle menu clicks."""
        tab_map = {"upload": 0, "schedule": 1, "downloads": 2, "settings": 3}
        if cmd in tab_map:
            self.notebook.select(tab_map[cmd])
    
    def create_notebook(self):
        """Create tabbed interface."""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
    
    def create_upload_tab(self):
        """Create main upload/mashup tab."""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Upload")
        
        # FILE DOWNLOADER
        dl_frame = ttk.LabelFrame(tab, text="File Downloader", padding=5)
        dl_frame.pack(fill='x', pady=(0, 10))
        
        row1 = ttk.Frame(dl_frame)
        row1.pack(fill='x', pady=2)
        ttk.Label(row1, text="Logo link", width=12).pack(side='left')
        self.logo_entry = ttk.Entry(row1, width=55)
        self.logo_entry.pack(side='left', padx=5)
        
        row2 = ttk.Frame(dl_frame)
        row2.pack(fill='x', pady=2)
        ttk.Label(row2, text="Audio/Video", width=12).pack(side='left')
        self.video_entry = ttk.Entry(row2, width=55)
        self.video_entry.pack(side='left', padx=5)
        
        row3 = ttk.Frame(dl_frame)
        row3.pack(fill='x', pady=2)
        ttk.Label(row3, text="Title:", width=12).pack(side='left')
        self.title_entry = ttk.Entry(row3, width=40)
        self.title_entry.pack(side='left', padx=5)
        self.title_entry.insert(0, "High Quality Rip - The Video Game")
        ttk.Button(row3, text="📋 Parse Clipboard", command=self.parse_clipboard).pack(side='left', padx=5)
        
        # RAVE.DJ MASHUP
        mashup_frame = ttk.LabelFrame(tab, text="🎵 Rave.DJ Mashup (2 Tracks → Auto Remix)", padding=5)
        mashup_frame.pack(fill='x', pady=(0, 10))
        
        t1_row = ttk.Frame(mashup_frame)
        t1_row.pack(fill='x', pady=2)
        ttk.Label(t1_row, text="Track 1:", width=10).pack(side='left')
        self.track1_entry = ttk.Entry(t1_row, width=55)
        self.track1_entry.pack(side='left', padx=5)
        
        t2_row = ttk.Frame(mashup_frame)
        t2_row.pack(fill='x', pady=2)
        ttk.Label(t2_row, text="Track 2:", width=10).pack(side='left')
        self.track2_entry = ttk.Entry(t2_row, width=55)
        self.track2_entry.pack(side='left', padx=5)
        
        opt_row = ttk.Frame(mashup_frame)
        opt_row.pack(fill='x', pady=5)
        self.use_ravedj = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_row, text="Use Rave.DJ", variable=self.use_ravedj).pack(side='left')
        self.open_spotify = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_row, text="Open Spotify login", variable=self.open_spotify).pack(side='left', padx=10)
        
        # DESCRIPTION + SCHEDULER
        desc_frame = ttk.Frame(tab)
        desc_frame.pack(fill='x', pady=(0, 10))
        
        left_desc = ttk.Frame(desc_frame)
        left_desc.pack(side='left', fill='both', expand=True)
        
        self.desc_text = scrolledtext.ScrolledText(left_desc, width=35, height=5, font=('Consolas', 9))
        self.desc_text.pack(fill='both', expand=True)
        self.desc_text.insert('1.0', "High quality rip description...")
        
        right_sched = ttk.LabelFrame(desc_frame, text="Scheduler", padding=5)
        right_sched.pack(side='right', fill='y', padx=(10, 0))
        
        now = datetime.now()
        sched_grid = ttk.Frame(right_sched)
        sched_grid.pack()
        
        for i, (lbl, val, rng) in enumerate([
            ("Year", now.year, (2020, 2030)),
            ("Month", now.month, (1, 12)),
            ("Day", now.day, (1, 31)),
            ("Hour", now.hour, (0, 23)),
            ("Min", now.minute, (0, 59)),
        ]):
            ttk.Label(sched_grid, text=lbl).grid(row=i, column=0, sticky='e')
            spin = ttk.Spinbox(sched_grid, from_=rng[0], to=rng[1], width=6)
            spin.grid(row=i, column=1)
            spin.set(val)
            setattr(self, f"{lbl.lower()}_spin", spin)
        
        ttk.Button(right_sched, text="⏱ Now", command=self.set_current_date).pack(pady=3)
        
        # TAGS
        tags_row = ttk.Frame(tab)
        tags_row.pack(fill='x', pady=(0, 5))
        ttk.Label(tags_row, text="Tags:").pack(side='left')
        self.tags_entry = ttk.Entry(tags_row, width=50)
        self.tags_entry.pack(side='left', padx=5, fill='x', expand=True)
        
        # PLAYLIST
        playlist_frame = ttk.LabelFrame(tab, text="Playlist Stuff", padding=5)
        playlist_frame.pack(fill='x', pady=(0, 10))
        
        pl_row1 = ttk.Frame(playlist_frame)
        pl_row1.pack(fill='x', pady=2)
        ttk.Label(pl_row1, text="Playlist:").pack(side='left')
        self.playlist_entry = ttk.Entry(pl_row1, width=45)
        self.playlist_entry.pack(side='left', padx=5)
        
        pl_row2 = ttk.Frame(playlist_frame)
        pl_row2.pack(fill='x', pady=2)
        self.create_playlist = tk.BooleanVar()
        ttk.Checkbutton(pl_row2, text="Create new playlist?", variable=self.create_playlist).pack(side='left')
        self.add_bottom = tk.BooleanVar()
        ttk.Checkbutton(pl_row2, text="Add bottom text", variable=self.add_bottom).pack(side='left', padx=10)
        
        pl_row3 = ttk.Frame(playlist_frame)
        pl_row3.pack(fill='x', pady=2)
        self.game_entry = ttk.Entry(pl_row3, width=30)
        self.game_entry.pack(side='left', padx=5)
        self.game_entry.insert(0, "Good Eats: The Video Game")
        ttk.Button(pl_row3, text="Autocomplete", command=self.autocomplete_playlist).pack(side='left', padx=5)
        
        # UPLOADING
        upload_frame = ttk.LabelFrame(tab, text="Uploading", padding=5)
        upload_frame.pack(fill='x', pady=(0, 10))
        
        btn_row = ttk.Frame(upload_frame)
        btn_row.pack(fill='x', pady=5)
        
        self.yt_btn = tk.Button(btn_row, text="🔴 Upload and render", bg='#ff4444', fg='white',
                                font=('Segoe UI', 10, 'bold'), padx=10, pady=3, command=self.upload_render)
        self.yt_btn.pack(side='left', padx=5)
        
        ttk.Button(btn_row, text="Upload", command=self.upload_only).pack(side='left', padx=5)
        
        self.ext_btn = tk.Button(btn_row, text="▶ Use external", bg='#333', fg='white',
                                 font=('Segoe UI', 10), padx=10, pady=3, command=self.use_external)
        self.ext_btn.pack(side='right', padx=5)
        
        ttk.Button(btn_row, text="Use rendered", command=self.use_rendered).pack(side='right', padx=5)
        
        ttk.Button(upload_frame, text="📷 Thumbnail", command=self.create_thumbnail).pack(pady=3)
        
        # REMIX BUTTON
        remix_frame = ttk.Frame(tab)
        remix_frame.pack(fill='x', pady=5)
        
        self.remix_btn = tk.Button(remix_frame, text="🔥 CREATE RAVE.DJ MASHUP → Pr", bg='#9933ff', fg='white',
                                   font=('Segoe UI', 12, 'bold'), padx=15, pady=6, command=self.create_mashup)
        self.remix_btn.pack(fill='x', padx=15)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(remix_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill='x', padx=15, pady=5)
        
        self.output_label = ttk.Label(remix_frame, text="Output: REMIX_PR_READY.mp4", foreground='gray')
        self.output_label.pack()
    
    def create_schedule_tab(self):
        """Schedule tab."""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Schedule")
        
        ttk.Label(tab, text="📅 Scheduled Uploads", font=('Segoe UI', 12, 'bold')).pack(pady=10)
        
        columns = ('title', 'date', 'status')
        self.schedule_tree = ttk.Treeview(tab, columns=columns, show='headings', height=15)
        self.schedule_tree.heading('title', text='Title')
        self.schedule_tree.heading('date', text='Date')
        self.schedule_tree.heading('status', text='Status')
        self.schedule_tree.pack(fill='both', expand=True, pady=10)
        
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill='x')
        ttk.Button(btn_frame, text="➕ Add", command=self.add_schedule).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="❌ Remove", command=self.remove_schedule).pack(side='left', padx=5)
    
    def create_downloads_tab(self):
        """Downloads/log tab."""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Downloads")
        
        ttk.Label(tab, text="📥 Log Console", font=('Segoe UI', 12, 'bold')).pack(pady=10)
        
        self.log_console = scrolledtext.ScrolledText(tab, width=65, height=25, font=('Consolas', 9),
                                                      bg='#1a1a2e', fg='#00ff88')
        self.log_console.pack(fill='both', expand=True, pady=10)
        self.log_console.insert('end', f"[HQ RIPPER 7.2] Ready!\n")
        self.log_console.insert('end', f"[INFO] Selenium: {'✓' if HAS_SELENIUM else '✗'}\n")
        self.log_console.insert('end', f"[INFO] Requests: {'✓' if HAS_REQUESTS else '✗'}\n")
        self.log_console.insert('end', f"[INFO] WebDriver Manager: {'✓' if HAS_WEBDRIVER_MANAGER else '✗'}\n")
        self.log_console.insert('end', "-" * 50 + "\n")
    
    def create_settings_tab(self):
        """Settings tab."""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Settings")
        
        ttk.Label(tab, text="⚙ Settings", font=('Segoe UI', 12, 'bold')).pack(pady=10)
        
        out_frame = ttk.LabelFrame(tab, text="Output", padding=10)
        out_frame.pack(fill='x', pady=10)
        
        row1 = ttk.Frame(out_frame)
        row1.pack(fill='x', pady=5)
        ttk.Label(row1, text="Directory:").pack(side='left')
        self.output_dir = ttk.Entry(row1, width=40)
        self.output_dir.pack(side='left', padx=5)
        self.output_dir.insert(0, os.getcwd())
        ttk.Button(row1, text="Browse", command=self.browse_output).pack(side='left')
        
        row2 = ttk.Frame(out_frame)
        row2.pack(fill='x', pady=5)
        ttk.Label(row2, text="Filename:").pack(side='left')
        self.output_name = ttk.Entry(row2, width=40)
        self.output_name.pack(side='left', padx=5)
        self.output_name.insert(0, "REMIX_PR_READY.mp4")
        
        browser_frame = ttk.LabelFrame(tab, text="Browser", padding=10)
        browser_frame.pack(fill='x', pady=10)
        
        self.auto_close = tk.BooleanVar(value=True)
        ttk.Checkbutton(browser_frame, text="Auto-close browser after completion", variable=self.auto_close).pack(anchor='w')
        
        about_frame = ttk.LabelFrame(tab, text="About", padding=10)
        about_frame.pack(fill='x', pady=10)
        
        ttk.Label(about_frame, text=f"{APP_NAME}", font=('Segoe UI', 11, 'bold')).pack()
        ttk.Label(about_frame, text=f"Version {VERSION}").pack()
        ttk.Label(about_frame, text="Team Flames / Samsoft / Flames Co.").pack()
    
    def create_status_bar(self):
        """Status bar."""
        self.status_bar = ttk.Label(self.root, text="Ready", relief='sunken', anchor='w')
        self.status_bar.pack(fill='x', side='bottom')
    
    # CALLBACKS
    def log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_console.insert('end', f"[{timestamp}] {msg}\n")
        self.log_console.see('end')
        self.status_bar.config(text=msg)
        self.root.update_idletasks()
    
    def set_progress(self, value):
        self.progress_var.set(value)
        self.root.update_idletasks()
    
    def parse_clipboard(self):
        try:
            clip = self.root.clipboard_get()
            if 'youtube' in clip.lower() or 'spotify' in clip.lower():
                self.video_entry.delete(0, 'end')
                self.video_entry.insert(0, clip)
        except: pass
    
    def set_current_date(self):
        now = datetime.now()
        self.year_spin.set(now.year)
        self.month_spin.set(now.month)
        self.day_spin.set(now.day)
        self.hour_spin.set(now.hour)
        self.min_spin.set(now.minute)
    
    def autocomplete_playlist(self):
        game = self.game_entry.get()
        if game:
            self.playlist_entry.delete(0, 'end')
            self.playlist_entry.insert(0, game)
    
    def browse_output(self):
        path = filedialog.askdirectory()
        if path:
            self.output_dir.delete(0, 'end')
            self.output_dir.insert(0, path)
    
    def upload_render(self):
        self.log("[UPLOAD] Upload and render")
        messagebox.showinfo("Upload", "YouTube API integration placeholder")
    
    def upload_only(self):
        self.log("[UPLOAD] Upload only")
    
    def use_external(self):
        output = os.path.join(self.output_dir.get(), self.output_name.get())
        if os.path.exists(output):
            if sys.platform == 'darwin':
                os.system(f'open "{output}"')
            elif sys.platform == 'win32':
                os.startfile(output)
            else:
                os.system(f'xdg-open "{output}"')
    
    def use_rendered(self):
        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.mkv *.webm")])
        if path:
            self.video_entry.delete(0, 'end')
            self.video_entry.insert(0, path)
    
    def create_thumbnail(self):
        messagebox.showinfo("Thumbnail", "Thumbnail generator")
    
    def add_schedule(self):
        title = self.title_entry.get() or "Untitled"
        date = f"{self.year_spin.get()}-{int(self.month_spin.get()):02d}-{int(self.day_spin.get()):02d}"
        self.schedule_tree.insert('', 'end', values=(title, date, 'Pending'))
    
    def remove_schedule(self):
        for item in self.schedule_tree.selection():
            self.schedule_tree.delete(item)
    
    def create_mashup(self):
        if self.is_processing:
            messagebox.showwarning("Busy", "Already processing!")
            return
        
        track1 = self.track1_entry.get().strip()
        track2 = self.track2_entry.get().strip()
        
        if not track1 or not track2:
            messagebox.showerror("Error", "Enter both Track 1 and Track 2!")
            return
        
        if not RaveDJEngine.verify_link(track1):
            messagebox.showerror("Error", f"Invalid Track 1: {track1}")
            return
        
        if not RaveDJEngine.verify_link(track2):
            messagebox.showerror("Error", f"Invalid Track 2: {track2}")
            return
        
        self.is_processing = True
        self.remix_btn.config(state='disabled', text="⏳ Processing...")
        
        thread = threading.Thread(target=self._mashup_thread, args=(track1, track2), daemon=True)
        thread.start()
    
    def _mashup_thread(self, track1, track2):
        output_file = os.path.join(self.output_dir.get(), self.output_name.get())
        
        try:
            if self.use_ravedj.get() and HAS_SELENIUM:
                self.log("[MASHUP] Starting Rave.DJ...")
                self.set_progress(5)
                
                engine = RaveDJEngine(log_callback=self.log, progress_callback=self.set_progress)
                engine.init_browser()
                engine.get_site()
                engine.dismiss_cookies()
                
                if self.open_spotify.get():
                    engine.open_spotify_tab()
                
                clean1 = RaveDJEngine.clean_url(track1)
                clean2 = RaveDJEngine.clean_url(track2)
                
                self.set_progress(15)
                if not engine.add_track(clean1):
                    raise Exception("Failed to add Track 1")
                
                self.set_progress(20)
                if not engine.add_track(clean2):
                    raise Exception("Failed to add Track 2")
                
                engine.create_mashup(output_file)
                
                if self.auto_close.get():
                    engine.close()
                
                self.root.after(0, lambda: messagebox.showinfo("Success! 🎉", f"Mashup created!\n\n{output_file}"))
            else:
                self.log("[MASHUP] Using fallback DAW...")
                self.set_progress(50)
                wav_file = SimpleDAW.create_placeholder_mix(output_file)
                self.set_progress(100)
                self.root.after(0, lambda: messagebox.showinfo("Fallback", f"Created: {wav_file}"))
        
        except Exception as e:
            self.log(f"[ERROR] {str(e)}")
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        
        finally:
            self.is_processing = False
            self.root.after(0, lambda: self.remix_btn.config(state='normal', text="🔥 CREATE RAVE.DJ MASHUP → Pr"))
            self.root.after(0, lambda: self.set_progress(0))
    
    def run(self):
        self.root.mainloop()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═" * 50)
    print("  CAT'S EZGUNNER HQ RIPPER 7.2")
    print("  SiIvaGunner-style Mashup Tool")
    print("  Team Flames / Samsoft / Flames Co.")
    print("═" * 50 + "\n")
    
    app = HQRipperGUI()
    app.run()
