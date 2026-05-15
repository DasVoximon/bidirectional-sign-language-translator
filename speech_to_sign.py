# =============================================================================
# speech_to_sign.py
# Direction 2: Text / Speech  →  Sign Language (Skeleton Animation)
#
# Integrates with Application.py — run this file standalone OR embed the
# SpeechToSignApp class into the existing Tkinter root window.
#
# Dependencies (add to requirements.txt):
#   SpeechRecognition==3.10.4
#   pyaudio==0.2.14          ← required by SpeechRecognition for mic input
# =============================================================================

import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import math
import threading
import time
import queue

# SpeechRecognition is optional — app works without it (text input only)
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

# ---------------------------------------------------------------------------
# ASL Landmark Definitions
# Each letter is defined as a dict of landmark coordinates (x, y) for the
# 21 MediaPipe hand points, rendered on a 400×400 canvas.
# Origin (0,0) = top-left.  Wrist = pt[0], fingertips = pts[4,8,12,16,20].
#
# Coordinate system matches your existing Application.py white-canvas renderer
# exactly so the skeleton looks identical to Direction 1 output.
# ---------------------------------------------------------------------------

# Base relaxed hand — fingers slightly open, palm facing camera
_BASE = {
    0:  (200, 340),   # wrist
    1:  (170, 295),   # thumb CMC
    2:  (145, 265),   # thumb MCP
    3:  (125, 240),   # thumb IP
    4:  (108, 218),   # thumb tip
    5:  (175, 250),   # index MCP
    6:  (170, 210),   # index PIP
    7:  (168, 178),   # index DIP
    8:  (167, 150),   # index tip
    9:  (200, 245),   # middle MCP
    10: (198, 203),   # middle PIP
    11: (197, 170),   # middle DIP
    12: (196, 142),   # middle tip
    13: (225, 250),   # ring MCP
    14: (228, 210),   # ring PIP
    15: (230, 178),   # ring DIP
    16: (231, 150),   # ring tip
    17: (250, 260),   # pinky MCP
    18: (258, 223),   # pinky PIP
    19: (263, 196),   # pinky DIP
    20: (267, 172),   # pinky tip
}

def _pt(base, overrides):
    """Return a full 21-point landmark dict from base + overrides."""
    pts = dict(base)
    pts.update(overrides)
    return pts

# Closed fist helper — all fingers curled
_FIST = _pt(_BASE, {
    4:  (155, 270),
    6:  (178, 285), 7:  (182, 305), 8:  (178, 320),
    10: (200, 285), 11: (202, 308), 12: (200, 322),
    14: (222, 283), 15: (226, 305), 16: (224, 320),
    18: (244, 288), 19: (250, 308), 20: (250, 322),
})

# ---------------------------------------------------------------------------
# Full A–Z + SPACE landmark dictionaries
# ---------------------------------------------------------------------------
ASL_LANDMARKS = {

    'A': _pt(_FIST, {4: (148, 250)}),          # fist, thumb beside index

    'B': _pt(_BASE, {                            # four fingers straight up, thumb across
        4:  (155, 270),
        6:  (175, 195), 7:  (174, 158), 8:  (173, 128),
        10: (200, 192), 11: (199, 155), 12: (198, 125),
        14: (224, 195), 15: (226, 158), 16: (227, 128),
        18: (247, 202), 19: (252, 168), 20: (254, 140),
    }),

    'C': _pt(_BASE, {                            # curved C-shape
        4:  (130, 215),
        6:  (172, 198), 7:  (162, 170), 8:  (148, 148),
        10: (198, 192), 11: (192, 162), 12: (182, 140),
        14: (222, 198), 15: (218, 170), 16: (210, 150),
        18: (244, 210), 19: (242, 185), 20: (238, 165),
    }),

    'D': _pt(_BASE, {                            # index up, others curl to thumb
        4:  (148, 200),
        6:  (172, 198), 7:  (170, 162), 8:  (168, 135),
        10: (200, 280), 11: (200, 300), 12: (198, 315),
        12: (190, 210),   # middle touches thumb
        14: (222, 280), 15: (228, 302), 16: (226, 318),
        18: (246, 285), 19: (252, 306), 20: (252, 320),
    }),

    'E': _pt(_BASE, {                            # fingers bent at MCP, thumb tucked
        4:  (165, 282),
        6:  (175, 248), 7:  (180, 270), 8:  (178, 285),
        10: (200, 245), 11: (202, 268), 12: (200, 282),
        14: (224, 248), 15: (228, 268), 16: (226, 282),
        18: (246, 255), 19: (252, 272), 20: (252, 285),
    }),

    'F': _pt(_BASE, {                            # index+thumb circle, others up
        4:  (155, 205),
        8:  (162, 215),   # index tip touches thumb
        6:  (168, 240), 7:  (162, 228),
        10: (200, 192), 11: (199, 155), 12: (198, 125),
        14: (224, 195), 15: (226, 158), 16: (227, 128),
        18: (247, 202), 19: (252, 168), 20: (254, 140),
    }),

    'G': _pt(_BASE, {                            # index pointing sideways, thumb parallel
        4:  (115, 268),
        6:  (165, 262), 7:  (142, 258), 8:  (120, 255),
        10: (200, 280), 11: (200, 300), 12: (198, 316),
        14: (224, 280), 15: (228, 302), 16: (226, 318),
        18: (246, 285), 19: (252, 306), 20: (252, 322),
    }),

    'H': _pt(_BASE, {                            # index+middle pointing sideways
        4:  (118, 272),
        6:  (165, 262), 7:  (142, 258), 8:  (120, 255),
        10: (190, 268), 11: (170, 264), 12: (150, 262),
        14: (224, 280), 15: (228, 302), 16: (226, 318),
        18: (246, 285), 19: (252, 306), 20: (252, 322),
    }),

    'I': _pt(_FIST, {                            # pinky up
        18: (256, 218), 19: (262, 190), 20: (266, 165),
        4:  (155, 268),
    }),

    'J': _pt(_FIST, {                            # pinky up + hook (same static as I)
        18: (256, 218), 19: (262, 190), 20: (266, 165),
        4:  (155, 268),
    }),

    'K': _pt(_BASE, {                            # index+middle up spread, thumb between
        4:  (160, 215),
        6:  (168, 198), 7:  (165, 162), 8:  (162, 135),
        10: (198, 195), 11: (196, 160), 12: (194, 132),
        14: (224, 280), 15: (228, 302), 16: (226, 318),
        18: (246, 285), 19: (252, 306), 20: (252, 322),
    }),

    'L': _pt(_BASE, {                            # L-shape: index up, thumb out
        4:  (115, 255),
        6:  (172, 195), 7:  (170, 158), 8:  (168, 130),
        10: (200, 280), 11: (200, 300), 12: (198, 316),
        14: (224, 280), 15: (228, 302), 16: (226, 318),
        18: (246, 285), 19: (252, 306), 20: (252, 322),
    }),

    'M': _pt(_FIST, {                            # thumb under three fingers
        4:  (195, 295),
        8:  (178, 300), 12: (200, 300), 16: (222, 300),
    }),

    'N': _pt(_FIST, {                            # thumb under two fingers
        4:  (188, 295),
        8:  (178, 298), 12: (200, 298),
        16: (224, 280), 15: (228, 302), 18: (246, 285),
    }),

    'O': _pt(_BASE, {                            # O-shape: all fingers curved to thumb
        4:  (148, 210),
        6:  (170, 205), 7:  (162, 185), 8:  (152, 170),
        10: (196, 200), 11: (190, 180), 12: (182, 165),
        14: (220, 205), 15: (216, 185), 16: (210, 170),
        18: (242, 215), 19: (240, 196), 20: (236, 180),
    }),

    'P': _pt(_BASE, {                            # like K but pointing downward
        4:  (158, 285),
        6:  (172, 270), 7:  (172, 295), 8:  (170, 315),
        10: (200, 268), 11: (200, 290), 12: (198, 308),
        14: (224, 280), 15: (228, 302), 16: (226, 318),
        18: (246, 285), 19: (252, 306), 20: (252, 322),
    }),

    'Q': _pt(_BASE, {                            # G pointing down
        4:  (118, 285),
        6:  (165, 275), 7:  (145, 285), 8:  (125, 295),
        10: (200, 280), 11: (200, 300), 12: (198, 316),
        14: (224, 280), 15: (228, 302), 16: (226, 318),
        18: (246, 285), 19: (252, 306), 20: (252, 322),
    }),

    'R': _pt(_BASE, {                            # index+middle crossed
        6:  (172, 195), 7:  (174, 162), 8:  (176, 135),
        10: (185, 192), 11: (182, 158), 12: (178, 130),
        14: (224, 280), 15: (228, 302), 16: (226, 318),
        18: (246, 285), 19: (252, 306), 20: (252, 322),
        4:  (148, 268),
    }),

    'S': _pt(_FIST, {4: (168, 278)}),           # fist, thumb over fingers

    'T': _pt(_FIST, {                            # thumb between index+middle
        4:  (178, 265),
        8:  (178, 298),
    }),

    'U': _pt(_BASE, {                            # index+middle up together
        6:  (172, 195), 7:  (172, 160), 8:  (171, 130),
        10: (198, 192), 11: (197, 158), 12: (196, 128),
        14: (224, 280), 15: (228, 302), 16: (226, 318),
        18: (246, 285), 19: (252, 306), 20: (252, 322),
        4:  (148, 268),
    }),

    'V': _pt(_BASE, {                            # index+middle up spread (victory)
        6:  (168, 195), 7:  (165, 160), 8:  (163, 130),
        10: (202, 192), 11: (204, 158), 12: (205, 128),
        14: (224, 280), 15: (228, 302), 16: (226, 318),
        18: (246, 285), 19: (252, 306), 20: (252, 322),
        4:  (148, 268),
    }),

    'W': _pt(_BASE, {                            # index+middle+ring up spread
        6:  (165, 195), 7:  (162, 160), 8:  (160, 130),
        10: (198, 192), 11: (197, 158), 12: (196, 128),
        14: (228, 195), 15: (232, 160), 16: (234, 130),
        18: (248, 285), 19: (254, 308), 20: (255, 322),
        4:  (148, 268),
    }),

    'X': _pt(_BASE, {                            # index hooked
        6:  (172, 220), 7:  (165, 205), 8:  (155, 195),
        10: (200, 280), 11: (200, 300), 12: (198, 316),
        14: (224, 280), 15: (228, 302), 16: (226, 318),
        18: (246, 285), 19: (252, 306), 20: (252, 322),
        4:  (148, 258),
    }),

    'Y': _pt(_FIST, {                            # thumb + pinky out
        4:  (115, 258),
        18: (255, 218), 19: (262, 190), 20: (268, 165),
    }),

    'Z': _pt(_BASE, {                            # index pointing (draws Z — static = D-like)
        6:  (172, 198), 7:  (170, 162), 8:  (168, 135),
        10: (200, 280), 11: (200, 300), 12: (198, 316),
        14: (224, 280), 15: (228, 302), 16: (226, 318),
        18: (246, 285), 19: (252, 306), 20: (252, 322),
        4:  (148, 258),
    }),

    ' ': _pt(_BASE, {}),                         # space = open relaxed hand
}


# ---------------------------------------------------------------------------
# Skeleton renderer — mirrors Application.py's drawing code exactly
# ---------------------------------------------------------------------------

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),       # thumb
    (5,6),(6,7),(7,8),             # index
    (9,10),(10,11),(11,12),        # middle
    (13,14),(14,15),(15,16),       # ring
    (17,18),(18,19),(19,20),       # pinky
    (5,9),(9,13),(13,17),          # palm across
    (0,5),(0,17),                  # wrist to edge MCPs
]


def render_skeleton(pts_dict, canvas_size=400,
                    line_color=(0, 255, 0), dot_color=(0, 0, 255),
                    bg_color=(255, 255, 255)):
    """
    Render a hand skeleton onto a blank canvas.

    Parameters
    ----------
    pts_dict : dict  {landmark_index: (x, y)}
    canvas_size : int
    line_color, dot_color, bg_color : BGR tuples

    Returns
    -------
    numpy array (H, W, 3) uint8  — RGB image ready for PIL/Tkinter
    """
    canvas = np.full((canvas_size, canvas_size, 3), bg_color, dtype=np.uint8)

    for a, b in CONNECTIONS:
        if a in pts_dict and b in pts_dict:
            cv2.line(canvas,
                     (int(pts_dict[a][0]), int(pts_dict[a][1])),
                     (int(pts_dict[b][0]), int(pts_dict[b][1])),
                     line_color, 3)

    for i in range(21):
        if i in pts_dict:
            cv2.circle(canvas,
                       (int(pts_dict[i][0]), int(pts_dict[i][1])),
                       5, dot_color, -1)

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Interpolation helpers for smooth transition between poses
# ---------------------------------------------------------------------------

def lerp_pose(pose_a, pose_b, t):
    """Linear interpolation between two landmark dicts, t in [0,1]."""
    result = {}
    keys = set(pose_a.keys()) | set(pose_b.keys())
    for k in keys:
        ax, ay = pose_a.get(k, (200, 300))
        bx, by = pose_b.get(k, (200, 300))
        result[k] = (ax + (bx - ax) * t, ay + (by - ay) * t)
    return result


def ease_in_out(t):
    """Smooth step easing function."""
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------------------
# Text normaliser — converts input text to a sequence of sign tokens
# ---------------------------------------------------------------------------

# Simple word-level sign vocabulary (extend as needed)
WORD_SIGNS = {
    'hello':   ['H','E','L','L','O'],
    'thanks':  ['T','H','A','N','K','S'],
    'please':  ['P','L','E','A','S','E'],
    'yes':     ['Y','E','S'],
    'no':      ['N','O'],
    'help':    ['H','E','L','P'],
    'sorry':   ['S','O','R','R','Y'],
    'good':    ['G','O','O','D'],
    'bad':     ['B','A','D'],
    'love':    ['L','O','V','E'],
    'you':     ['Y','O','U'],
    'me':      ['M','E'],
    'i':       ['I'],
    'thank':   ['T','H','A','N','K'],
    'name':    ['N','A','M','E'],
    'what':    ['W','H','A','T'],
    'where':   ['W','H','E','R','E'],
    'when':    ['W','H','E','N'],
    'how':     ['H','O','W'],
    'why':     ['W','H','Y'],
}

def text_to_sign_tokens(text):
    """
    Convert a text string to a list of sign tokens.
    Each token is either a single uppercase letter ('A'–'Z') or ' ' (space).
    Words found in WORD_SIGNS are expanded; others are fingerspelled.
    """
    tokens = []
    words = text.strip().split()
    for idx, word in enumerate(words):
        w = word.lower().strip(".,!?;:'\"")
        if w in WORD_SIGNS:
            tokens.extend(WORD_SIGNS[w])
        else:
            # fingerspell letter by letter
            for ch in w.upper():
                if ch in ASL_LANDMARKS:
                    tokens.append(ch)
        if idx < len(words) - 1:
            tokens.append(' ')   # inter-word pause
    return tokens


# ---------------------------------------------------------------------------
# SpeechToSignApp — the Direction 2 Tkinter window
# ---------------------------------------------------------------------------

class SpeechToSignApp:
    """
    Standalone Direction 2 window.
    Can be launched independently or embedded alongside Direction 1.
    """

    HOLD_FRAMES   = 25    # frames to hold each sign pose
    TRANSIT_FRAMES = 15   # frames for transition between signs
    FPS_MS        = 40    # ~25 fps refresh (milliseconds)

    def __init__(self, master=None):
        """
        Parameters
        ----------
        master : tk.Tk or tk.Toplevel, optional
            If None, creates its own Tk root.
        """
        self._owns_root = master is None
        if self._owns_root:
            self.root = tk.Tk()
        else:
            self.root = tk.Toplevel(master)

        self.root.title("Text / Speech  →  Sign Language")
        self.root.geometry("860x680")
        self.root.configure(bg="#1a1a2e")
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)

        # Animation state
        self._tokens        = []
        self._token_idx     = 0
        self._frame_counter = 0
        self._phase         = 'idle'    # 'hold' | 'transit' | 'idle'
        self._current_pose  = dict(_BASE)
        self._next_pose     = dict(_BASE)
        self._is_playing    = False
        self._stop_flag     = False

        # Speech recognition state
        self._sr_thread  = None
        self._sr_queue   = queue.Queue()
        self._listening  = False

        self._build_ui()
        self._animate()   # start animation loop

        if self._owns_root:
            self.root.mainloop()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = self.root

        # ── Title ──────────────────────────────────────────────────────
        tk.Label(root,
                 text="SIGN LANGUAGE OUTPUT",
                 font=("Courier New", 18, "bold"),
                 fg="#e0e0e0", bg="#1a1a2e"
                 ).pack(pady=(18, 4))

        tk.Label(root,
                 text="Direction 2 : Text / Speech  →  Sign Skeleton",
                 font=("Courier New", 10),
                 fg="#7a7aaa", bg="#1a1a2e"
                 ).pack(pady=(0, 12))

        # ── Skeleton Canvas ────────────────────────────────────────────
        canvas_frame = tk.Frame(root, bg="#0d0d1a",
                                highlightbackground="#3a3a6a",
                                highlightthickness=2)
        canvas_frame.pack(pady=4)

        self.skeleton_label = tk.Label(canvas_frame, bg="#0d0d1a")
        self.skeleton_label.pack(padx=4, pady=4)

        # ── Current sign label ─────────────────────────────────────────
        self.sign_display = tk.Label(root,
                                     text="—",
                                     font=("Courier New", 48, "bold"),
                                     fg="#00ff9f", bg="#1a1a2e")
        self.sign_display.pack(pady=4)

        # ── Progress / token strip ─────────────────────────────────────
        self.token_strip = tk.Label(root,
                                    text="",
                                    font=("Courier New", 13),
                                    fg="#8888bb", bg="#1a1a2e",
                                    wraplength=820)
        self.token_strip.pack(pady=(0, 8))

        # ── Input area ─────────────────────────────────────────────────
        input_frame = tk.Frame(root, bg="#1a1a2e")
        input_frame.pack(fill=tk.X, padx=30, pady=4)

        self.text_entry = tk.Entry(input_frame,
                                   font=("Courier New", 15),
                                   bg="#0d0d1a", fg="#e0e0e0",
                                   insertbackground="#00ff9f",
                                   relief=tk.FLAT,
                                   bd=0)
        self.text_entry.pack(side=tk.LEFT, fill=tk.X, expand=True,
                              ipady=10, padx=(0, 8))
        self.text_entry.bind("<Return>", lambda e: self._on_translate())

        tk.Button(input_frame,
                  text="SIGN IT",
                  font=("Courier New", 12, "bold"),
                  bg="#00aa6f", fg="#ffffff",
                  activebackground="#00cc88",
                  relief=tk.FLAT, padx=14, pady=8,
                  command=self._on_translate
                  ).pack(side=tk.LEFT)

        # ── Voice button ───────────────────────────────────────────────
        btn_frame = tk.Frame(root, bg="#1a1a2e")
        btn_frame.pack(pady=6)

        self.mic_btn = tk.Button(btn_frame,
                                 text="🎤  SPEAK",
                                 font=("Courier New", 12),
                                 bg="#1e3a5f", fg="#aaddff",
                                 activebackground="#2a5080",
                                 relief=tk.FLAT, padx=14, pady=8,
                                 command=self._on_mic,
                                 state=tk.NORMAL if SR_AVAILABLE else tk.DISABLED)
        self.mic_btn.pack(side=tk.LEFT, padx=8)

        tk.Button(btn_frame,
                  text="⏹  STOP",
                  font=("Courier New", 12),
                  bg="#3a1a1a", fg="#ffaaaa",
                  activebackground="#502020",
                  relief=tk.FLAT, padx=14, pady=8,
                  command=self._on_stop
                  ).pack(side=tk.LEFT, padx=8)

        # ── Status bar ─────────────────────────────────────────────────
        self.status_label = tk.Label(root,
                                     text="Ready" if SR_AVAILABLE
                                          else "Ready  (install SpeechRecognition for mic input)",
                                     font=("Courier New", 9),
                                     fg="#555580", bg="#1a1a2e")
        self.status_label.pack(side=tk.BOTTOM, pady=6)

        # Show idle skeleton
        self._render_pose(self._current_pose)

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------

    def _on_translate(self):
        text = self.text_entry.get().strip()
        if not text:
            return
        self._start_animation(text)

    def _on_stop(self):
        self._stop_flag = True
        self._is_playing = False
        self._phase = 'idle'
        self._tokens = []
        self._token_idx = 0
        self.sign_display.config(text="—")
        self.token_strip.config(text="")
        self._set_status("Stopped.")

    def _on_mic(self):
        if not SR_AVAILABLE:
            self._set_status("SpeechRecognition not installed. Run: pip install SpeechRecognition pyaudio")
            return
        if self._listening:
            return
        self._listening = True
        self.mic_btn.config(text="🔴  Listening...", fg="#ff6644")
        self._set_status("Listening — speak now...")
        self._sr_thread = threading.Thread(target=self._listen_worker, daemon=True)
        self._sr_thread.start()
        self.root.after(100, self._check_sr_queue)

    # ------------------------------------------------------------------
    # Speech recognition worker (runs in background thread)
    # ------------------------------------------------------------------

    def _listen_worker(self):
        r = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source, duration=0.5)
                audio = r.listen(source, timeout=6, phrase_time_limit=8)
            text = r.recognize_google(audio)
            self._sr_queue.put(('result', text))
        except sr.WaitTimeoutError:
            self._sr_queue.put(('error', 'No speech detected. Try again.'))
        except sr.UnknownValueError:
            self._sr_queue.put(('error', 'Could not understand audio.'))
        except sr.RequestError as e:
            self._sr_queue.put(('error', f'STT service error: {e}'))
        except Exception as e:
            self._sr_queue.put(('error', str(e)))

    def _check_sr_queue(self):
        try:
            kind, payload = self._sr_queue.get_nowait()
            self._listening = False
            self.mic_btn.config(text="🎤  SPEAK", fg="#aaddff")
            if kind == 'result':
                self.text_entry.delete(0, tk.END)
                self.text_entry.insert(0, payload)
                self._set_status(f'Heard: "{payload}"')
                self._start_animation(payload)
            else:
                self._set_status(payload)
        except queue.Empty:
            self.root.after(100, self._check_sr_queue)

    # ------------------------------------------------------------------
    # Animation control
    # ------------------------------------------------------------------

    def _start_animation(self, text):
        tokens = text_to_sign_tokens(text)
        if not tokens:
            return
        self._stop_flag   = False
        self._tokens      = tokens
        self._token_idx   = 0
        self._frame_counter = 0
        self._phase       = 'hold'
        self._is_playing  = True

        first_token = tokens[0]
        self._current_pose = ASL_LANDMARKS.get(first_token, _BASE)
        self._update_token_strip(0)
        self.sign_display.config(text=first_token if first_token != ' ' else '(space)')
        self._set_status(f"Signing: {len(tokens)} gesture(s)")

    def _animate(self):
        """Main animation tick — called every FPS_MS milliseconds via after()."""
        if self._is_playing and not self._stop_flag:
            self._tick()
        self.root.after(self.FPS_MS, self._animate)

    def _tick(self):
        if self._phase == 'hold':
            # Show current sign for HOLD_FRAMES frames
            self._render_pose(self._current_pose)
            self._frame_counter += 1
            if self._frame_counter >= self.HOLD_FRAMES:
                self._frame_counter = 0
                # Move to next token
                self._token_idx += 1
                if self._token_idx >= len(self._tokens):
                    self._finish()
                    return
                next_token = self._tokens[self._token_idx]
                self._next_pose = ASL_LANDMARKS.get(next_token, _BASE)
                self._phase = 'transit'
                self.sign_display.config(
                    text=next_token if next_token != ' ' else '(space)')
                self._update_token_strip(self._token_idx)

        elif self._phase == 'transit':
            # Smoothly interpolate from current_pose to next_pose
            t = self._frame_counter / self.TRANSIT_FRAMES
            t_eased = ease_in_out(min(t, 1.0))
            interp = lerp_pose(self._current_pose, self._next_pose, t_eased)
            self._render_pose(interp)
            self._frame_counter += 1
            if self._frame_counter >= self.TRANSIT_FRAMES:
                self._frame_counter  = 0
                self._current_pose   = self._next_pose
                self._phase          = 'hold'

    def _finish(self):
        self._is_playing  = False
        self._phase       = 'idle'
        self._token_idx   = 0
        self._render_pose(dict(_BASE))
        self.sign_display.config(text="✓")
        self._set_status("Done. Enter new text or speak.")
        self.token_strip.config(text="")

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _render_pose(self, pts):
        img = render_skeleton(pts)
        pil_img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=pil_img)
        self.skeleton_label.imgtk = imgtk   # keep reference
        self.skeleton_label.config(image=imgtk)

    def _update_token_strip(self, current_idx):
        parts = []
        for i, tok in enumerate(self._tokens):
            display = tok if tok != ' ' else '·'
            if i < current_idx:
                parts.append(f"[{display}]")
            elif i == current_idx:
                parts.append(f"►{display}◄")
            else:
                parts.append(display)
        self.token_strip.config(text="  ".join(parts))

    def _set_status(self, msg):
        self.status_label.config(text=msg)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def on_close(self):
        self._stop_flag = True
        if self._owns_root:
            self.root.destroy()
        else:
            self.root.destroy()


# =============================================================================
# Integration helper — call this from Application.py to add a "Direction 2"
# button that opens this window alongside the existing Direction 1 window.
# =============================================================================

def add_direction2_button(app_instance):
    """
    Call this from Application.__init__() after the GUI is built.

    Example (add to end of Application.__init__):
        from speech_to_sign import add_direction2_button
        add_direction2_button(self)
    """
    btn = tk.Button(
        app_instance.root,
        text="Text→Sign\n(Dir. 2)",
        font=("Josefin Sans", 14),
        bg="#1e3a5f",
        fg="#aaddff",
        activebackground="#2a5080",
        relief=tk.FLAT,
        padx=10, pady=8,
        command=lambda: SpeechToSignApp(master=app_instance.root)
    )
    btn.place(x=1100, y=630)
    return btn


# =============================================================================
# Entry point — run Direction 2 standalone
# =============================================================================

if __name__ == "__main__":
    SpeechToSignApp()
