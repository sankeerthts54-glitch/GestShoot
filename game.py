"""
Hand Gesture Space Shooter  —  v3.1  "Neon Horizon"
=======================================================
NEW in v3:
  - Kalman-filter hand tracking  (smooth, predictive, jitter-free)
  - Normalized pinch detection   (scales with hand distance from camera)
  - Deadzone + coast-on-loss     (ship glides when hand vanishes)
  - Neon glow rendering system   (single-pass Gaussian bloom layer)
  - 3-layer parallax star field
  - Premium multi-polygon spaceship with animated engine
  - Scout + Elite enemy designs
  - Frosted-glass HUD panels with corner brackets
  - Combo multiplier  (rapid kills = bonus points)
  - Screen shake on life-loss
  - Per-level background tint shift

NEW in v3.1:
  - Centre-zone gameplay  (ship + enemies locked to middle 64 % of screen)
  - Hand X remapped from inner webcam range → full zone width
  - Enemy drift now bounces off zone walls, never escapes
  - Snappier Kalman filter for reduced input lag

Controls:
  Move hand   -- steer spaceship (index finger tip)
  Pinch       -- shoot
  R           -- restart
  Q / ESC     -- quit
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode
import random, math, time, os, json, threading, urllib.request

os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# ═══════════════════════════════════════════════════════
#  SOUND ENGINE
# ═══════════════════════════════════════════════════════
try:
    import sounddevice as sd
    _AUDIO_OK = True
except ImportError:
    _AUDIO_OK = False

SR = 22050

def _play(w):
    if not _AUDIO_OK: return
    def _r():
        try: sd.play(w.astype(np.float32), samplerate=SR, blocking=False)
        except: pass
    threading.Thread(target=_r, daemon=True).start()

def _sine(freq, dur, vol=0.3, fade=0.01):
    n = int(SR * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    w = np.sin(2 * np.pi * freq * t) * vol
    fe = int(SR * fade)
    if fe > 0:
        env = np.ones(n)
        env[:fe] = np.linspace(0, 1, fe)
        env[-fe:] = np.linspace(1, 0, fe)
        w *= env
    return w

def _noise(dur=0.1, vol=0.3):
    n = int(SR * dur)
    w = (np.random.rand(n) * 2 - 1) * vol
    for i in range(1, n):
        w[i] = 0.65 * w[i] + 0.35 * w[i - 1]
    return w * np.linspace(1, 0, n) ** 2

def _sweep(f0, f1, dur, vol=0.3):
    n = int(SR * dur)
    f = np.linspace(f0, f1, n)
    phase = np.cumsum(f / SR) * 2 * np.pi
    return np.sin(phase) * vol * np.linspace(1, 0, n) ** 1.5

def _arp(freqs, nd=0.07, vol=0.28):
    return np.concatenate([_sine(f, nd, vol) for f in freqs])

def _mix(a, b):
    if len(a) >= len(b):
        out = a.copy(); out[:len(b)] += b
    else:
        out = b.copy(); out[:len(a)] += a
    return out

SFX = {}
def build_sfx():
    SFX["shoot"]     = _mix(_sine(920, 0.05, 0.20), _sine(580, 0.09, 0.10))
    SFX["explode"]   = _mix(_noise(0.22, 0.48), _sine(105, 0.15, 0.22))
    SFX["life_lost"] = _sweep(500, 65, 0.42, 0.44)
    SFX["level_up"]  = _arp([523, 659, 784, 1047], 0.075, 0.28)
    SFX["game_over"] = np.concatenate([_sweep(750, 110, 0.58, 0.40), _sine(85, 0.50, 0.32)])

def sfx(name):
    w = SFX.get(name)
    if w is not None: _play(w)


# ═══════════════════════════════════════════════════════
#  HIGH SCORE
# ═══════════════════════════════════════════════════════
_HS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "highscore.json")

def load_hs() -> int:
    try:
        with open(_HS_FILE) as f: return int(json.load(f).get("high_score", 0))
    except: return 0

def save_hs(score: int):
    try:
        if score > load_hs():
            with open(_HS_FILE, "w") as f: json.dump({"high_score": score}, f)
    except: pass


# ═══════════════════════════════════════════════════════
#  MODEL AUTO-DOWNLOAD
# ═══════════════════════════════════════════════════════
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("  Downloading hand landmark model (~7.8 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("  Done.")


# ═══════════════════════════════════════════════════════
#  CONSTANTS & PALETTE
# ═══════════════════════════════════════════════════════
WINDOW_TITLE = "Hand Gesture Space Shooter  |  Neon Horizon v3"
WAIT_MS      = 33           # ~30 FPS

OVERLAY_ALPHA    = 0.32     # webcam brightness at this fraction
GLOW_STRENGTH    = 0.55     # bloom intensity
GLOW_BLUR        = 21       # Gaussian kernel size (odd)

# Ship
SHIP_W, SHIP_H = 46, 56

# Bullets
BULLET_W, BULLET_H, BULLET_SPEED = 5, 15, 17

# Enemies
ENEMY_SIZE       = 36
BASE_SPD         = 2.8
BASE_SPAWN       = 48
MIN_SPAWN        = 16
MAX_SPD          = 9.5

# Pinch — NORMALIZED to hand span (robust to camera distance)
PINCH_NORM_THR   = 0.145    # fraction of wrist-to-midtip span
PINCH_COOLDOWN   = 18       # frames between shots

# Kalman hand tracker
KFILT_Q_POS     = 2.5       # process noise: position  (higher = snappier)
KFILT_Q_VEL     = 3.5       # process noise: velocity
KFILT_R         = 1.8       # measurement noise        (lower  = snappier)
COAST_FRAMES    = 20        # frames to coast when hand disappears

# Explosions
EXP_LIFE   = 14
EXP_R_MAX  = 40

# Difficulty
KILLS_PER_LVL  = 8
STARTING_LIVES = 5

# Combo
COMBO_WIN   = 85    # frames within which kills chain
COMBO_CAP   = 8

# Screen shake
SHAKE_DUR   = 16
SHAKE_MAG   = 8

# HUD
FONT         = cv2.FONT_HERSHEY_SIMPLEX
FS_XL        = 1.55
FS_L         = 1.10
FS_M         = 0.72
FS_S         = 0.54
INSTR_SECS   = 6.0
BANNER_F     = 75

# Star field: (count, r_min, r_max, b_min, b_max, parallax)
STAR_LAYERS = [
    (60, 1, 1,  50, 120, 0.018),   # distant
    (32, 1, 2, 120, 195, 0.055),   # mid
    (16, 2, 3, 200, 255, 0.130),   # close
]

# ── Active-zone: centre ~18%–82% of screen width ────────
# Hand recognition is unreliable at the far left/right edges.
# All ship movement and enemy spawning is confined here.
ZONE_FRAC_L = 0.18    # left  boundary as a fraction of screen width
ZONE_FRAC_R = 0.82    # right boundary as a fraction of screen width

# Hand input remapping: the hand realistically spans only the inner
# portion of the webcam frame (not the literal edges).
# These fractions define the hand-position range that maps to
# [zone_l .. zone_r].  Anything outside is clamped to the edges.
HAND_REMAP_L = 0.20   # hand at 20 % of frame  → ship at zone left edge
HAND_REMAP_R = 0.80   # hand at 80 % of frame  → ship at zone right edge

# Landmark indices (MediaPipe 21-point)
LM_WRIST    = 0
LM_THUMB    = 4
LM_INDEX    = 8
LM_MID_TIP  = 12

HAND_CONN = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (9,10),(10,11),(11,12),
    (13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# ── BGR colour palette ──────────────────────────────
C_CYAN   = (255, 252,  18)   # ship / primary UI
C_BLUE   = (255,  85,   5)   # bullets
C_PINK   = (210,   0, 245)   # combo / elite
C_ORANGE = (  0, 140, 255)   # engine / explosion
C_YELLOW = (  0, 230, 255)   # score popups / warning
C_RED    = ( 15,  15, 252)   # basic enemy
C_RED2   = ( 60, 100, 255)   # elite enemy
C_GREEN  = ( 10, 240,  80)   # OK accents
C_WHITE  = (255, 255, 255)
C_PANEL  = (  8,  12,  22)   # HUD dark fill (navy)
C_BORDER = ( 60,  80, 120)   # panel border subtle
C_GRAY   = (110, 120, 138)
C_BLACK  = (  0,   0,   0)


# ═══════════════════════════════════════════════════════
#  KALMAN HAND TRACKER
# ═══════════════════════════════════════════════════════
class KalmanTracker:
    """
    4-state linear Kalman filter: state = [x, y, vx, vy].
    Provides smooth, predictive hand position even with brief occlusions.
    """
    def __init__(self):
        self.x  = np.zeros(4, dtype=float)
        self.P  = np.eye(4) * 600.0
        self.F  = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=float)
        self.H  = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
        self.Q  = np.diag([KFILT_Q_POS, KFILT_Q_POS, KFILT_Q_VEL, KFILT_Q_VEL])
        self.R  = np.eye(2) * KFILT_R
        self.ok = False

    def update(self, mx: float, my: float):
        """New measurement available — predict + correct."""
        z = np.array([mx, my])
        if not self.ok:
            self.x  = np.array([mx, my, 0.0, 0.0])
            self.ok = True
            return int(mx), int(my)
        xp = self.F @ self.x
        Pp = self.F @ self.P @ self.F.T + self.Q
        y  = z - self.H @ xp
        S  = self.H @ Pp @ self.H.T + self.R
        K  = Pp @ self.H.T @ np.linalg.inv(S)
        self.x = xp + K @ y
        self.P = (np.eye(4) - K @ self.H) @ Pp
        return int(round(self.x[0])), int(round(self.x[1]))

    def coast(self):
        """No measurement — propagate model, damp velocity."""
        self.x    = self.F @ self.x
        self.P    = self.F @ self.P @ self.F.T + self.Q
        self.x[2] *= 0.80   # velocity decay
        self.x[3] *= 0.80
        return int(round(self.x[0])), int(round(self.x[1]))

    def reset(self):
        self.P  = np.eye(4) * 600.0
        self.ok = False


# ═══════════════════════════════════════════════════════
#  PARALLAX STAR FIELD
# ═══════════════════════════════════════════════════════
class ParallaxStars:
    def __init__(self, w, h):
        self.w, self.h = w, h
        rng = np.random.default_rng(77)
        self.layers = []
        for (cnt, rmin, rmax, bmin, bmax, pf) in STAR_LAYERS:
            layer = []
            for _ in range(cnt):
                layer.append({
                    "x":  rng.uniform(0, w),
                    "y":  rng.uniform(0, h),
                    "r":  int(rng.integers(rmin, rmax + 1)),
                    "b":  int(rng.integers(bmin, bmax + 1)),
                    "pf": pf,
                    "tw": rng.uniform(0, math.pi * 2),
                    "ts": rng.uniform(0.025, 0.10),
                })
            self.layers.append(layer)
        self._px, self._py = float(w // 2), float(h // 2)

    def draw(self, frame, ship_x, ship_y, fi):
        dx = (ship_x - self._px) * -1
        dy = (ship_y - self._py) * -1
        self._px, self._py = float(ship_x), float(ship_y)
        for layer in self.layers:
            for s in layer:
                s["x"] = (s["x"] + dx * s["pf"]) % self.w
                s["y"] = (s["y"] + dy * s["pf"]) % self.h
                twinkle  = math.sin(fi * s["ts"] + s["tw"])
                br = int(max(40, min(255, s["b"] + twinkle * 35)))
                cv2.circle(frame, (int(s["x"]), int(s["y"])), s["r"], (br, br, br), -1)


# ═══════════════════════════════════════════════════════
#  GLOW / BLOOM SYSTEM
# ═══════════════════════════════════════════════════════
def apply_glow(frame: np.ndarray, glow_layer: np.ndarray):
    """Blur glow_layer and add-blend onto frame for neon bloom."""
    blurred = cv2.GaussianBlur(glow_layer, (GLOW_BLUR, GLOW_BLUR), 0)
    cv2.addWeighted(frame, 1.0, blurred, GLOW_STRENGTH, 0, frame)


# ═══════════════════════════════════════════════════════
#  DRAWING UTILITIES
# ═══════════════════════════════════════════════════════
def poly(canvas, pts, fill, outline=None, lw=2):
    arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(canvas, [arr], fill)
    if outline is not None:
        cv2.polylines(canvas, [arr], isClosed=True, color=outline, thickness=lw)


def shadow_text(frame, text, pos, scale, color, thickness=2):
    x, y = pos
    cv2.putText(frame, text, (x+2, y+2), FONT, scale, C_BLACK, thickness+2, cv2.LINE_AA)
    cv2.putText(frame, text, (x,   y),   FONT, scale, color,   thickness,   cv2.LINE_AA)


def centered_text(frame, text, cy, scale, color, thickness=2, sw=1280):
    tw = cv2.getTextSize(text, FONT, scale, thickness)[0][0]
    shadow_text(frame, text, ((sw - tw) // 2, cy), scale, color, thickness)


def draw_panel(frame, x1, y1, x2, y2,
               fill=C_PANEL, alpha=0.82,
               border=C_BORDER, bw=1, corner_len=10, accent=None):
    """Frosted-glass panel with corner bracket accents."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), fill, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    if border:
        cv2.rectangle(frame, (x1, y1), (x2, y2), border, bw)
    # Corner brackets
    ac = accent if accent else C_CYAN
    for (cx, cy, sx, sy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (cx, cy), (cx + sx*corner_len, cy), ac, 2)
        cv2.line(frame, (cx, cy), (cx, cy + sy*corner_len), ac, 2)


def draw_hand_skeleton(frame, lm_px, pinching: bool):
    """Draw the 21-point hand skeleton with pinch highlight."""
    line_col = (50, 220, 50) if not pinching else (180, 0, 240)
    dot_col  = C_WHITE
    for (a, b) in HAND_CONN:
        cv2.line(frame, lm_px[a], lm_px[b], line_col, 1, cv2.LINE_AA)
    for i, (x, y) in enumerate(lm_px):
        r = 5 if i in (LM_THUMB, LM_INDEX) else 3
        cv2.circle(frame, (x, y), r, dot_col, -1)
        cv2.circle(frame, (x, y), r, line_col, 1)
    # Highlight thumb and index tips during pinch
    if pinching:
        cv2.circle(frame, lm_px[LM_THUMB], 7, C_PINK, 2)
        cv2.circle(frame, lm_px[LM_INDEX], 7, C_PINK, 2)
        cv2.line(frame, lm_px[LM_THUMB], lm_px[LM_INDEX], C_PINK, 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════
#  SPACESHIP (premium multi-polygon design)
# ═══════════════════════════════════════════════════════
def _ship_pts(cx, cy, w=SHIP_W, h=SHIP_H):
    """Return all polygon groups for the ship."""
    hw, hh = w // 2, h // 2

    # Main hull (5-point)
    hull = [
        (cx,          cy - hh),          # nose
        (cx - w//5,   cy - h//8),        # left shoulder
        (cx - w//8,   cy + hh),          # left base
        (cx + w//8,   cy + hh),          # right base
        (cx + w//5,   cy - h//8),        # right shoulder
    ]
    # Left wing
    lwing = [
        (cx - w//5,   cy - h//8),
        (cx - hw,     cy + h//5),
        (cx - w*2//5, cy + hh),
        (cx - w//8,   cy + hh),
    ]
    # Right wing (mirror)
    rwing = [
        (cx + w//5,   cy - h//8),
        (cx + hw,     cy + h//5),
        (cx + w*2//5, cy + hh),
        (cx + w//8,   cy + hh),
    ]
    # Left wing tip accent
    ltip = [
        (cx - hw,       cy + h//5),
        (cx - hw - 8,   cy + h//4),
        (cx - w*2//5,   cy + hh),
    ]
    rtip = [
        (cx + hw,       cy + h//5),
        (cx + hw + 8,   cy + h//4),
        (cx + w*2//5,   cy + hh),
    ]
    return hull, lwing, rwing, ltip, rtip


def draw_ship(frame, glow, cx, cy, invincible=0):
    hull_c = C_WHITE if (invincible > 0 and invincible % 4 < 2) else C_CYAN
    dim_c  = tuple(max(0, int(c * 0.45)) for c in hull_c)
    acc_c  = tuple(min(255, int(c * 1.1)) for c in hull_c)

    hull, lwing, rwing, ltip, rtip = _ship_pts(cx, cy)

    # Engine flame (animated, two-tone)
    hw = SHIP_W // 2
    hh = SHIP_H // 2
    fl = random.randint(7, 20)
    flame_pts = [(cx - hw//3, cy+hh), (cx, cy+hh+fl), (cx + hw//3, cy+hh)]
    fl2 = random.randint(3, 11)
    flame2_pts = [(cx - hw//6, cy+hh), (cx, cy+hh+fl2), (cx + hw//6, cy+hh)]

    # Draw on glow layer first
    poly(glow, hull,  hull_c)
    poly(glow, lwing, dim_c)
    poly(glow, rwing, dim_c)
    poly(glow, flame_pts, C_ORANGE)

    # Draw crisp on frame
    poly(frame, flame_pts,  C_ORANGE)
    poly(frame, flame2_pts, C_YELLOW)
    poly(frame, lwing, dim_c,  outline=hull_c, lw=1)
    poly(frame, rwing, dim_c,  outline=hull_c, lw=1)
    poly(frame, ltip,  dim_c,  outline=acc_c,  lw=1)
    poly(frame, rtip,  dim_c,  outline=acc_c,  lw=1)
    poly(frame, hull,  hull_c, outline=C_WHITE, lw=2)

    # Cockpit
    cpx, cpy = cx, cy - SHIP_H // 4
    cv2.circle(glow,  (cpx, cpy), 8, C_WHITE, -1)
    cv2.circle(frame, (cpx, cpy), 6, C_WHITE, -1)
    cv2.circle(frame, (cpx, cpy), 4, hull_c,  -1)

    # Wing gun dots (glowing)
    for gx in (cx - SHIP_W*2//5 + 4, cx + SHIP_W*2//5 - 4):
        gy = cy + SHIP_H // 4
        cv2.circle(glow,  (gx, gy), 5, C_YELLOW, -1)
        cv2.circle(frame, (gx, gy), 3, C_YELLOW, -1)


# ═══════════════════════════════════════════════════════
#  BULLET
# ═══════════════════════════════════════════════════════
class Bullet:
    def __init__(self, cx, cy):
        self.x = cx - BULLET_W // 2
        self.y = cy - SHIP_H // 2 - BULLET_H

    def update(self):
        self.y -= BULLET_SPEED

    def off_screen(self):
        return self.y + BULLET_H < 0

    def draw(self, frame, glow):
        x1, y1 = int(self.x), int(self.y)
        x2, y2 = x1 + BULLET_W, y1 + BULLET_H
        # Glow core
        cv2.rectangle(glow,  (x1-2, y1-2), (x2+2, y2+2), C_BLUE,  -1)
        # Crisp
        cv2.rectangle(frame, (x1, y1), (x2, y2),     C_BLUE,  -1)
        cv2.rectangle(frame, (x1+1, y1+2), (x2-1, y2-3), C_WHITE, -1)

    def rect(self):
        return (int(self.x), int(self.y), BULLET_W, BULLET_H)


# ═══════════════════════════════════════════════════════
#  ENEMY
# ═══════════════════════════════════════════════════════
class Enemy:
    def __init__(self, x, sw, sh, speed, drift_range, zone_l=0, zone_r=None, elite=False):
        self.cx     = float(x)
        self.cy     = float(-ENEMY_SIZE - 4)
        self.sw     = sw
        self.sh     = sh
        self.zone_l = zone_l
        self.zone_r = zone_r if zone_r is not None else sw
        self.speed  = speed
        self.drift  = random.uniform(-drift_range, drift_range)
        self.elite  = elite
        self.size   = int(ENEMY_SIZE * (1.5 if elite else 1.0))
        self.hp     = 2 if elite else 1

    def update(self):
        self.cy += self.speed
        self.cx += self.drift
        # Bounce off zone walls (not screen edges)
        hs = self.size // 2
        if self.cx - hs < self.zone_l:
            self.cx = float(self.zone_l + hs)
            self.drift = abs(self.drift)   # push right
        elif self.cx + hs > self.zone_r:
            self.cx = float(self.zone_r - hs)
            self.drift = -abs(self.drift)  # push left

    def reached_bottom(self):
        return self.cy + self.size // 2 >= self.sh

    def draw(self, frame, glow, level=1):
        cx, cy = int(self.cx), int(self.cy)
        hs = self.size // 2

        if self.elite:
            self._draw_elite(frame, glow, cx, cy, hs, level)
        else:
            self._draw_scout(frame, glow, cx, cy, hs, level)

    def _scout_color(self, level):
        """Color shifts from red toward orange-pink as levels rise."""
        t = min(1.0, (level - 1) / 12.0)
        b = int(15  + t * 60)
        g = int(15  + t * 55)
        r = int(252 - t * 20)
        return (b, g, r)

    def _draw_scout(self, frame, glow, cx, cy, hs, level):
        col = self._scout_color(level)
        dim = tuple(max(0, int(c * 0.5)) for c in col)

        # Body: downward triangle
        body = [(cx, cy + hs), (cx - int(hs*0.7), cy - int(hs*0.5)), (cx + int(hs*0.7), cy - int(hs*0.5))]
        # Side fins
        lfin = [(cx - int(hs*0.7), cy - int(hs*0.5)), (cx - hs, cy - int(hs*0.8)), (cx - int(hs*0.3), cy)]
        rfin = [(cx + int(hs*0.7), cy - int(hs*0.5)), (cx + hs, cy - int(hs*0.8)), (cx + int(hs*0.3), cy)]
        # Top plate
        top  = [(cx - int(hs*0.7), cy - int(hs*0.5)), (cx + int(hs*0.7), cy - int(hs*0.5)), (cx, cy - int(hs*0.9))]

        poly(glow,  body, col)
        poly(frame, lfin, dim, outline=col, lw=1)
        poly(frame, rfin, dim, outline=col, lw=1)
        poly(frame, top,  dim, outline=col, lw=1)
        poly(frame, body, col, outline=(0, 50, 200), lw=2)

        # Cockpit dot
        cv2.circle(glow,  (cx, cy - hs//3), 6, C_WHITE, -1)
        cv2.circle(frame, (cx, cy - hs//3), 4, C_WHITE, -1)
        cv2.circle(frame, (cx, cy - hs//3), 2, col,     -1)

    def _draw_elite(self, frame, glow, cx, cy, hs, level):
        col = C_RED2
        acc = C_PINK

        # Diamond body
        diamond = [(cx, cy - hs), (cx + hs, cy), (cx, cy + hs), (cx - hs, cy)]
        # Inner diamond
        ins = hs * 2 // 3
        inner   = [(cx, cy-ins), (cx+ins, cy), (cx, cy+ins), (cx-ins, cy)]

        poly(glow,  diamond, col)
        poly(frame, diamond, col,     outline=acc, lw=2)
        poly(frame, inner,   (40,0,180), outline=C_WHITE, lw=1)

        # Cardinal spikes
        for (dx, dy) in [(0,-hs-6),(0,hs+6),(-hs-6,0),(hs+6,0)]:
            spike = [(cx+dx-4, cy+dy-4), (cx+dx, cy+dy+8), (cx+dx+4, cy+dy-4)]
            poly(frame, spike, acc)

        cv2.circle(glow,  (cx, cy), 8, C_WHITE, -1)
        cv2.circle(frame, (cx, cy), 5, C_WHITE, -1)
        cv2.circle(frame, (cx, cy), 3, acc,     -1)

    def rect(self):
        cx, cy = int(self.cx), int(self.cy)
        hs = self.size // 2
        return (cx - hs, cy - hs, self.size, self.size)


# ═══════════════════════════════════════════════════════
#  EXPLOSION  (layered particle system)
# ═══════════════════════════════════════════════════════
class Explosion:
    def __init__(self, cx, cy, big=False):
        self.cx      = float(cx)
        self.cy      = float(cy)
        self.life    = 0
        self.maxlife = EXP_LIFE + (8 if big else 0)
        self.maxr    = EXP_R_MAX * (1.7 if big else 1.0)
        self.big     = big
        # Debris particles
        n = 14 if big else 8
        self.debris = [
            {
                "vx": random.uniform(-4, 4),
                "vy": random.uniform(-5, 1),
                "x":  cx, "y": cy,
                "r":  random.randint(2, 4),
                "c":  random.choice([C_ORANGE, C_YELLOW, C_WHITE]),
            }
            for _ in range(n)
        ]

    def update(self):
        self.life += 1
        for d in self.debris:
            d["x"] += d["vx"]
            d["y"] += d["vy"]
            d["vy"] += 0.3   # gravity
            d["vx"] *= 0.92

    def done(self):
        return self.life >= self.maxlife

    def draw(self, frame, glow):
        t      = self.life / self.maxlife
        r      = int(t * self.maxr)
        cx, cy = int(self.cx), int(self.cy)

        # Expanding outer ring
        if r > 0:
            cv2.circle(glow,  (cx, cy), r,        C_ORANGE, 4)
            cv2.circle(frame, (cx, cy), r,        C_ORANGE, 3)

        # Inner fill
        if r > 6:
            inner = max(1, r - 8)
            cv2.circle(glow,  (cx, cy), inner, C_YELLOW, -1)
            cv2.circle(frame, (cx, cy), inner, C_YELLOW, -1)

        # Rotating spokes
        for i in range(8):
            angle  = (math.pi / 4) * i + t * math.pi
            px = int(cx + r * 0.72 * math.cos(angle))
            py = int(cy + r * 0.72 * math.sin(angle))
            cv2.circle(glow,  (px, py), 4, C_YELLOW, -1)
            cv2.circle(frame, (px, py), 2, C_WHITE,  -1)

        # Debris particles
        alpha = 1.0 - t
        for d in self.debris:
            dr = max(1, int(d["r"] * alpha))
            px, py = int(d["x"]), int(d["y"])
            cv2.circle(glow,  (px, py), dr + 2, d["c"], -1)
            cv2.circle(frame, (px, py), dr,     d["c"], -1)


# ═══════════════════════════════════════════════════════
#  SCORE POPUP
# ═══════════════════════════════════════════════════════
class ScorePopup:
    def __init__(self, cx, cy, text, color=C_YELLOW):
        self.x    = float(cx)
        self.y    = float(cy)
        self.text = text
        self.col  = color
        self.life = 0
        self.MAX  = 45

    def update(self):
        self.y   -= 1.4
        self.life += 1

    def done(self):
        return self.life >= self.MAX

    def draw(self, frame):
        alpha = 1.0 - self.life / self.MAX
        scale = 0.60 + 0.25 * (1.0 - alpha)
        col   = tuple(int(c * alpha) for c in self.col)
        cv2.putText(frame, self.text,
                    (int(self.x) - 18, int(self.y)),
                    FONT, scale, col, 2, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════
#  COMBO TRACKER
# ═══════════════════════════════════════════════════════
class ComboTracker:
    def __init__(self):
        self.combo       = 1
        self.frames_since = 0

    def on_kill(self):
        if self.frames_since <= COMBO_WIN:
            self.combo = min(COMBO_CAP, self.combo + 1)
        else:
            self.combo = 1
        self.frames_since = 0
        return self.combo

    def tick(self):
        self.frames_since += 1
        if self.frames_since > COMBO_WIN + 5:
            self.combo = 1

    def reset(self):
        self.combo = 1
        self.frames_since = 999


# ═══════════════════════════════════════════════════════
#  COLLISION
# ═══════════════════════════════════════════════════════
def rects_overlap(r1, r2):
    x1,y1,w1,h1 = r1
    x2,y2,w2,h2 = r2
    return not (x1+w1 < x2 or x2+w2 < x1 or y1+h1 < y2 or y2+h2 < y1)


# ═══════════════════════════════════════════════════════
#  DIFFICULTY
# ═══════════════════════════════════════════════════════
def get_level(kills):      return kills // KILLS_PER_LVL + 1
def get_spd(lvl):          return min(MAX_SPD, BASE_SPD + (lvl-1)*0.65)
def get_spawn(lvl):        return max(MIN_SPAWN, BASE_SPAWN - (lvl-1)*3)
def get_drift(lvl):        return min(2.8, 0.45 + (lvl-1)*0.28)
def get_elite_every(lvl):  return max(3, 6 - (lvl-1)//3)   # elite every N spawns


# ═══════════════════════════════════════════════════════
#  HUD DRAWING
# ═══════════════════════════════════════════════════════
def draw_hud(frame, state, sw, sh, hs):
    score  = state["score"]
    lives  = state["lives"]
    lvl    = state["level"]
    kills  = state["total_kills"]
    elapsed= time.time() - state["start_time"]
    combo  = state["combo"].combo

    # ── LEFT PANEL: score + high score ───────────
    draw_panel(frame, 8, 8, 220, 96, accent=C_CYAN)
    shadow_text(frame, f"SCORE",         (22, 36), FS_S, C_GRAY, 1)
    shadow_text(frame, f"{score:,}",     (22, 72), FS_L, C_WHITE, 2)
    hs_col = C_YELLOW if score >= hs and score > 0 else C_GRAY
    shadow_text(frame, f"BEST {hs:,}",  (22, 92), FS_S, hs_col, 1)

    # ── CENTRE TOP: level + progress bar ─────────
    lvl_c = C_CYAN if lvl < 5 else (C_YELLOW if lvl < 9 else C_PINK)
    lvl_text = f"LEVEL  {lvl}"
    centered_text(frame, lvl_text, 38, FS_L, lvl_c, thickness=2, sw=sw)

    # Progress bar
    bw, bh = 180, 7
    bx = sw//2 - bw//2
    by = 52
    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (30, 35, 55), -1)
    fill = int(bw * (kills % KILLS_PER_LVL) / KILLS_PER_LVL)
    if fill > 0:
        cv2.rectangle(frame, (bx, by), (bx+fill, by+bh), lvl_c, -1)
    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), C_BORDER, 1)

    # ── RIGHT PANEL: lives ────────────────────────
    lw = 52 + max(0, lives) * 26
    lx = sw - lw - 8
    draw_panel(frame, lx, 8, sw - 8, 68, accent=C_CYAN)
    shadow_text(frame, "LIVES", (lx + 10, 34), FS_S, C_GRAY, 1)
    for i in range(max(0, lives)):
        ox = lx + 18 + i * 26
        oy = 56
        poly(frame, [(ox, oy-14), (ox-9, oy), (ox+9, oy)], C_CYAN, outline=C_WHITE, lw=1)

    # ── COMBO badge ───────────────────────────────
    if combo > 1:
        frames_left = COMBO_WIN - state["combo"].frames_since
        t_fade = max(0.0, min(1.0, frames_left / (COMBO_WIN * 0.35)))
        col = tuple(int(c * t_fade) for c in C_PINK)
        combo_txt = f"x{combo}  COMBO"
        tw = cv2.getTextSize(combo_txt, FONT, FS_M, 2)[0][0]
        cx2 = (sw - tw) // 2
        shadow_text(frame, combo_txt, (cx2, 88), FS_M, col, 2)

    # ── Instruction (first 6 s) ───────────────────
    if elapsed < INSTR_SECS:
        centered_text(frame, "PINCH  TO  SHOOT", sh - 22, FS_S, C_GRAY, 1, sw)

    # ── No-hand warning ───────────────────────────
    if not state.get("hand_ok", True):
        coast = state.get("coast_frames", 0)
        if coast <= 0:
            msg = "SHOW YOUR HAND"
            tw  = cv2.getTextSize(msg, FONT, FS_L, 3)[0][0]
            shadow_text(frame, msg, ((sw-tw)//2, sh//2), FS_L, C_YELLOW, 3)

    # ── Level-up banner ───────────────────────────
    if state["banner"] > 0:
        t  = state["banner"] / BANNER_F
        alpha = min(1.0, t*3) if t < 0.33 else (1.0 if t < 0.67 else (t-0.67)*3)
        alpha = max(0.0, alpha)
        col_b = tuple(int(c * alpha) for c in C_YELLOW)
        col_s = tuple(int(c * alpha) for c in lvl_c)
        centered_text(frame, f"LEVEL  {lvl}", sh//2 - 36, FS_XL*1.1, col_b, 3, sw)
        centered_text(frame, "LEVEL  UP !",    sh//2 + 28, FS_M,      col_s, 2, sw)


def draw_game_over(frame, state, sw, sh, hs):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (sw,sh), (4,6,14), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)

    score   = state["score"]
    lvl     = state["level"]
    new_hs  = score > 0 and score >= hs
    final_hs= max(score, hs)

    # Central panel
    pw, ph = 440, 320
    px, py = (sw-pw)//2, (sh-ph)//2
    draw_panel(frame, px, py, px+pw, py+ph, fill=(8,10,20), alpha=0.92,
               border=C_CYAN, bw=2, corner_len=16, accent=C_CYAN)

    y = py + 52
    centered_text(frame, "GAME  OVER",          y, FS_XL, C_RED,   3, sw); y += 60
    centered_text(frame, f"SCORE   {score:,}",  y, FS_L,  C_WHITE, 2, sw); y += 44
    centered_text(frame, f"LEVEL REACHED  {lvl}",y, FS_M, C_CYAN,  2, sw); y += 32

    if new_hs:
        centered_text(frame, "NEW HIGH SCORE!", y, FS_M, C_YELLOW, 2, sw); y += 30
    centered_text(frame, f"BEST   {final_hs:,}",y, FS_M,
                  C_YELLOW if new_hs else C_GRAY, 2, sw); y += 44

    centered_text(frame, "R  to Restart",  y,     FS_S, C_GRAY, 1, sw); y += 24
    centered_text(frame, "Q  to Quit",     y,     FS_S, C_GRAY, 1, sw)


# ═══════════════════════════════════════════════════════
#  SCREEN SHAKE
# ═══════════════════════════════════════════════════════
def apply_shake(frame, intensity):
    if intensity <= 0:
        return frame
    dx = random.randint(-intensity, intensity)
    dy = random.randint(-intensity, intensity)
    M  = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]),
                          borderMode=cv2.BORDER_REFLECT_101)


# ═══════════════════════════════════════════════════════
#  GAME STATE
# ═══════════════════════════════════════════════════════
def new_state(sw, sh):
    return {
        "score":       0,
        "lives":       STARTING_LIVES,
        "total_kills": 0,
        "level":       1,
        "bullets":     [],
        "enemies":     [],
        "explosions":  [],
        "popups":      [],
        "combo":       ComboTracker(),
        "ship_x":      sw // 2,
        "ship_y":      sh // 2,
        "fi":          0,
        "pinch_cd":    0,
        "invincible":  0,
        "shake":       0,
        "banner":      0,
        "game_over":   False,
        "start_time":  time.time(),
        "hand_ok":     False,
        "coast_frames":0,
        "spawn_count": 0,
    }


# ═══════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════
def main():
    print("=" * 56)
    print("   Hand Gesture Space Shooter  |  Neon Horizon v3")
    print("=" * 56)
    print("   Move hand       -- steer spaceship")
    print("   Pinch gesture   -- shoot")
    print("   Q / ESC         -- quit    |    R -- restart")
    print("=" * 56)

    if _AUDIO_OK:
        build_sfx()
        print("   Sound:  ON")
    else:
        print("   Sound:  OFF  (pip install sounddevice)")

    hs = load_hs()
    print(f"   Best score:  {hs}")
    print("=" * 56)

    ensure_model()

    # ── Camera opens FIRST so it warms up while MediaPipe loads ──
    # CAP_DSHOW avoids Windows MSMF locking issues
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: webcam unavailable.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    time.sleep(1.0)   # let the DSHOW pipeline warm up

    # ── MediaPipe ─────────────────────────────────
    base_opts  = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    lm_opts    = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(lm_opts)

    # Probe first valid frame (camera may need a moment after MediaPipe init)
    probe = None
    for _ in range(60):
        ret, probe = cap.read()
        if ret and probe is not None:
            break
        time.sleep(0.05)
    if probe is None:
        print("ERROR: cannot read camera — try closing other apps using the webcam.")
        cap.release()
        return

    sh, sw = probe.shape[:2]
    print(f"   Camera:  {sw}x{sh}")

    # ── One-time init ─────────────────────────────
    state      = new_state(sw, sh)
    tracker    = KalmanTracker()
    stars      = ParallaxStars(sw, sh)
    dark_bg    = np.zeros((sh, sw, 3), dtype=np.uint8)
    glow_layer = np.zeros((sh, sw, 3), dtype=np.uint8)
    ts_ms      = 0

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, sw, sh)

    # ── MAIN LOOP ────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # 1. Darken webcam for space feel
        cv2.addWeighted(frame, OVERLAY_ALPHA, dark_bg, 1-OVERLAY_ALPHA, 0, frame)

        # 2. Subtle level-based background tint
        lvl = state["level"]
        tint_b = min(30, (lvl-1) * 2)
        tint_r = min(20, (lvl-1) * 1)
        if tint_b or tint_r:
            tint = np.full((sh, sw, 3), (tint_b, 0, tint_r), dtype=np.uint8)
            cv2.addWeighted(frame, 1.0, tint, 0.12, 0, frame)

        # 3. Parallax stars
        stars.draw(frame, state["ship_x"], state["ship_y"], state["fi"])

        # 4. Clear glow layer
        glow_layer[:] = 0

        # 5. MediaPipe hand detection
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms   += WAIT_MS
        result   = landmarker.detect_for_video(mp_img, ts_ms)

        hand_ok      = False
        pinch_now    = False
        pinch_norm_v = 1.0   # will store normalized pinch distance for indicator

        if result.hand_landmarks:
            hand_ok = True
            state["coast_frames"] = COAST_FRAMES
            lm = result.hand_landmarks[0]

            # Raw index fingertip position
            raw_x = lm[LM_INDEX].x * sw
            raw_y = lm[LM_INDEX].y * sh

            # Remap hand X → active zone
            # Clip the expected hand range (HAND_REMAP_L..HAND_REMAP_R)
            # so partial hand sweeps still reach zone edges.
            zone_l = int(sw * ZONE_FRAC_L)
            zone_r = int(sw * ZONE_FRAC_R)
            zone_w = zone_r - zone_l
            hand_frac = (lm[LM_INDEX].x - HAND_REMAP_L) / (HAND_REMAP_R - HAND_REMAP_L)
            hand_frac = max(0.0, min(1.0, hand_frac))   # clamp to [0..1]
            mapped_x  = zone_l + hand_frac * zone_w

            # Kalman update
            fx, fy = tracker.update(mapped_x, raw_y)
            fx = max(zone_l + SHIP_W, min(zone_r - SHIP_W, fx))
            fy = max(SHIP_H, min(sh - SHIP_H, fy))
            state["ship_x"] = fx
            state["ship_y"] = fy

            # Normalized pinch detection
            wrist = np.array([lm[LM_WRIST].x * sw, lm[LM_WRIST].y * sh])
            mid   = np.array([lm[LM_MID_TIP].x * sw, lm[LM_MID_TIP].y * sh])
            hand_span = max(1.0, float(np.linalg.norm(mid - wrist)))

            thumb = np.array([lm[LM_THUMB].x * sw, lm[LM_THUMB].y * sh])
            idx   = np.array([lm[LM_INDEX].x * sw, lm[LM_INDEX].y * sh])
            pinch_dist = float(np.linalg.norm(thumb - idx))
            pinch_norm_v = pinch_dist / hand_span
            pinch_now  = pinch_norm_v < PINCH_NORM_THR

            # Draw hand skeleton
            lm_px = [(int(p.x*sw), int(p.y*sh)) for p in lm]
            draw_hand_skeleton(frame, lm_px, pinch_now)

        else:
            # Coast with Kalman prediction
            if state["coast_frames"] > 0:
                state["coast_frames"] -= 1
                zone_l = int(sw * ZONE_FRAC_L)
                zone_r = int(sw * ZONE_FRAC_R)
                fx, fy = tracker.coast()
                fx = max(zone_l + SHIP_W, min(zone_r - SHIP_W, fx))
                fy = max(SHIP_H, min(sh - SHIP_H, fy))
                state["ship_x"] = fx
                state["ship_y"] = fy

        state["hand_ok"] = hand_ok

        # ── GAME LOGIC ──────────────────────────
        if not state["game_over"]:
            fi   = state["fi"]
            e_sp = get_spd(lvl)
            e_sr = get_spawn(lvl)
            e_dr = get_drift(lvl)
            e_el = get_elite_every(lvl)

            # Timers
            if state["invincible"] > 0: state["invincible"] -= 1
            if state["shake"] > 0:      state["shake"]      -= 1
            if state["banner"] > 0:     state["banner"]     -= 1
            state["combo"].tick()

            # Shoot
            if state["pinch_cd"] > 0:
                state["pinch_cd"] -= 1
            if pinch_now and state["pinch_cd"] == 0:
                state["bullets"].append(Bullet(state["ship_x"], state["ship_y"]))
                state["pinch_cd"] = PINCH_COOLDOWN
                sfx("shoot")

            # Spawn enemies — restrict to active zone
            if fi % e_sr == 0:
                zone_l = int(sw * ZONE_FRAC_L)
                zone_r = int(sw * ZONE_FRAC_R)
                ex = random.randint(zone_l + ENEMY_SIZE, zone_r - ENEMY_SIZE)
                state["spawn_count"] += 1
                is_elite = (state["spawn_count"] % e_el == 0)
                state["enemies"].append(
                    Enemy(ex, sw, sh, e_sp, e_dr,
                          zone_l=zone_l, zone_r=zone_r, elite=is_elite))

            # Update bullets
            for b in state["bullets"]: b.update()
            state["bullets"] = [b for b in state["bullets"] if not b.off_screen()]

            # Update enemies
            for e in state["enemies"]: e.update()

            # Enemies reaching bottom
            lost  = [e for e in state["enemies"] if e.reached_bottom()]
            alive = [e for e in state["enemies"] if not e.reached_bottom()]
            if lost and state["invincible"] == 0:
                state["lives"]     -= 1
                state["invincible"] = 65
                state["shake"]      = SHAKE_DUR
                state["combo"].reset()
                sfx("life_lost")
                for e in lost:
                    state["explosions"].append(Explosion(e.cx, sh - 12, big=True))
            state["enemies"] = alive

            if state["lives"] <= 0:
                state["game_over"] = True
                save_hs(state["score"])
                hs = load_hs()
                sfx("game_over")

            # Collision: bullets vs enemies
            kept_e, kept_b = [], list(state["bullets"])
            for e in state["enemies"]:
                hit = False
                for b in kept_b[:]:
                    if rects_overlap(b.rect(), e.rect()):
                        kept_b.remove(b)
                        e.hp -= 1
                        if e.hp <= 0:
                            state["explosions"].append(Explosion(e.cx, e.cy, big=e.elite))
                            combo  = state["combo"].on_kill()
                            base   = 3 if e.elite else 1
                            bonus  = 1 + (lvl - 1) // 3
                            pts    = base * bonus * combo
                            state["score"]       += pts
                            state["total_kills"] += 1
                            col_p = C_PINK if combo > 2 else C_YELLOW
                            label = f"+{pts}" if combo < 2 else f"+{pts} x{combo}"
                            state["popups"].append(ScorePopup(e.cx, e.cy, label, col_p))
                            sfx("explode")
                            hit = True
                            # Level-up
                            new_lvl = get_level(state["total_kills"])
                            if new_lvl > state["level"]:
                                state["level"]  = new_lvl
                                state["banner"] = BANNER_F
                                sfx("level_up")
                        else:
                            # Elite took a hit but survived
                            state["popups"].append(ScorePopup(e.cx, e.cy, "HIT", C_ORANGE))
                            hit = True  # bullet consumed
                        break
                if not hit or e.hp > 0:
                    if e.hp > 0: kept_e.append(e)
            state["enemies"] = kept_e
            state["bullets"] = kept_b

            # Update effects
            for ex in state["explosions"]: ex.update()
            state["explosions"] = [ex for ex in state["explosions"] if not ex.done()]
            for p in state["popups"]: p.update()
            state["popups"] = [p for p in state["popups"] if not p.done()]

            # ── DRAW ACTIVE-ZONE BOUNDARIES ──────
            zone_l = int(sw * ZONE_FRAC_L)
            zone_r = int(sw * ZONE_FRAC_R)
            # Dim the out-of-zone columns
            frame[:, :zone_l] = (frame[:, :zone_l].astype(np.float32) * 0.45).astype(np.uint8)
            frame[:, zone_r:] = (frame[:, zone_r:].astype(np.float32) * 0.45).astype(np.uint8)
            # Subtle boundary lines
            for lx in (zone_l, zone_r):
                cv2.line(frame, (lx, 0), (lx, sh), (80, 110, 180), 1)

            # ── DRAW GAME ELEMENTS ───────────────
            for e in state["enemies"]:
                e.draw(frame, glow_layer, lvl)
            for b in state["bullets"]:
                b.draw(frame, glow_layer)
            draw_ship(frame, glow_layer,
                      state["ship_x"], state["ship_y"], state["invincible"])
            for ex in state["explosions"]:
                ex.draw(frame, glow_layer)

            # Apply bloom
            apply_glow(frame, glow_layer)

            # Score popups (drawn after glow, always readable)
            for p in state["popups"]: p.draw(frame)

            # Pinch proximity indicator on ship
            if hand_ok:
                frac = max(0.0, 1.0 - pinch_norm_v / PINCH_NORM_THR)
                ring_c = C_PINK if pinch_now else C_GRAY
                ring_r = SHIP_W // 2 + 12
                cv2.circle(frame, (state["ship_x"], state["ship_y"]),
                           ring_r, ring_c,
                           2 if pinch_now else 1)
                # Arc fill showing closeness to pinch
                if frac > 0.1:
                    sweep = int(360 * frac)
                    cv2.ellipse(frame,
                                (state["ship_x"], state["ship_y"]),
                                (ring_r, ring_r), -90, 0, sweep,
                                C_CYAN, 2)

            # HUD
            draw_hud(frame, state, sw, sh, hs)
            state["fi"] += 1

        else:
            # Still draw glow pass on empty layer for background beauty
            apply_glow(frame, glow_layer)
            draw_game_over(frame, state, sw, sh, hs)

        # Screen shake
        shake_i = int(SHAKE_MAG * state["shake"] / SHAKE_DUR)
        if shake_i > 0:
            frame = apply_shake(frame, shake_i)

        # Display
        cv2.imshow(WINDOW_TITLE, frame)

        key = cv2.waitKey(WAIT_MS) & 0xFF
        if key in (ord("q"), 27):
            save_hs(state["score"])
            print(f"   Quit.  Score: {state['score']}  |  Best: {load_hs()}")
            break
        elif key == ord("r"):
            save_hs(state["score"])
            hs    = load_hs()
            state = new_state(sw, sh)
            tracker.reset()
            print("   Restarted.")

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
