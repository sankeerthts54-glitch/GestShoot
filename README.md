# Hand Gesture Space Shooter 🚀 - Neon Horizon v3.1

Real-time space shooter controlled entirely by hand gestures using MediaPipe and OpenCV.
No game engine — just pure webcam + computer vision magic.

## How to Play

| Gesture | Action |
|---|---|
| ✋ Move hand | Steer the spaceship |
| 🤏 Pinch thumb + index finger | Shoot bullets |
| R key | Restart game |
| Q or ESC | Quit |

## Rules
- Destroy enemies before they reach the bottom of the screen
- You start with **5 lives** — each enemy that gets past costs 1 life
- Build **combos** by chaining rapid kills for bonus points
- Game over when all lives are lost
- Beat your high score!

## Setup

```bash
# Install dependencies
pip install mediapipe opencv-python numpy sounddevice

# Run the game
python game.py
```

## Requirements
- Python 3.12+
- Webcam (built-in or USB)
- Good lighting for best hand tracking

## What's New in v3.1
- **Centre-Zone Gameplay**: Action is restricted to the middle 64% of the screen for reliable hand recognition. Hand movements are remapped seamlessly so you don't need to reach the physical edges of your webcam.
- **Kalman Filter Tracking**: Snappier, smooth, predictive hand tracking that removes jitter.
- **Dynamic Graphics**: Neon glow rendering system (single-pass Gaussian bloom), 3-layer parallax star field, frosted-glass HUD, and animated multi-polygon ships.
- **Sound Engine**: Synthesized retro sound effects using `sounddevice` and `numpy`.
- **Advanced Combat**: Pinch detection normalized to hand distance, combo multiplier system, and Elite enemy types.
- **Game Feel Enhancements**: Screen shake, level up banners, and dynamic background tinting.

## Tech Stack

| Library | Purpose |
|---|---|
| **MediaPipe** | Real-time hand landmark detection (21 points) |
| **OpenCV** | Webcam capture, rendering, drawing |
| **NumPy** | Coordinate math, Kalman filtering, Audio synthesis |
| **SoundDevice** | Audio playback |

## Controls At a Glance

```
PINCH to shoot     ─── bring thumb and index finger close together
MOVE HAND          ─── index finger tip controls the ship
R                  ─── restart
Q / ESC            ─── quit
```
