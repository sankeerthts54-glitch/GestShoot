# Hand Gesture Space Shooter 🚀

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
- You start with **3 lives** — each enemy that gets past costs 1 life
- Game over when all lives are lost
- Beat your high score!

## Setup

```bash
# Install dependencies
pip install mediapipe opencv-python numpy

# Run the game
python game.py
```

## Requirements
- Python 3.12+
- Webcam (built-in or USB)
- Good lighting for best hand tracking

## Tech Stack

| Library | Purpose |
|---|---|
| **MediaPipe** | Real-time hand landmark detection (21 points) |
| **OpenCV** | Webcam capture, rendering, drawing |
| **NumPy** | Coordinate math |

## Features
- 🎯 Real-time hand tracking via MediaPipe (21-point landmark model)
- 🌟 Space atmosphere: darkened webcam feed + randomised star field
- 💥 Particle explosion effects on enemy destruction
- 🛸 Smooth spaceship movement with position interpolation
- ⏱ 20-frame shot cooldown (no spam)
- 📊 Live score & lives HUD
- 🔄 Instant restart with R key

## Controls At a Glance

```
PINCH to shoot     ─── bring thumb and index finger close together
MOVE HAND          ─── index finger tip controls the ship
R                  ─── restart
Q / ESC            ─── quit
```
