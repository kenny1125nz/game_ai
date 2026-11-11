# OpenCV Game Bot

## Overview

A simple game automation bot that reads the screen, makes rule-based decisions, and sends inputs to the game.

## Core Components

### 1. Screen Reader
- Captures game screen using OpenCV
- Extracts visual information (colors, shapes, positions)

### 2. Decision Engine
- Rule-based logic to determine actions
- Example: "If health bar is red, use healing item"

### 3. Input Controller
- Sends simulated mouse clicks and keyboard presses to the game

## Flow

```
Screen Capture → Rule Processing → Input Action
    (OpenCV)       (If/Then)      (Mouse/Keyboard)
```

## Project Structure

```
game_ai/
├── readme.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── screen_reader.py
│   ├── decision_engine.py
│   └── input_controller.py
└── templates/
```

## Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Run

```bash
python src/main.py
```

## License

MIT
