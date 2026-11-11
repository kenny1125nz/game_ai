"""Main entry point for the game bot."""

import time
import traceback
from d2r.controller import execute_multi
from common.screen_reader import ScreenReader
from d2r.game import D2RGame
import sys
from typing import Any, Tuple

from d2r.cnn_rune_detector import CNNRuneDetector
from d2r.data.data_capture import DataCapture
from ocr.d2r_ocr import D2rOCR

from display.overlay2 import Overlay


def create_rune_detector(
    model_size: str, threshold=0.65, calibrated: bool = False, offset=(0, 0)
):
    capture = DataCapture("D2R", "./data/")

    overlay = Overlay(offset)
    detector = CNNRuneDetector(
        model_size=model_size,
        threshold=threshold,
        calibrated=calibrated,
        data_capture=capture.save,
    )

    def rune_detector(
        screen_data: Any, scale: tuple[float, float], highrune_min_idx=16
    ) -> list[Tuple]:
        """Detect runes from screen data."""
        return detector.find_runes(
            screen_data, scale, highrune_min_idx=highrune_min_idx
        )

    return rune_detector


def main():
    capture = DataCapture("ocr", "./data/")

    model_size = sys.argv[1]
    calibrated = len(sys.argv) > 2 and sys.argv[2] == "calibrated"

    # Initialize screen reader for Diablo II: Resurrected
    screen = ScreenReader("Diablo II: Resurrected")
    print(f"Window found at: {screen.position}, size: {screen.resolution}")

    x, y = screen.position
    w, h = screen.resolution
    scale_factor = (w / 1920, h / 1080)
    screen_size = "1k"
    if w == 3840:
        screen_size = "4k"

    detector = create_rune_detector(model_size, 0.65, calibrated, offset=(x, y))
    game = D2RGame(detector)
    print(f"Initialized {game.title} game instance")

    ocr = D2rOCR("./models/d2r_ocr.onnx", capture=capture)

    # Main loop
    frame_count = 0
    state = None
    stage_old = None
    while True:
        try:
            screen_data = screen.capture()
            frame_count += 1

            if state is not None:
                stage_old = state.game_stage
            state = game.parse_screen(screen_data, scale=scale_factor)

            commands = game.make_decision(state)
            execute_multi(commands, x, y)

            # Status update every 100 frames
            if frame_count % 100 == 0:
                print(f"Still running... Press ESC to exit")

            if not stage_old == state.game_stage:
                print(f"Game Stage:{state.game_stage}")

            detected = ocr.read_screen(screen_data)
            if len(detected) > 0:
                words = list(map(lambda x: x[0], detected))
                print(words)

            time.sleep(1)
        except KeyboardInterrupt:
            print("\nInterrupted by user, exiting...")
            break
        except Exception as e:
            traceback.print_exc()
            print(f"Error in capture/processing loop: {e}")
            time.sleep(1)  # Wait before retry


if __name__ == "__main__":
    main()
