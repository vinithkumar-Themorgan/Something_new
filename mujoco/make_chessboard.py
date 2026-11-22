"""Generate a printable chessboard for camera calibration.

Usage:  python make_chessboard.py
Output: chessboard_9x6_25mm.png (next to this script)

Print at 100% scale (no 'fit to page'), then measure one black square with a ruler
and set SQUARE_MM in calibrate_chessboard.py to the measured value.
"""
from pathlib import Path
import cv2
import numpy as np

INNER_COLS, INNER_ROWS = 9, 6      # inner corners (squares = inner + 1)
SQUARE_MM = 25.0                   # nominal square size; measure the printed one
DPI = 300
MARGIN_MM = 10.0
OUT = Path(__file__).resolve().parent / f'chessboard_{INNER_COLS}x{INNER_ROWS}_{int(SQUARE_MM)}mm.png'


def main():
    px_per_mm = DPI / 25.4
    sq = int(round(SQUARE_MM * px_per_mm))
    margin = int(round(MARGIN_MM * px_per_mm))
    cols, rows = INNER_COLS + 1, INNER_ROWS + 1
    board = np.zeros((rows * sq, cols * sq), np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                board[r * sq:(r + 1) * sq, c * sq:(c + 1) * sq] = 255
    img = np.full((board.shape[0] + 2 * margin, board.shape[1] + 2 * margin), 255, np.uint8)
    img[margin:margin + board.shape[0], margin:margin + board.shape[1]] = board
    cv2.imwrite(str(OUT), img)
    w_mm = board.shape[1] / px_per_mm
    h_mm = board.shape[0] / px_per_mm
    print(f'Saved {OUT}')
    print(f'Board area: {w_mm:.1f} x {h_mm:.1f} mm, square {SQUARE_MM} mm, {DPI} dpi')
    print('Print at 100% scale. Measure one black square and update SQUARE_MM in calibrate_chessboard.py.')


if __name__ == '__main__':
    main()
