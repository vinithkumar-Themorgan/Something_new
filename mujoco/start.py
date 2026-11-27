"""Capture ArUco corners at several carriage positions for camera pose fitting.

Usage:  python capture_markers.py COM5
Moves the physical carriage (after confirmation) through TARGETS_MM, and at each position
records the corners of markers 10, 23 and 42 from the physical camera.
Output: capture_markers.json next to this script.

Keep the hardware E-stop in reach and the rail clear.
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import serial

from hw_move_test import read_tel, send_and_wait, TRAVEL_MM, FULL_PULSES

CAMERA_INDEX = 0
CAPTURE_W, CAPTURE_H = 640, 480          # same as the tracking app
TARGETS_MM = [0.0, 108.0, 216.0, 324.0, 432.0]
FRAMES_PER_POSITION = 15
SETTLE_S = 1.0
MARKER_IDS = (10, 23, 42)
OUT = Path(__file__).resolve().parent / 'capture_markers.json'


def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError('Camera not available')
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def grab_corners(cap, detector, n):
    """Median corners per marker id over n frames."""
    samples = {mid: [] for mid in MARKER_IDS}
    for _ in range(n):
        ok, frame = cap.read()
        if not ok:
            continue
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if ids is None:
            continue
        for c, i in zip(corners, ids.flatten()):
            if int(i) in samples:
                samples[int(i)].append(c.reshape(4, 2))
    out = {}
    for mid, arr in samples.items():
        if len(arr) >= n // 2:
            out[str(mid)] = np.median(np.stack(arr), axis=0).tolist()
    return out


def main():
    if len(sys.argv) < 3:
        sys.exit('Usage: python capture_markers.py COMx')
    ser = serial.Serial(sys.argv[1], 115200, timeout=0.05, write_timeout=0.5)
    time.sleep(1.8)
    ser.reset_input_buffer()
    tel = read_tel(ser)
    if tel is None:
        sys.exit('No telemetry from the board.')
    if tel['estop'] or tel['swstop']:
        sys.exit('E-stop or software stop active. Release it first.')

    cap = open_camera()
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                                       cv2.aruco.DetectorParameters())
    print(f'Camera {int(cap.get(3))}x{int(cap.get(4))}. Positions: {TARGETS_MM} mm')
    if input("Type GO to move the carriage and capture: ").strip() != 'GO':
        print('Cancelled. No motion.')
        return

    if not tel['known']:
        print('HOME ...'); _, tel = send_and_wait(ser, 'HOME', 'HOME')
        print('END ...'); _, tel = send_and_wait(ser, 'END', 'END')
        if not tel['known']:
            sys.exit('Position still unknown after HOME/END.')

    entries = []
    try:
        for target in TARGETS_MM:
            target_pulses = round(target * FULL_PULSES / TRAVEL_MM)
            delta = tel['pulses'] - target_pulses
            if abs(delta) > 2:
                print(f'Move to {target:.0f} mm ...')
                _, tel = send_and_wait(ser, f'M {delta}', f'{target:.0f} mm')
            time.sleep(SETTLE_S)
            tel = read_tel(ser) or tel
            pos_mm = tel['pulses'] * TRAVEL_MM / FULL_PULSES
            markers = grab_corners(cap, detector, FRAMES_PER_POSITION)
            print(f'  at {pos_mm:.1f} mm (commanded {target:.0f}): markers seen {sorted(map(int, markers))}')
            entries.append(dict(q_mm=target, measured_mm=round(pos_mm, 2), markers=markers))
    except RuntimeError as exc:
        print('ERROR:', exc)
    finally:
        ser.write(b'STOP\n')
        ser.close()
        cap.release()

    OUT.write_text(json.dumps(dict(created=datetime.now().isoformat(timespec='seconds'),
                                   image_size=[CAPTURE_W, CAPTURE_H], entries=entries), indent=2))
    print(f'Saved {OUT}  ({len(entries)} positions)')


if __name__ == '__main__':
    main()
