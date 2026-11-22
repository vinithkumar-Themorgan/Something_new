"""Capture ArUco corners at several carriage positions for camera pose fitting.

Run with no arguments:  python capture_markers.py
Set PORT below to the board's COM port (COM3 on this PC).

Moves the carriage (after you type GO) through TARGETS_MM. At each position it records the
corners of markers 10, 23 and 42 from the physical camera. Output: capture_markers.json here.

Keep the hardware E-stop in reach and the rail clear. Close the INTERCEPT-X GUI and any other
camera program first, because only one program can use the camera at a time.
"""
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import serial

PORT = 'COM3'
TRAVEL_MM = 432.0
FULL_PULSES = 34192
TARGETS_MM = [0.0, 108.0, 216.0, 324.0, 432.0]
CAMERA_INDEX = 0
CAPTURE_W, CAPTURE_H = 640, 480      # same as the tracking app
CAPTURE_FPS = 30                     # same as the tracking app
OPEN_ATTEMPTS = 3                    # the camera sometimes needs a second try after opening
PREFLIGHT_FRAMES = 15
PREFLIGHT_MIN_OK = 5
FRAMES_PER_POSITION = 15
SETTLE_S = 1.0
MARKER_IDS = (10, 23, 42)
MOVE_TIMEOUT_S = 20.0
START_TIMEOUT_S = 2.0
OUT = Path(__file__).resolve().parent / 'capture_markers.json'


def parse_tel(line):
    a = line.split(',', 11)
    if len(a) < 12:
        return None
    return dict(running=int(a[1]), rev=int(a[2]), fwd=int(a[3]), estop=int(a[4]),
                swstop=int(a[5]), known=int(a[6]), pulses=int(a[9]), reason=a[11])


def read_tel(ser, timeout=1.0):
    """Ask the board for its status and return the parsed TEL fields, or None."""
    ser.write(b'TELEM\n')
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = ser.readline().decode('utf-8', 'replace').strip()
        if line.startswith('TEL,'):
            tel = parse_tel(line)
            if tel is not None:
                return tel
    return None


def send_and_wait(ser, cmd, label):
    """Send a motion command and wait until the motor has run and stopped."""
    ser.write((cmd + '\n').encode('ascii'))
    start = time.time()
    seen_running = False
    while True:
        if time.time() - start > MOVE_TIMEOUT_S:
            raise RuntimeError(f'{label}: timed out')
        tel = read_tel(ser, timeout=0.3)
        if tel is None:
            continue
        if tel['running']:
            seen_running = True
            continue
        if seen_running:
            time.sleep(0.5)
            return read_tel(ser) or tel
        if time.time() - start > START_TIMEOUT_S:
            raise RuntimeError(f'{label}: motor did not start (board says: {tel["reason"]})')


def open_ready_camera():
    """Open the camera and confirm it delivers frames. Retries, because the first open can fail."""
    for attempt in range(1, OPEN_ATTEMPTS + 1):
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(CAMERA_INDEX)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
            cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            time.sleep(1.0)                 # give the stream time to start
            good = sum(int(cap.read()[0]) for _ in range(PREFLIGHT_FRAMES))
            if good >= PREFLIGHT_MIN_OK:
                print(f'Camera ready on attempt {attempt}: {good}/{PREFLIGHT_FRAMES} test frames OK')
                return cap
            print(f'Attempt {attempt}: only {good}/{PREFLIGHT_FRAMES} frames OK')
        cap.release()
        time.sleep(2.0)
    return None


def grab_corners(cap, detector, n):
    """Median corner positions per marker ID over n frames."""
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
    ser = serial.Serial(PORT, 115200, timeout=0.05, write_timeout=0.5)
    time.sleep(1.8)                       # board may reboot when the port opens
    ser.reset_input_buffer()
    tel = read_tel(ser)
    if tel is None:
        ser.close()
        raise SystemExit(f'No status reply from the board on {PORT}. Check the port and the GUI is closed.')
    if tel['estop'] or tel['swstop']:
        ser.close()
        raise SystemExit('E-stop or software stop is active. Release it first.')

    # Camera check happens BEFORE any motion
    cap = open_ready_camera()
    if cap is None:
        ser.close()
        raise SystemExit('The camera gives no frames after 3 attempts. No motion was made. '
                         'Close other camera programs, or restart the PC if the camera stays stuck.')
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                                       cv2.aruco.DetectorParameters())
    print(f'Board on {PORT}. Positions: {TARGETS_MM} mm. Rail must be clear and the E-stop in reach.')
    if input('Type GO to move the carriage and capture: ').strip() != 'GO':
        print('Cancelled. No motion.')
        cap.release()
        ser.close()
        return

    entries = []
    try:
        if not tel['known']:
            print('HOME ...')
            tel = send_and_wait(ser, 'HOME', 'HOME')
            print('END ...')
            tel = send_and_wait(ser, 'END', 'END')
            if not tel['known']:
                raise RuntimeError('Position still unknown after HOME/END. Check the limit switches.')

        for target in TARGETS_MM:
            target_pulses = round(target * FULL_PULSES / TRAVEL_MM)
            delta = tel['pulses'] - target_pulses        # M +n moves forward (pulses decrease)
            if abs(delta) > 2:
                print(f'Move to {target:.0f} mm ...')
                tel = send_and_wait(ser, f'M {delta}', f'{target:.0f} mm')
            time.sleep(SETTLE_S)
            tel = read_tel(ser) or tel
            pos_mm = tel['pulses'] * TRAVEL_MM / FULL_PULSES
            markers = grab_corners(cap, detector, FRAMES_PER_POSITION)
            print(f'  at {pos_mm:.1f} mm (commanded {target:.0f}): markers seen {sorted(map(int, markers))}')
            entries.append(dict(q_mm=target, measured_mm=round(pos_mm, 2), markers=markers))
    except KeyboardInterrupt:
        print('\nInterrupted. Sending STOP.')
    except RuntimeError as exc:
        print('\nERROR:', exc)
    finally:
        try:
            ser.write(b'STOP\n')
        except serial.SerialException:
            pass
        ser.close()
        cap.release()

    OUT.write_text(json.dumps(dict(created=datetime.now().isoformat(timespec='seconds'),
                                   image_size=[CAPTURE_W, CAPTURE_H], entries=entries), indent=2))
    print(f'Saved {OUT} ({len(entries)} positions)')


if __name__ == '__main__':
    main()
