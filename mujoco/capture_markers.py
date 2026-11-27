"""Capture ArUco corners at several carriage positions for camera pose fitting.

Run: python capture_markers.py

The camera is verified BEFORE COM3 is opened or any motion is requested.
Keep the hardware E-stop within reach, clear the rail, and close other camera apps.

Important: The movement command and pulse convention are carried over from the
original working controller script. Check HOME/END direction before running.
"""
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import serial

PORT = 'COM3'
BAUDRATE = 115200
TRAVEL_MM = 432.0
FULL_PULSES = 34192
TARGETS_MM = [0.0, 108.0, 216.0, 324.0, 432.0]
CAMERA_INDEX = 0
CAPTURE_W, CAPTURE_H = 640, 480
CAPTURE_FPS = 30
OPEN_ATTEMPTS = 3
PREFLIGHT_FRAMES = 15
PREFLIGHT_MIN_OK = 10
FRAMES_PER_POSITION = 15
SETTLE_S = 1.0
MARKER_IDS = (10, 23, 42)
MOVE_TIMEOUT_S = 20.0
START_TIMEOUT_S = 2.0
OUT = Path(__file__).resolve().parent / 'capture_markers.json'


def parse_tel(line):
    """Extract relevant fields from the controller's TEL message."""
    try:
        a = line.split(',', 11)
        if len(a) < 12 or a[0] != 'TEL':
            return None
        return dict(
            running=int(a[1]), rev=int(a[2]), fwd=int(a[3]),
            estop=int(a[4]), swstop=int(a[5]), known=int(a[6]),
            pulses=int(a[9]), reason=a[11].strip()
        )
    except (ValueError, IndexError):
        return None


def read_tel(ser, timeout=1.0):
    """Poll the board, ignoring unrelated serial replies."""
    ser.write(b'TELEM\n')
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode('utf-8', 'replace').strip()
        if line.startswith('TEL,'):
            tel = parse_tel(line)
            if tel is not None:
                return tel
    return None


def ensure_safe(tel, label):
    if tel['estop'] or tel['swstop']:
        raise RuntimeError(f'{label}: hardware/software E-stop active')


def send_and_wait(ser, cmd, label):
    """Wait for a commanded move to start, finish, and report a safe state."""
    # Clear any leftover telemetry before issuing the new command.
    ser.reset_input_buffer()
    ser.write((cmd + '\n').encode('ascii'))
    start = time.monotonic()
    seen_running = False
    last_tel = None
    while time.monotonic() - start < MOVE_TIMEOUT_S:
        tel = read_tel(ser, timeout=0.3)
        if tel is None:
            continue
        last_tel = tel
        ensure_safe(tel, label)
        if tel['running']:
            seen_running = True
            continue
        if seen_running:
            # Confirm stopped, rather than trusting one transitional read.
            time.sleep(0.15)
            confirmed = read_tel(ser, timeout=1.0)
            if confirmed is None:
                raise RuntimeError(f'{label}: no final telemetry')
            ensure_safe(confirmed, label)
            if confirmed['running']:
                seen_running = True
                continue
            return confirmed
        if time.monotonic() - start > START_TIMEOUT_S:
            raise RuntimeError(
                f'{label}: motor did not start (board reason: {tel["reason"]})'
            )
    raise RuntimeError(f'{label}: motion timeout; last telemetry={last_tel}')


def open_ready_camera():
    """Try the proven DirectShow/default format first, without forcing MJPEG.

    A failed or unstable camera is always released. Never fall back to MSMF
    silently: the previous logs showed MSMF opened but delivered 0 frames.
    """
    for attempt in range(1, OPEN_ATTEMPTS + 1):
        print(f'Opening camera index {CAMERA_INDEX} with DirectShow, attempt {attempt}/{OPEN_ATTEMPTS}', flush=True)
        cap = cv2.VideoCapture()
        ready = False
        try:
            if not cap.open(CAMERA_INDEX, cv2.CAP_DSHOW):
                print('  DirectShow camera open FAILED', flush=True)
                continue

            # Match the successful standalone test exactly: no MJPEG;
            # set resolution, but do not force FPS or buffer size.
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
            good = 0
            sample_shape = None
            started = time.monotonic()
            for _ in range(PREFLIGHT_FRAMES):
                ok, frame = cap.read()
                if ok and frame is not None and frame.size > 0:
                    good += 1
                    sample_shape = frame.shape[:2]
            elapsed = time.monotonic() - started
            print(f'  Valid frames: {good}/{PREFLIGHT_FRAMES}; read rate: {PREFLIGHT_FRAMES/max(elapsed, 0.001):.1f} calls/s', flush=True)
            if good < PREFLIGHT_MIN_OK or sample_shape is None:
                continue

            h, w = sample_shape
            print(f'  Actual camera resolution: {w}x{h}', flush=True)
            if (w, h) != (CAPTURE_W, CAPTURE_H):
                print(f'  NOTE: requested {CAPTURE_W}x{CAPTURE_H}, but using actual {w}x{h} for JSON.', flush=True)
            ready = True
            return cap, [w, h]
        except cv2.error as exc:
            print(f'  OpenCV camera exception: {exc}', flush=True)
        finally:
            if not ready:
                cap.release()
        time.sleep(1.0)
    return None, None


def grab_corners(cap, detector, n, expected_size):
    """Return median corners by ID and detection counts across usable frames."""
    samples = {mid: [] for mid in MARKER_IDS}
    good_frames = 0
    for _ in range(n):
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            continue
        h, w = frame.shape[:2]
        if [w, h] != expected_size:
            raise RuntimeError(
                f'Camera resolution changed from {expected_size} to {[w, h]} during capture'
            )
        good_frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None:
            continue
        for corner, mid in zip(corners, ids.flatten()):
            mid = int(mid)
            if mid in samples:
                samples[mid].append(corner.reshape(4, 2))

    if good_frames < max(5, n // 2):
        raise RuntimeError(
            f'Camera frame delivery failed during capture ({good_frames}/{n} valid frames). '
            'Stopping calibration; no empty result will be silently saved.'
        )

    # Require detections in at least half the *valid* frames per ID.
    threshold = max(1, (good_frames + 1) // 2)
    markers = {}
    for mid, arr in samples.items():
        if len(arr) >= threshold:
            markers[str(mid)] = np.median(np.stack(arr), axis=0).tolist()
    counts = {str(mid): len(arr) for mid, arr in samples.items()}
    return markers, counts, good_frames


def main():
    # Camera MUST be ready before any serial connection or movement.
    cap, image_size = open_ready_camera()
    if cap is None:
        raise SystemExit(
            'Camera failed DirectShow preflight; NO motion made. '
            'Close other camera apps and re-run the standalone camera test.'
        )

    ser = None
    entries = []
    try:
        detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
            cv2.aruco.DetectorParameters()
        )

        print('Connecting to controller...', flush=True)
        ser = serial.Serial(PORT, BAUDRATE, timeout=0.05, write_timeout=0.5)
        time.sleep(1.8)  # ESP32 may reboot when COM port opens
        ser.reset_input_buffer()
        tel = read_tel(ser)
        if tel is None:
            raise RuntimeError(f'No telemetry from {PORT}. Close INTERCEPT-X GUI and verify port.')
        ensure_safe(tel, 'Preflight')

        print(f'Camera ready, board on {PORT}. Targets: {TARGETS_MM} mm.', flush=True)
        print('Verify the rail is clear, HOME/END direction is correct, and E-stop is within reach.')
        if input('Type GO to start motion and capture: ').strip() != 'GO':
            print('Cancelled. No motion requested.')
            return

        tel = read_tel(ser) or tel
        ensure_safe(tel, 'Before motion')

        if not tel['known']:
            print('HOME ...', flush=True)
            tel = send_and_wait(ser, 'HOME', 'HOME')
            print('END ...', flush=True)
            tel = send_and_wait(ser, 'END', 'END')
            if not tel['known']:
                raise RuntimeError('Position unknown after HOME/END; inspect limit switches.')

        for target in TARGETS_MM:
            target_pulses = round(target * FULL_PULSES / TRAVEL_MM)
            # Preserved from original firmware convention: M +n => pulses decrease.
            delta = tel['pulses'] - target_pulses
            if abs(delta) > 2:
                print(f'Moving to {target:.0f} mm (M {delta}) ...', flush=True)
                tel = send_and_wait(ser, f'M {delta}', f'{target:.0f} mm')

            time.sleep(SETTLE_S)
            tel = read_tel(ser) or tel
            ensure_safe(tel, 'At capture position')
            if tel['running']:
                raise RuntimeError('Controller reports motion during image capture')

            pos_mm = tel['pulses'] * TRAVEL_MM / FULL_PULSES
            markers, counts, valid = grab_corners(
                cap, detector, FRAMES_PER_POSITION, image_size
            )
            print(
                f'  At {pos_mm:.1f} mm (target {target:.0f}); '
                f'frames {valid}/{FRAMES_PER_POSITION}; '
                f'IDs accepted: {sorted(map(int, markers))}; counts: {counts}',
                flush=True
            )
            entries.append(dict(
                q_mm=target,
                measured_mm=round(pos_mm, 2),
                markers=markers,
                marker_counts=counts,
                valid_frames=valid
            ))

    except KeyboardInterrupt:
        print('\nInterrupted; requesting STOP.', flush=True)
    except (RuntimeError, serial.SerialException, cv2.error) as exc:
        print(f'\nERROR: {exc}', flush=True)
    finally:
        if ser is not None:
            try:
                ser.write(b'STOP\n')
                ser.flush()
            except (serial.SerialException, OSError):
                pass
            ser.close()
        cap.release()

        # Save partial results when capture began; no file for cancelled runs.
        if entries:
            payload = dict(
                created=datetime.now().isoformat(timespec='seconds'),
                image_size=image_size,
                entries=entries
            )
            OUT.write_text(json.dumps(payload, indent=2), encoding='utf-8')
            print(f'Saved {OUT} ({len(entries)}/{len(TARGETS_MM)} positions)', flush=True)


if __name__ == '__main__':
    main()
