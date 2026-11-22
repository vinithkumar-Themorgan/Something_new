"""Calibrate the physical camera's focal length and distortion with a chessboard.

Usage:  python calibrate_chessboard.py
Controls:  SPACE = capture the current view (only when the board is detected)
           Q     = finish and compute (needs at least 15 good captures)

Hold the board flat and rigid, at different distances, angles and positions across the image,
including near the corners. Use the same resolution as the tracking app (CAPTURE_W x CAPTURE_H).
Writes camera_intrinsics.json next to this script.
"""
from datetime import datetime
from pathlib import Path
import json
import cv2
import numpy as np

CAMERA_INDEX = 0
CAPTURE_W, CAPTURE_H = 640, 480      # must match the tracking app
INNER_COLS, INNER_ROWS = 9, 6        # must match the printed board
SQUARE_MM = 26.0                     # measured size of one printed square
MIN_CAPTURES = 15
OUT = Path(__file__).resolve().parent / 'camera_intrinsics.json'
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)

# Camera lock settings. OFF by default: on this camera driver, setting these controls broke the
# video stream. Lock focus and exposure in the camera's own software instead, then leave this False.
LOCK_CAMERA_CONTROLS = False
FOCUS_VALUE = None           # used only if LOCK_CAMERA_CONTROLS is True
EXPOSURE_VALUE = None        # used only if LOCK_CAMERA_CONTROLS is True


def open_camera():
    # Same open sequence as the INTERCEPT-X app, so both see the same camera.
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError('Camera not available')
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    settings = {}
    if LOCK_CAMERA_CONTROLS:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        if FOCUS_VALUE is not None:
            cap.set(cv2.CAP_PROP_FOCUS, FOCUS_VALUE)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        if EXPOSURE_VALUE is not None:
            cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE_VALUE)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        settings = dict(autofocus=cap.get(cv2.CAP_PROP_AUTOFOCUS), focus=cap.get(cv2.CAP_PROP_FOCUS),
                        auto_exposure=cap.get(cv2.CAP_PROP_AUTO_EXPOSURE), exposure=cap.get(cv2.CAP_PROP_EXPOSURE),
                        auto_wb=cap.get(cv2.CAP_PROP_AUTO_WB))
        print('Camera controls read back:', settings)
    # Warm-up: discard the first frames while the stream starts
    for _ in range(10):
        cap.read()
    return cap, settings


def main():
    cap, cam_settings = open_camera()
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Camera {w}x{h}. Hold the board in view. SPACE = capture, Q = compute.')

    # Board coordinates in mm: z = 0 plane, origin at the first inner corner
    objp = np.zeros((INNER_COLS * INNER_ROWS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:INNER_COLS, 0:INNER_ROWS].T.reshape(-1, 2) * SQUARE_MM

    obj_points, img_points = [], []
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (INNER_COLS, INNER_ROWS), flags)
        view = frame.copy()
        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
            cv2.drawChessboardCorners(view, (INNER_COLS, INNER_ROWS), corners, found)
        cv2.putText(view, f'captures: {len(obj_points)}  board: {"YES" if found else "no"}  SPACE=capture Q=compute',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow('Chessboard calibration', view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and found:
            obj_points.append(objp)
            img_points.append(corners)
            print(f'captured {len(obj_points)}')
        elif key in (ord('q'), 27):
            break
    cap.release()
    cv2.destroyAllWindows()

    if len(obj_points) < MIN_CAPTURES:
        raise SystemExit(f'Only {len(obj_points)} captures; need at least {MIN_CAPTURES}. Capture more views.')

    rms, K, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    print(f'\nReprojection RMS: {rms:.3f} px  (good if below about 0.5 px)')
    print(f'fx {fx:.2f}  fy {fy:.2f}  cx {cx:.2f}  cy {cy:.2f}')
    print(f'distortion k1..k3: {dist.ravel()[:3]}')
    if rms > 1.0:
        print('RMS is high. Recapture with sharper, flatter views and more variety.')

    data = dict(
        K=K.tolist(), dist=dist.ravel().tolist(), image_size=[w, h],
        rms_px=float(rms), captures=len(obj_points),
        square_mm=SQUARE_MM, inner_corners=[INNER_COLS, INNER_ROWS],
        camera_settings=cam_settings,
        created=datetime.now().isoformat(timespec='seconds'),
    )
    # Timestamped file so runs don't overwrite each other
    out = OUT.with_name(f'camera_intrinsics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    out.write_text(json.dumps(data, indent=2))
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
