"""Visual check of the physical camera pose.

Fits the camera pose from markers 10 and 23 (fixed markers), then grabs one camera frame and draws:
  green  = detected marker corners
  red    = projected model corners (from the fitted pose)
Also prints the pixel error for markers 10, 23 and 42. Nothing moves; the carriage stays where it is.

Usage: python verify_overlay.py
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np

import camera_fit as cf

HERE = Path(__file__).resolve().parent
INTRINSICS = HERE / 'intrinsics_run2.json'
CAPTURE = HERE / 'capture_markers.json'
POSE_OUT = HERE / 'camera_pose_10_23.json'
OUT_PNG = HERE / 'overlay_check.png'
CAMERA_INDEX = 0
W, H = 640, 480
CARRIAGE_MM_NOW = 432.0               # last capture finished at 432 mm


def fit_pose_10_23():
    intr = json.loads(INTRINSICS.read_text())
    K = np.array(intr['K'], dtype=np.float64)
    dist = np.array(intr['dist'], dtype=np.float64)
    cap = json.loads(CAPTURE.read_text())
    f = float((K[0, 0] + K[1, 1]) / 2)
    cx, cy = float(K[0, 2]), float(K[1, 2])
    obs = []
    for e in cap['entries']:
        q = e['q_mm'] / 1000.0
        for mid, corners in e['markers'].items():
            if int(mid) == 42:
                continue
            und = cv2.undistortPoints(np.array(corners).reshape(-1, 1, 2), K, dist, P=K).reshape(4, 2)
            obs.append((int(mid), q, und))
    best = None
    look = np.array([0.0, 0.0, 0.85])
    for az_deg in (-100, -90, -80):
        for el_deg in (-30, -20, -10):
            for dist_m in (1.0, 1.5, 2.0):
                az, el = math.radians(az_deg), math.radians(el_deg)
                fd = np.array([math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)])
                pos = look - dist_m * fd
                fwd = look - pos
                fwd /= np.linalg.norm(fwd)
                rvec, tvec = cf.rvec_tvec_from_pose(pos, fwd, np.array([0.0, 0.0, 1.0]))
                init = np.concatenate([[f, cx, cy], rvec, tvec, [0.03, 0.04]])
                p, rms, opt = cf.fit_camera(obs, W, H, init, fixed_f=f, fixed_pp=(cx, cy))
                if best is None or rms < best[1]:
                    best = (p, rms, opt)
    params, rms, options = best
    R, _ = cv2.Rodrigues(params[3:6])
    cam_pos = -R.T @ params[6:9]
    POSE_OUT.write_text(json.dumps(dict(
        K=[[params[0], 0, params[1]], [0, params[0], params[2]], [0, 0, 1]],
        rvec=params[3:6].tolist(), tvec=params[6:9].tolist(),
        camera_position_m=cam_pos.tolist(), rms_px=float(rms),
        pattern_half_m={'10_23': float(params[9])},
        orientation_options={str(k): list(v) for k, v in options.items()},
        image_size=[W, H], distortion=dist.tolist(), focal_px=f,
        note='fit from markers 10 and 23 only; intrinsics_run2.json',
    ), indent=2))
    print(f'Pose from 10/23: RMS {rms:.3f} px, camera position {np.round(cam_pos, 3)} m')
    return params, options, K, dist


def open_camera():
    for _ in range(3):
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(CAMERA_INDEX)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            time_ok = sum(int(cap.read()[0]) for _ in range(15))
            if time_ok >= 10:
                return cap
        cap.release()
        cv2.waitKey(2000)
    return None


def main():
    params, options, K, dist = fit_pose_10_23()
    cap = open_camera()
    if cap is None:
        raise SystemExit('Camera gives no frames. Close other camera programs and try again. Nothing moved.')
    frame = None
    for _ in range(10):                   # take a fresh frame
        ok, fr = cap.read()
        if ok:
            frame = fr
    cap.release()
    if frame is None:
        raise SystemExit('No frame read.')

    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                                       cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    detected = {}
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            detected[int(i)] = c.reshape(4, 2)

    vis = frame.copy()
    q_m = CARRIAGE_MM_NOW / 1000.0
    print('Marker  error(mean px)   note')
    for mid in (10, 23, 42):
        half = params[10] if mid == 42 else params[9]
        opt = options.get(mid)
        if opt is None:                   # 42 was not in the 10/23 fit: use its last-fitted orientation
            opt = (2, True)
        world = cf.option_corners(cf.marker_centre(mid, q_m), half, opt)
        proj = cf.project(params, world, W, H)
        for p in proj:
            cv2.circle(vis, (int(round(p[0])), int(round(p[1]))), 4, (0, 0, 255), -1)
        if mid in detected:
            d = detected[mid]
            for p in d:
                cv2.circle(vis, (int(round(p[0])), int(round(p[1]))), 4, (0, 255, 0), 2)
            err = np.linalg.norm(proj - d, axis=1).mean()
            print(f'  {mid:3d}   {err:8.2f}        detected and compared')
        else:
            print(f'  {mid:3d}   not detected in this frame')
    cv2.putText(vis, 'green = detected corners, red = projected model', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imwrite(str(OUT_PNG), vis)
    print(f'Saved {OUT_PNG}')


if __name__ == '__main__':
    main()
