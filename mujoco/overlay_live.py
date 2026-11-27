"""Live overlay on the physical camera: the virtual ball path, drawn with a pose that updates every frame.

The pose comes from markers 10 and 23 on each frame (live_pose.py), so the overlay follows the
camera if it is moved. Marker 42 is not used for the pose.

  green  = detected marker corners (10 and 23)
  red    = projected model corners of 10 and 23 (should sit on the green)
  orange = virtual ball path for a sample throw
  yellow = rim crossing point

Keys: Q quit, S save the frame.
Usage: python overlay_live.py
"""
import time

import cv2
import numpy as np

from live_pose import LivePose

CAMERA_INDEX = 0
W, H = 640, 480
GRAVITY = 9.81
BUCKET_X_HOME_M = -0.219
RIM_Z_M = 0.755 + 0.1901
LAUNCH_POS_M = np.array([0.0, 0.6, 1.1])
SAMPLE_T_S = 1.2
SAMPLE_LANDING_MM = 216.0


def ball_path(T, landing_mm, steps=40):
    """Ballistic path from the launcher to the rim at the landing point (same maths as the app)."""
    xL = BUCKET_X_HOME_M + landing_mm / 1000.0
    p0 = LAUNCH_POS_M
    v0 = np.array([(xL - p0[0]) / T, (0.0 - p0[1]) / T,
                   (RIM_Z_M - p0[2] + 0.5 * GRAVITY * T * T) / T])
    ts = np.linspace(0.0, T, steps)
    pts = p0 + np.outer(ts, v0) + 0.5 * np.outer(ts * ts, [0.0, 0.0, -GRAVITY])
    rim = np.array([xL, 0.0, RIM_Z_M])
    return pts, rim


def open_camera():
    # Same open sequence as the capture run that worked: DirectShow, size only.
    for attempt in range(1, 4):
        cap = cv2.VideoCapture()
        if cap.open(CAMERA_INDEX, cv2.CAP_DSHOW):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
            good = sum(int(cap.read()[0]) for _ in range(15))
            print(f'camera attempt {attempt}: {good}/15 frames OK')
            if good >= 10:
                return cap
        cap.release()
        time.sleep(2.0)
    return None


def draw_polyline(img, pts, color, thickness=2):
    pts = np.round(pts).astype(int)
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(img, tuple(a), tuple(b), color, thickness)


def main():
    lp = LivePose()
    cap = open_camera()
    if cap is None:
        raise SystemExit('Camera gives no frames. Close other camera programs and try again.')
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                                       cv2.aruco.DetectorParameters())
    pts_ball, rim = ball_path(SAMPLE_T_S, SAMPLE_LANDING_MM)
    print('Live overlay running. Q quits, S saves a frame.')

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        detected = {}
        if ids is not None:
            for c, i in zip(corners, ids.flatten()):
                if int(i) in (10, 23):
                    detected[int(i)] = c.reshape(4, 2)
        tracked = lp.update(detected)
        vis = frame.copy()
        for mid, d in detected.items():
            for p in d:
                cv2.circle(vis, (int(round(p[0])), int(round(p[1]))), 4, (0, 255, 0), 2)

        if tracked:
            # Model corners of 10 and 23 projected with the live pose (should sit on the green)
            proj = lp.project(lp.obj)
            for p in proj:
                cv2.circle(vis, (int(round(p[0])), int(round(p[1]))), 3, (0, 0, 255), -1)
            draw_polyline(vis, lp.project(pts_ball), (0, 140, 255), 2)
            r = lp.project(rim)
            if len(r):
                cv2.circle(vis, (int(round(r[0][0])), int(round(r[0][1]))), 7, (0, 255, 255), 2)
            cam = lp.camera_position()
            status = f'pose RMS {lp.rms:.2f}px  camera ({cam[0]:.2f}, {cam[1]:.2f}, {cam[2]:.2f}) m'
        else:
            status = 'POSE LOST: markers 10 and 23 not both visible'
        cv2.putText(vis, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow('INTERCEPT-X live overlay', vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('s'):
            cv2.imwrite('overlay_live_frame.png', vis)
            print('saved overlay_live_frame.png')
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
