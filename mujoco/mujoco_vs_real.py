"""Live check: MuJoCo view vs the real camera, with the same camera pose.

Each frame:
  1. Live pose from markers 10 and 23, filtered (live_pose.py).
  2. Carriage position from the camera: marker 42 is matched against the model (no serial needed).
     If the board is connected and marker 42 is not visible, the board's pulse count is used instead.
  3. MuJoCo camera set to the live pose (match_virtual_camera.py), carriage at that position.
  4. Renders the rig and a sample ball; the same ball is drawn on the real frame (projection).
  5. Compares marker centres (10, 23, 42): MuJoCo render vs real detections, in pixels.

Windows: 'real + overlay', 'MuJoCo matched', 'blend 50%'.
Keys: Q quit, S save.

Usage: python mujoco_vs_real.py [COM3]     (COM port optional, used only as a fallback)
"""
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import mujoco
import numpy as np
import serial

import match_virtual_camera as mvc
from live_pose import LivePose

XML = Path('C:/Users/vinith/AppData/Local/Temp/claude/5b5ff089e563181a/380e24e4-3b72-4177-84a5-ec9c5a570d75/scratchpad/catch_x_user.xml')
CAMERA_INDEX = 0
W, H = 640, 480
CARRIAGE_MM_DEFAULT = 216.0
TRAVEL_MM = 432.0
FULL_PULSES = 34192
GRAVITY = 9.81
BUCKET_X_HOME_M = -0.219
RIM_Z_M = 0.755 + 0.1901
LAUNCH_POS_M = np.array([0.0, 0.6, 1.1])
SAMPLE_T = 1.2                 # physics flight time of the sample throw (real seconds)
SAMPLE_LANDING_MM = 216.0
SLOWMO = 0.5                   # display speed of the sample ball (0.5 = half speed); carriage stays real-time
MM_WINDOW = 5                  # median window for the camera-based carriage position
TARGET_PX = 8.0                # acceptable average corner error between MuJoCo and real


def ball_at(t):
    """Ball position t seconds after launch (physics time)."""
    xL = BUCKET_X_HOME_M + SAMPLE_LANDING_MM / 1000.0
    p0 = LAUNCH_POS_M
    v0 = np.array([(xL - p0[0]) / SAMPLE_T, (0 - p0[1]) / SAMPLE_T,
                   (RIM_Z_M - p0[2] + 0.5 * GRAVITY * SAMPLE_T ** 2) / SAMPLE_T])
    return p0 + v0 * t + 0.5 * np.array([0, 0, -GRAVITY]) * t * t


def read_carriage_mm(ser):
    """Read-only TELEM query. Returns mm, or None if unknown."""
    ser.write(b'TELEM\n')
    deadline = time.time() + 0.2
    while time.time() < deadline:
        line = ser.readline().decode('utf-8', 'replace').strip()
        if line.startswith('TEL,'):
            a = line.split(',', 11)
            if len(a) == 12 and int(a[6]):
                return int(a[9]) * TRAVEL_MM / FULL_PULSES
            return None
    return None


def open_camera():
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


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else None
    ser = serial.Serial(port, 115200, timeout=0.05, write_timeout=0.2) if port else None
    if ser:
        time.sleep(1.8)
        ser.reset_input_buffer()

    lp = LivePose()
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    qa = int(model.jnt_qposadr[model.joint('carriage_slide').id])
    ball_adr = int(model.jnt_qposadr[model.joint('ball_free').id])
    rend = mujoco.Renderer(model, height=H, width=W)
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                                       cv2.aruco.DetectorParameters())
    cap = open_camera()
    if cap is None:
        raise SystemExit('Camera gives no frames. Close other camera programs and try again.')
    cam_free = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam_free)
    mm_hist = deque(maxlen=MM_WINDOW)
    err_hist = deque(maxlen=60)          # recent average pixel errors
    t_start = time.monotonic()
    cycle = SAMPLE_T + 1.0               # ball shot every cycle (display seconds), then a pause
    print(f'Running. Ball at {SLOWMO:.1f}x speed. Target: average error under {TARGET_PX:.0f} px. Q quits, S saves.')

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        real = {}
        if ids is not None:
            for c, i in zip(corners, ids.flatten()):
                if int(i) in (10, 23, 42):
                    real[int(i)] = c.reshape(4, 2)
        tracked = lp.update({k: v for k, v in real.items() if k in (10, 23)})
        if not tracked:
            cv2.imshow('real + overlay', frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
            continue

        # Carriage position: camera (marker 42) first, board pulse count as fallback
        if 42 in real:
            mm_hist.append(lp.carriage_mm_from_42(real[42]))
            mm = float(np.median(mm_hist))
            source = 'camera'
        else:
            mm = read_carriage_mm(ser) if ser else None
            source = 'board' if mm is not None else 'none'
            if mm is None:
                mm = float(np.median(mm_hist)) if mm_hist else CARRIAGE_MM_DEFAULT

        cam = mvc.mujoco_camera_from_pose(dict(rvec=lp.rvec.ravel().tolist(), tvec=lp.tvec.ravel().tolist()), lp.K)
        cam_free.distance = 1.0
        cam_free.lookat[:] = cam['pos'] + cam_free.distance * cam['fwd']
        cam_free.azimuth = cam['az']
        cam_free.elevation = cam['el']
        model.vis.global_.fovy = cam['fovy']

        data.qpos[qa] = mm / 1000.0
        t_disp = (time.monotonic() - t_start) * SLOWMO
        t_rel = t_disp % cycle
        ball_visible = t_rel < SAMPLE_T
        if ball_visible:
            data.qpos[ball_adr:ball_adr + 3] = ball_at(t_rel)
            data.qpos[ball_adr + 3:ball_adr + 7] = [1, 0, 0, 0]
        else:
            data.qpos[ball_adr:ball_adr + 3] = [0, 0, -10]
        mujoco.mj_forward(model, data)
        rend.update_scene(data, camera=cam_free)
        sim = cv2.cvtColor(rend.render(), cv2.COLOR_RGB2BGR)
        shift = np.float32([[1, 0, cam['dx']], [0, 1, cam['dy']]])
        sim = cv2.warpAffine(sim, shift, (W, H))

        # Marker comparison: rendered centres vs real detections
        s_corners, s_ids, _ = detector.detectMarkers(cv2.cvtColor(sim, cv2.COLOR_BGR2GRAY))
        sim_marks = {}
        if s_ids is not None:
            for c, i in zip(s_corners, s_ids.flatten()):
                sim_marks[int(i)] = c.reshape(4, 2)
        errs = {}
        for mid in (10, 23, 42):
            if mid in real and mid in sim_marks:
                errs[mid] = float(np.linalg.norm(real[mid] - sim_marks[mid], axis=1).mean())
        if errs:
            err_hist.append(np.mean(list(errs.values())))
        avg_err = float(np.median(err_hist)) if err_hist else float('nan')

        vis = frame.copy()
        for d in real.values():
            for p in d:
                cv2.circle(vis, (int(round(p[0])), int(round(p[1]))), 4, (0, 255, 0), 2)
        for p in lp.project(lp.obj):
            cv2.circle(vis, (int(round(p[0])), int(round(p[1]))), 3, (0, 0, 255), -1)
        if ball_visible:
            pb = lp.project(ball_at(t_rel))
            if len(pb):
                cv2.circle(vis, (int(pb[0][0]), int(pb[0][1])), 8, (0, 140, 255), -1)
        status = (f'carriage {mm:.0f} mm ({source})  pose RMS {lp.rms:.2f}px  '
                  f'avg error {avg_err:.1f}px (target <{TARGET_PX:.0f})')
        cv2.putText(vis, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        blend = cv2.addWeighted(frame, 0.5, sim, 0.5, 0)

        cv2.imshow('real + overlay', vis)
        cv2.imshow('MuJoCo matched', sim)
        cv2.imshow('blend 50%', blend)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('s'):
            cv2.imwrite('mvr_real_overlay.png', vis)
            cv2.imwrite('mvr_mujoco.png', sim)
            cv2.imwrite('mvr_blend.png', blend)
            print('saved mvr_*.png')
    cap.release()
    cv2.destroyAllWindows()
    if ser:
        ser.close()


if __name__ == '__main__':
    main()
