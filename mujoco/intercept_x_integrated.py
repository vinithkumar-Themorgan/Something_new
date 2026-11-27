"""INTERCEPT-X | physical catcher + MuJoCo twin + camera overlay + virtual ball catch.

What runs in the vision thread, every camera frame:
  * Live camera pose from fixed markers 10 and 23 (live_pose.py, filtered).
  * Carriage position from the camera: marker 42 matched to the model (fallback: board pulse count).
  * MuJoCo twin rendered with the same camera as the real frame (match_virtual_camera.py).
  * Real frame overlay: marker corners, the virtual ball, and the rim point.

Catch check at impact time, using the carriage position from the camera (or pulses if the camera is lost):
  caught if the bucket centre is within 65 mm of the ball (170 mm opening - 40 mm ball).

Needs in the same folder: live_pose.py, camera_fit.py, match_virtual_camera.py, camera_pose_10_23.json,
intrinsics_run2.json (used by match_virtual_camera.py), and catch_x.xml with meshes/.
Always use a hardware E-stop.
"""
from collections import deque
from pathlib import Path
import queue, threading, time, traceback, tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import cv2
import mujoco
import serial
from serial.tools import list_ports
from PIL import Image, ImageTk

import camera_fit as cf
import match_virtual_camera as mvc
from live_pose import LivePose

ROOT = Path(__file__).resolve().parent
CAMERA_INDEX = 0
CAPTURE_W, CAPTURE_H = 640, 480
CAPTURE_FPS = 30
RENDER_HZ = 30
DISPLAY_W, DISPLAY_H = 580, 330
GUI_REFRESH_MS = 15
TELEM_POLL_S = 0.05
MARKERS = (23, 10, 42)
TRAVEL_MM = 432.0
FULL_PULSES = 34192
GOTO_TOLERANCE_MM = 1.0
GOTO_START_TIMEOUT_S = 2.0
ACCEL_PULSES_S2 = 20000.0            # must match firmware ACCEL
START_PULSES_S = 500.0               # must match firmware START_FREQ
MM_WINDOW = 9                        # median window for the camera carriage position

# ---- Virtual ball (geometry from catch_x.xml and catcher.stl) ----
GRAVITY = 9.81
BALL_WINDOW_MM = 65.0                # (170 - 40) / 2 = 65 mm (kept for reference)
OPEN_X_MM, OPEN_Y_MM, BALL_D_MM = 170.0, 173.0, 40.0   # catcher opening (measured) and ball
CATCH_EDGE_M = 0.0                # 0 = the full 170 x 173 mm opening is the catch range
FLOOR_Z_M = 0.755 + 0.0889           # bucket floor (carriage z + catcher bottom)
BUCKET_X_HOME_M = -0.219
BUCKET_RIM_Z_M = 0.755 + 0.1901
LAUNCH_POS_M = (0.0, 0.6, 1.1)
THROW_MIN_T, THROW_MAX_T = 0.3, 3.0
SIDE_OFFSET_M = 1.00                # launch: ball starts this far to the LEFT of the bucket, outside the frame
# ---- Catcher contour (drawn from marker 42) ----
CATCHER_LENGTH_MM = 170.0                 # opening length, along the rail (x)
CATCHER_WIDTH_MM = 173.0                  # opening width, across the rail (y)
CATCHER_HEIGHT_MM = 101.0                 # rim above the floor
CATCHER_OFFSET_MM = (0.0, -89.6, -50.0)   # box bottom centre relative to marker 42 centre (x along rail, y across, z up)
CATCHER_CONTOUR_COLOR = (255, 255, 0)     # BGR: cyan
BALL_RADIUS_M = 0.020              # ping-pong ball radius (40 mm ball)
TRAIL_SUBSTEPS = 4                  # dots drawn between recorded ball positions (denser dotted line)
TRAIL_S = 0.5                       # the orange line behind the ball shows this many seconds of flight
SIDE_HEIGHT_ABOVE_RIM_M = 0.10      # ... and this far above the rim
LAUNCH_MARGIN_S = 0.3               # extra wait after the catcher should be in place
SHOT_TAIL_S = 0.3                    # keep the ball drawn briefly after impact


def estimate_travel_time(steps, vmax):
    """Estimated seconds for a trapezoidal move of `steps` pulses (same ramp as the firmware)."""
    a, v0 = ACCEL_PULSES_S2, START_PULSES_S
    vmax = max(float(vmax), v0)
    d_ramp = (vmax ** 2 - v0 ** 2) / (2.0 * a)
    if steps >= 2.0 * d_ramp:
        return 2.0 * (vmax - v0) / a + (steps - 2.0 * d_ramp) / vmax
    vp = (a * steps + v0 ** 2) ** 0.5
    return 2.0 * (vp - v0) / a


def ball_path_point(shot, t_rel):
    return shot['p0'] + shot['v0'] * t_rel + 0.5 * np.array([0.0, 0.0, -GRAVITY]) * t_rel * t_rel


def catch_result(shot, catcher_x_m):
    """Catch = the ball touches the catcher contour box (the same box that is drawn from marker 42).

    The ball's surface touches the box when its centre is within one ball radius of the box. The box is
    CATCHER_LENGTH x CATCHER_WIDTH, from the floor to the rim, centred on the bucket (catcher_x_m, y = 0).
    Returns (caught, reason).
    """
    r = BALL_RADIUS_M
    hx = CATCHER_LENGTH_MM / 2000.0 + r
    hy = CATCHER_WIDTH_MM / 2000.0 + r
    z_lo = FLOOR_Z_M - r
    z_hi = BUCKET_RIM_Z_M + r
    for t in np.linspace(0.0, shot['T'], 300):
        p = ball_path_point(shot, t)
        if abs(p[0] - catcher_x_m) <= hx and abs(p[1]) <= hy and z_lo <= p[2] <= z_hi:
            return True, 'ball touched the catcher'
    return False, 'no contact with the catcher'

def draw_path_segments(img, lp, pts3d, color=(0, 140, 255), thickness=2, margin=400):
    """Draw a 3-D path as 2-D line segments. Only pairs of points that are both in front of the camera and
    near the image are joined, so no line is drawn across a gap (behind the camera or far off the frame)."""
    pts3d = np.asarray(pts3d, np.float64).reshape(-1, 3)
    R, _ = cv2.Rodrigues(lp.rvec)
    cam = (R @ pts3d.T + lp.tvec).T
    pix, _ = cv2.projectPoints(pts3d.reshape(-1, 1, 3), lp.rvec, lp.tvec, lp.K, lp.dist)
    pix = pix.reshape(-1, 2)
    h, w = img.shape[:2]
    ok = (cam[:, 2] > 0.05) & (np.abs(pix[:, 0] - w / 2) < w / 2 + margin) & (np.abs(pix[:, 1] - h / 2) < h / 2 + margin)
    for i in range(len(pix) - 1):
        if ok[i] and ok[i + 1]:
            a = tuple(int(round(v)) for v in pix[i])
            b = tuple(int(round(v)) for v in pix[i + 1])
            cv2.line(img, a, b, color, thickness)


def in_frame(lp, p3d, w, h):
    """True only if the point is inside the image both with the plain pinhole model and with the lens
    distortion. Far off-axis points can fold back into the image under the distortion model, so the pinhole
    check excludes them."""
    p3d = np.asarray(p3d, np.float64).reshape(3)
    R, _ = cv2.Rodrigues(lp.rvec)
    c = R @ p3d + lp.tvec.ravel()
    if c[2] <= 0.05:
        return False
    fx, fy, cx, cy = lp.K[0, 0], lp.K[1, 1], lp.K[0, 2], lp.K[1, 2]
    u = fx * c[0] / c[2] + cx
    v = fy * c[1] / c[2] + cy
    if not (0 <= u < w and 0 <= v < h):
        return False
    pix, _ = cv2.projectPoints(p3d.reshape(1, 1, 3), lp.rvec, lp.tvec, lp.K, lp.dist)
    x, y = pix.ravel()
    return 0 <= x < w and 0 <= y < h


def draw_dots(img, lp, pts3d, color=(0, 140, 255), radius=3):
    """Draw the 3-D points as dots. Only dots that are in front of the camera and inside the image are drawn,
    so nothing is drawn outside the frame."""
    pts3d = np.asarray(pts3d, np.float64).reshape(-1, 3)
    if len(pts3d) == 0:
        return
    R, _ = cv2.Rodrigues(lp.rvec)
    cam = (R @ pts3d.T + lp.tvec).T
    pix, _ = cv2.projectPoints(pts3d.reshape(-1, 1, 3), lp.rvec, lp.tvec, lp.K, lp.dist)
    h, w = img.shape[:2]
    for p in pts3d:
        if in_frame(lp, p, w, h):
            x, y = cv2.projectPoints(p.reshape(1, 1, 3), lp.rvec, lp.tvec, lp.K, lp.dist)[0].ravel()
            cv2.circle(img, (int(round(x)), int(round(y))), radius, color, -1)


def catcher_box_corners(mm):
    """8 corners of the catcher box (0-3 bottom, 4-7 top), from the carriage position in mm."""
    centre = cf.marker_centre(42, mm / 1000.0)            # marker 42 centre for this carriage position
    base = centre + np.array(CATCHER_OFFSET_MM) / 1000.0
    L, W, H = CATCHER_LENGTH_MM / 1000.0, CATCHER_WIDTH_MM / 1000.0, CATCHER_HEIGHT_MM / 1000.0
    corners = []
    for z in (0.0, H):
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            corners.append(base + np.array([sx * L / 2, sy * W / 2, z]))
    return np.array(corners)


def draw_catcher_box(img, lp, mm, shift=(0.0, 0.0), color=CATCHER_CONTOUR_COLOR, thickness=2):
    """Wireframe contour of the catcher. `shift` (pixels) moves the whole box onto the current marker-42
    detection, so the contour follows the bucket with no delay. Each edge is drawn only where both ends
    are inside the frame."""
    c = catcher_box_corners(mm)
    h, w = img.shape[:2]
    R, _ = cv2.Rodrigues(lp.rvec)
    depth = (c @ R.T + lp.tvec.ravel())[:, 2]
    pix = cv2.projectPoints(c.reshape(-1, 1, 3), lp.rvec, lp.tvec, lp.K, lp.dist)[0].reshape(-1, 2)
    pix = pix + np.asarray(shift, np.float64)
    for a_i, b_i in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
        if depth[a_i] > 0.05 and depth[b_i] > 0.05 and            0 <= pix[a_i][0] < w and 0 <= pix[a_i][1] < h and 0 <= pix[b_i][0] < w and 0 <= pix[b_i][1] < h:
            cv2.line(img, tuple(int(round(v)) for v in pix[a_i]), tuple(int(round(v)) for v in pix[b_i]), color, thickness)


def draw_ball_box(img, lp, p3d, color=(0, 255, 255), thickness=2):
    """Square around the ball, sized from the ball's real radius and the camera's focal length."""
    p3d = np.asarray(p3d, np.float64).reshape(3)
    R, _ = cv2.Rodrigues(lp.rvec)
    depth = float((R @ p3d + lp.tvec.ravel())[2])
    if depth <= 0.05:
        return
    pix, _ = cv2.projectPoints(p3d.reshape(1, 1, 3), lp.rvec, lp.tvec, lp.K, lp.dist)
    x, y = pix.ravel()
    half = max(4.0, float(lp.K[0, 0]) * BALL_RADIUS_M / depth)
    cv2.rectangle(img, (int(round(x - half)), int(round(y - half))),
                  (int(round(x + half)), int(round(y + half))), color, thickness)


def draw_polyline(img, pts, color, thickness=2):
    pts = np.round(pts).astype(int)
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(img, tuple(a), tuple(b), color, thickness)


class SerialLink:
    def __init__(self):
        self.port = None
        self.lock = threading.Lock()
        self.lines = queue.Queue()
        self.alive = True
        self.connected_name = None
        threading.Thread(target=self.reader, daemon=True).start()
    def connect(self, name):
        self.close()
        self.port = serial.Serial(name, 115200, timeout=0.05, write_timeout=0.15)
        self.connected_name = name
        time.sleep(1.8)
    def close(self):
        with self.lock:
            if self.port:
                try: self.port.close()
                except Exception: pass
                self.port = None
            self.connected_name = None
    def send(self, cmd):
        with self.lock:
            if not self.port or not self.port.is_open: return False
            try:
                self.port.write((cmd+'\n').encode('ascii'))
                return True
            except (serial.SerialException, OSError):
                self.lines.put('DISCONNECTED')
                return False
    def reader(self):
        while self.alive:
            p = self.port
            if p and p.is_open:
                try:
                    line = p.readline().decode('utf-8', 'replace').strip()
                    if line: self.lines.put(line)
                except (serial.SerialException, OSError):
                    self.lines.put('DISCONNECTED')
                    time.sleep(0.1)
            else: time.sleep(0.1)


class Camera:
    """Reads the USB camera continuously and keeps only the newest frame."""
    def __init__(self, index):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened(): self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened(): raise RuntimeError('USB camera unavailable')
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
        self.cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.new_frame = threading.Event()
        self.frame = None
        self.frame_id = 0
        self.read_fps = 0.0
        self.read_ms = 0.0
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
    def _loop(self):
        last_t = None
        while self.running:
            t0 = time.perf_counter()
            ok, frame = self.cap.read()
            t1 = time.perf_counter()
            if not ok:
                time.sleep(0.005)
                continue
            with self.lock:
                self.frame = frame
                self.frame_id += 1
                self.read_ms = (t1 - t0) * 1000.0 if self.read_ms == 0 else 0.8*self.read_ms + 0.2*(t1-t0)*1000.0
                if last_t is not None and t1 > last_t:
                    inst = 1.0 / (t1 - last_t)
                    self.read_fps = inst if self.read_fps == 0 else 0.8*self.read_fps + 0.2*inst
            last_t = t1
            self.new_frame.set()
    def latest(self, last_id):
        with self.lock:
            self.new_frame.clear()
            if self.frame_id == last_id:
                return None, last_id
            return self.frame, self.frame_id
    def close(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


class Vision:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / 'catch_x.xml'))
        self.data = mujoco.MjData(self.model)
        j = self.model.joint('carriage_slide').id
        self.qaddr = int(self.model.jnt_qposadr[j]); self.daddr = int(self.model.jnt_dofadr[j])
        self.low, self.high = map(float, self.model.jnt_range[j])
        self.data.qpos[self.qaddr] = self.low
        bj = self.model.joint('ball_free').id
        self.model.geom_rgba[self.model.geom('ball_geom').id] = [1.0, 0.0, 0.0, 1.0]
        self.ball_qadr = int(self.model.jnt_qposadr[bj])
        self.data.qpos[self.ball_qadr:self.ball_qadr + 3] = [0.0, 0.0, -10.0]
        mujoco.mj_forward(self.model, self.data)
        # Twin renderer: same size as the camera frame, so the matched camera lines up
        self.render_w = min(CAPTURE_W, int(self.model.vis.global_.offwidth))
        self.render_h = min(CAPTURE_H, int(self.model.vis.global_.offheight))
        if (self.render_w, self.render_h) != (CAPTURE_W, CAPTURE_H):
            print(f'NOTE: XML framebuffer {self.render_w}x{self.render_h} is smaller than {CAPTURE_W}x{CAPTURE_H}; '
                  'add <visual><global offwidth="640" offheight="480"/></visual> to catch_x.xml for the matched view.')
        self.renderer = mujoco.Renderer(self.model, width=self.render_w, height=self.render_h)
        self.cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(self.model, self.cam)
        self.cam.lookat[:] = [0, 0.5, 1.2]
        self.cam.distance = 1.; self.cam.azimuth = -90; self.cam.elevation = -15
        self.cam_matched = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(self.model, self.cam_matched)
        self.lp = LivePose()
        self.catcher_shift = np.zeros(2)
        self.trail = deque(maxlen=400)   # (time, ball position) reached so far in this throw
        self.last_shift = None
        self.has_matched = False
        self.camera = Camera(CAMERA_INDEX)
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                                                cv2.aruco.DetectorParameters())
        self.state_lock = threading.Lock()
        self.home_ref = None; self.end_ref = None; self.measured = None
        self.position_mm = None
        self.cam_mm = None                       # camera-based carriage position (median), None if lost
        self.cam_mm_window = deque(maxlen=MM_WINDOW)
        self.cam_hist = deque(maxlen=900)        # (monotonic time, mm) for catch judgement
        self.pose_tracked = False
        self.last_twin = None
        self.last_render_at = 0.0
        self._last_render_perf = None
        self.metrics = dict(vision_fps=0.0, detect_ms=0.0, render_ms=0.0, render_fps=0.0, markers=0)
        self._last_vision_perf = None
        self.ball_shot = None

    def process(self, frame):
        """Run tracking, the overlay and the twin on one camera frame. Returns (camera_rgb, twin_rgb)."""
        t0 = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        t1 = time.perf_counter()
        raw = {}
        if ids is not None:
            for c, i in zip(corners, ids.flatten()):
                raw[int(i)] = c.reshape(4, 2)

        # Live camera pose from the fixed markers 10 and 23
        self.lp.update({k: v for k, v in raw.items() if k in (10, 23)})
        tracked = self.lp.valid
        cam_mm = None
        if tracked and 42 in raw:
            cam_mm = self.lp.carriage_mm_from_42(raw[42])

        # Rail-ratio position (fallback when the camera pose is not available)
        pos = {i: raw[i].mean(axis=0) for i in MARKERS if i in raw}
        measured = None
        if all(i in pos for i in MARKERS):
            a, b, p = (pos[i] for i in MARKERS)
            v = b - a; den = float(np.dot(v, v))
            if den > 1.: measured = float(np.dot(p - a, v) / den)

        with self.state_lock:
            self.measured = measured
            self.pose_tracked = tracked
            if cam_mm is not None:
                self.cam_mm_window.append(cam_mm)
                self.cam_mm = float(np.median(self.cam_mm_window))
                self.cam_hist.append((time.monotonic(), self.cam_mm))
                self.position_mm = self.cam_mm
            elif tracked and self.cam_mm is not None:
                pass                      # marker 42 briefly hidden: hold the last camera position
            else:
                self.cam_mm = None
                if measured is not None and self.home_ref is not None and self.end_ref is not None and abs(self.end_ref - self.home_ref) > 1e-5:
                    progress = float(np.clip((measured - self.home_ref) / (self.end_ref - self.home_ref), 0, 1))
                    self.position_mm = progress * TRAVEL_MM
            if self.position_mm is not None:
                # 3-D catcher follows the current frame's marker-42 reading (no median lag)
                self.data.qpos[self.qaddr] = (cam_mm if cam_mm is not None else self.position_mm) / 1000.0

            # Virtual ball
            ball_pos = None
            shot_landing_mm = 0.0
            shot = self.ball_shot
            if shot is not None:
                t_rel = max(time.monotonic() - shot['t0'], 0.0)
                if t_rel >= shot['T'] + SHOT_TAIL_S:
                    self.ball_shot = None
                    shot = None
            path_pts = None
            waiting = shot is not None and time.monotonic() < shot['t0']   # catcher still moving
            if shot is not None and not waiting:
                shot_landing_mm = shot['landing_mm']
                # Only the part of the path still ahead of the ball (from its position to the rim)
                path_pts = np.array([ball_path_point(shot, tt) for tt in np.linspace(min(t_rel, shot['T']), shot['T'], 60)])
                # Decide the catch once, at impact, from the camera carriage position
                if t_rel >= shot['T'] and 'caught' not in shot:
                    if self.cam_mm is not None:
                        shot['caught'], shot['reason'] = catch_result(shot, BUCKET_X_HOME_M + self.cam_mm / 1000.0)
                    else:
                        shot['caught'], shot['reason'] = False, 'no camera position'
                if shot.get('caught') and t_rel >= shot['T']:
                    # Caught: the ball stays inside the bucket, drawn on top of the model
                    ball_pos = np.array([BUCKET_X_HOME_M + shot['landing_mm'] / 1000.0, 0.0, BUCKET_RIM_Z_M + 0.02])
                else:
                    ball_pos = ball_path_point(shot, t_rel)
                self.data.qpos[self.ball_qadr:self.ball_qadr + 3] = ball_pos
                if not waiting:
                    self.trail.append((time.monotonic(), ball_pos.copy()))
                self.data.qpos[self.ball_qadr + 3:self.ball_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
            else:
                self.data.qpos[self.ball_qadr:self.ball_qadr + 3] = [0.0, 0.0, -10.0]
        if shot is None:
            self.trail.clear()
        ball_draw = ball_pos
        # The following line is shown only while the ball itself is inside the camera image
        ball_in_view = False
        if ball_draw is not None and self.lp.valid:
            ball_in_view = in_frame(self.lp, ball_draw, frame.shape[1], frame.shape[0])
        if not ball_in_view:
            self.trail.clear()        # only the part of the flight seen inside the image is kept
        catcher_mm = self.cam_mm if self.cam_mm is not None else self.position_mm
        raw_mm = cam_mm if cam_mm is not None else catcher_mm      # this frame's reading, used for the contour
        # Contour offset: the current marker-42 detection minus where the model puts marker 42 (no delay)
        if 42 in raw and raw_mm is not None and self.lp.valid:
            proj42 = self.lp.project(cf.marker_centre(42, raw_mm / 1000.0).reshape(1, 3))
            if len(proj42):
                self.catcher_shift = raw[42].mean(axis=0) - proj42[0]
        path_draw = path_pts

        # Twin render at RENDER_HZ, with the matched camera when the pose is tracked
        now = time.monotonic()
        if self.last_twin is None or now - self.last_render_at >= 1.0 / RENDER_HZ:
            r0 = time.perf_counter()
            self.data.qvel[self.daddr] = 0
            mujoco.mj_forward(self.model, self.data)
            if tracked:
                cam = mvc.mujoco_camera_from_pose(dict(rvec=self.lp.rvec.ravel().tolist(), tvec=self.lp.tvec.ravel().tolist()), self.lp.K)
                self.cam_matched.distance = 1.0
                self.cam_matched.lookat[:] = cam['pos'] + cam['fwd']
                self.cam_matched.azimuth = cam['az']
                self.cam_matched.elevation = cam['el']
                self.model.vis.global_.fovy = cam['fovy']
                view = self.cam_matched
                shift = np.float32([[1, 0, cam['dx']], [0, 1, cam['dy']]])
                self.last_shift = shift
                self.has_matched = True
            else:
                # Keep the last matched view rather than jumping to the unmatched default camera
                view = self.cam_matched if self.has_matched else self.cam
                shift = self.last_shift if self.has_matched else None
            self.renderer.update_scene(self.data, camera=view)
            img = cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)
            if shift is not None:
                img = cv2.warpAffine(img, shift, (self.render_w, self.render_h))
            if tracked:
                # Marker outlines (model corners, live pose) and the ball, on top of the render
                cur_mm = raw_mm if raw_mm is not None else 0.0      # this frame's marker-42 reading
                opt42 = self.lp.opts.get(42, (2, True))
                outlines = [self.lp.project(self.lp.obj[0:4]), self.lp.project(self.lp.obj[4:8]),
                            self.lp.project(cf.option_corners(cf.marker_centre(42, cur_mm / 1000.0), self.lp.half42, opt42)) + self.catcher_shift]
                for poly in outlines:
                    if len(poly) == 4:
                        cv2.polylines(img, [np.round(poly).astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
                if ball_in_view:
                    draw_dots(img, self.lp, self._trail_points(), (0, 0, 255), 3)
                if raw_mm is not None:
                    draw_catcher_box(img, self.lp, raw_mm, self.catcher_shift)
                if ball_in_view and ball_draw is not None:
                    draw_ball_box(img, self.lp, ball_draw)
            # Drawing above used BGR; the GUI shows RGB, so convert back here
            self.last_twin = cv2.cvtColor(cv2.resize(img, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
            r1 = time.perf_counter()
            self.last_render_at = now
            with self.state_lock:
                self.metrics['render_ms'] = self._ema(self.metrics['render_ms'], (r1 - r0) * 1000.0)
                if self._last_render_perf is not None and r1 > self._last_render_perf:
                    self.metrics['render_fps'] = self._ema(self.metrics['render_fps'], 1.0 / (r1 - self._last_render_perf))
            self._last_render_perf = r1

        # Real frame overlay: marker corners, model corners, virtual ball
        vis = frame.copy()
        for mid, d in raw.items():
            if mid in (10, 23, 42):
                for p in d:
                    cv2.circle(vis, (int(round(p[0])), int(round(p[1]))), 4, (0, 255, 0), 2)
        if tracked:
            for p in self.lp.project(self.lp.obj):
                cv2.circle(vis, (int(round(p[0])), int(round(p[1]))), 3, (0, 0, 255), -1)
            if ball_in_view:
                draw_dots(vis, self.lp, self._trail_points(), (0, 0, 255), 3)
            if self.lp.valid and raw_mm is not None:
                draw_catcher_box(vis, self.lp, raw_mm, self.catcher_shift)
            if ball_draw is not None:
                pb = self.lp.project(ball_draw.reshape(1, 3))
                if len(pb):
                    cv2.circle(vis, (int(pb[0][0]), int(pb[0][1])), 8, (0, 0, 255), -1)
                    if ball_in_view:
                        draw_ball_box(vis, self.lp, ball_draw)
        status = (f'pose {"OK" if tracked else "LOST"} RMS {self.lp.rms or 0:.2f}px  '
                  f'carriage {"%.0f mm (camera)" % self.cam_mm if self.cam_mm is not None else "%s" % ("%.0f mm" % self.position_mm if self.position_mm is not None else "unknown")}')
        cv2.putText(vis, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cam_rgb = cv2.cvtColor(cv2.resize(vis, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)

        t2 = time.perf_counter()
        with self.state_lock:
            m = self.metrics
            m['detect_ms'] = self._ema(m['detect_ms'], (t1 - t0) * 1000.0)
            if self._last_vision_perf is not None and t2 > self._last_vision_perf:
                m['vision_fps'] = self._ema(m['vision_fps'], 1.0 / (t2 - self._last_vision_perf))
            self._last_vision_perf = t2
            m['markers'] = len(raw)
        return cam_rgb, self.last_twin

    def _trail_points(self):
        now = time.monotonic()
        pts = [p for (tt, p) in self.trail if now - tt <= TRAIL_S]
        if len(pts) < 2:
            return np.array(pts).reshape(-1, 3)
        out = []
        for a, b in zip(pts[:-1], pts[1:]):
            for k in range(TRAIL_SUBSTEPS):
                out.append(a + (b - a) * k / TRAIL_SUBSTEPS)
        out.append(pts[-1])
        return np.array(out)

    @staticmethod
    def _ema(old, new, alpha=0.2):
        return new if old == 0 else (1 - alpha) * old + alpha * new

    def calibrate_point(self, which):
        with self.state_lock:
            if self.measured is None: raise ValueError('All three markers must be visible')
            if which == 'home': self.home_ref = self.measured
            else: self.end_ref = self.measured

    def close(self):
        self.camera.close()
        self.renderer.close()


class App:
    def __init__(self, root):
        self.root = root; root.title('INTERCEPT-X | Control, Digital Twin & Catch'); root.geometry('1250x980')
        self.link = SerialLink(); self.vision = None
        self.vision_queue = queue.Queue(maxsize=1)
        self.vision_stop = threading.Event()
        self.vision_thread = None
        self.telemetry = {}; self.last_poll = 0; self.cal_state = None; self.cal_deadline = 0
        self.auto_enabled = False; self.vision_error = None
        self.goto_pulses = None; self.goto_mm = None; self.goto_t0 = 0.0; self.goto_seen_running = False
        self.pos_hist = deque(maxlen=600)       # board pulses: (monotonic, mm, running)
        self.shot = None
        self.status = tk.StringVar(value='DISCONNECTED'); self.freq = tk.StringVar(value='3000'); self.steps = tk.StringVar(value='1000')
        self.target_mm = tk.StringVar(value='200')
        self.goto_text = tk.StringVar(value='Target: none')
        self.throw_T = tk.StringVar(value='0.6')
        self.prepos = tk.BooleanVar(value=False)      # True: move the catcher first, then launch
        self.throw_x = tk.StringVar(value='216')
        self.throw_text = tk.StringVar(value='Throw: none')
        self.position = tk.StringVar(value='Unknown'); self.speed = tk.StringVar(value='0')
        self.performance = tk.StringVar(value='Camera FPS: --  |  Vision FPS: --  |  Detect: -- ms  |  MuJoCo: -- ms  |  Render FPS: --  |  Markers: --')
        self.last_msg = tk.StringVar(value='Connect ESP32 and start camera')

        top = ttk.Frame(root, padding=8); top.pack(fill='x')
        ttk.Label(top, text='INTERCEPT-X', font=('Segoe UI', 22, 'bold')).pack(side='left', padx=10)
        self.ports = ttk.Combobox(top, width=16, values=self.port_names()); self.ports.pack(side='left', padx=5)
        ttk.Button(top, text='Refresh ports', command=lambda: self.ports.configure(values=self.port_names())).pack(side='left')
        ttk.Button(top, text='Connect', command=self.connect).pack(side='left', padx=5)
        ttk.Button(top, text='Disconnect', command=self.disconnect).pack(side='left', padx=5)
        ttk.Button(top, text='Start camera + MuJoCo', command=self.start_vision).pack(side='left', padx=8)
        ttk.Label(top, textvariable=self.status, font=('Segoe UI', 11, 'bold')).pack(side='right')

        controls = ttk.LabelFrame(root, text='Motor control', padding=10); controls.pack(fill='x', padx=12, pady=4)
        for name, cmd in [('◀ Reverse', lambda: self.move(-1)), ('Forward ▶', lambda: self.move(1)), ('HOME', lambda: self.send('HOME')), ('END', lambda: self.send('END')), ('STOP', lambda: self.stop_all())]:
            ttk.Button(controls, text=name, command=cmd).pack(side='left', padx=4)
        tk.Button(controls, text='SOFTWARE E-STOP', bg='#ba2020', fg='white', font=('Segoe UI', 11, 'bold'), command=lambda: self.send('SESTOP')).pack(side='left', padx=10)
        ttk.Button(controls, text='Reset SW E-stop', command=self.clear_stop).pack(side='left')

        target = ttk.LabelFrame(root, text='Go to position', padding=10)
        target.pack(fill='x', padx=12, pady=4)
        ttk.Label(target, text='Target position (mm, 0 = forward limit, 432 = reverse/END):').pack(side='left')
        entry = ttk.Entry(target, textvariable=self.target_mm, width=10); entry.pack(side='left', padx=6)
        entry.bind('<Return>', lambda e: self.go_to_position())
        ttk.Button(target, text='Go to position', command=self.go_to_position).pack(side='left', padx=4)
        ttk.Button(target, text='0 mm', command=lambda: self.set_target_and_go(0)).pack(side='left', padx=4)
        ttk.Button(target, text='216 mm', command=lambda: self.set_target_and_go(216)).pack(side='left', padx=4)
        ttk.Button(target, text='432 mm', command=lambda: self.set_target_and_go(432)).pack(side='left', padx=4)
        ttk.Label(target, textvariable=self.goto_text, font=('Segoe UI', 10, 'bold')).pack(side='left', padx=12)

        throw = ttk.LabelFrame(root, text='Virtual ball throw and catch', padding=10)
        throw.pack(fill='x', padx=12, pady=4)
        ttk.Label(throw, text='Flight time (s):').pack(side='left')
        ttk.Entry(throw, textvariable=self.throw_T, width=6).pack(side='left', padx=4)
        ttk.Label(throw, text='Landing (mm from HOME):').pack(side='left', padx=(12, 0))
        ttk.Entry(throw, textvariable=self.throw_x, width=6).pack(side='left', padx=4)
        ttk.Button(throw, text='THROW', command=self.throw_ball).pack(side='left', padx=8)
        ttk.Checkbutton(throw, text='Move catcher before launch', variable=self.prepos).pack(side='left', padx=8)
        ttk.Label(throw, textvariable=self.throw_text, font=('Segoe UI', 10, 'bold')).pack(side='left', padx=12)

        settings = ttk.Frame(root, padding=8); settings.pack(fill='x')
        ttk.Label(settings, text='Step count').pack(side='left'); ttk.Entry(settings, textvariable=self.steps, width=10).pack(side='left', padx=4)
        ttk.Label(settings, text='Target pulses/s').pack(side='left', padx=(18, 0)); ttk.Entry(settings, textvariable=self.freq, width=10).pack(side='left', padx=4)
        ttk.Button(settings, text='Apply speed (idle only)', command=self.set_freq).pack(side='left', padx=6)
        ttk.Button(settings, text='Calibrate HOME → END', command=self.calibrate).pack(side='left', padx=10)
        ttk.Button(settings, text='Enable autonomous input', command=self.toggle_auto).pack(side='left')

        info = ttk.LabelFrame(root, text='Live status', padding=10); info.pack(fill='x', padx=12, pady=6)
        self.labels = {}
        for name in ('END LIMIT D2', 'HOME LIMIT D3', 'PHYSICAL E-STOP A1', 'SOFTWARE E-STOP', 'MOTOR', 'CAMERA POSE', 'CALIBRATION'):
            f = ttk.Frame(info); f.pack(side='left', padx=8)
            ttk.Label(f, text=name).pack(); lab = tk.Label(f, text='UNKNOWN', width=13, bg='#777', fg='white'); lab.pack(); self.labels[name] = lab
        vals = ttk.Frame(root, padding=10); vals.pack(fill='x')
        ttk.Label(vals, text='Carriage position:').pack(side='left'); ttk.Label(vals, textvariable=self.position, font=('Segoe UI', 12, 'bold')).pack(side='left', padx=6)
        ttk.Label(vals, text='Motor frequency:').pack(side='left', padx=(25, 0)); ttk.Label(vals, textvariable=self.speed, font=('Segoe UI', 12, 'bold')).pack(side='left', padx=6)
        ttk.Label(vals, text='Travel: 432 mm / 34192 pulses nominal').pack(side='right')
        ttk.Label(root, textvariable=self.performance, font=('Consolas', 10), anchor='w').pack(fill='x', padx=12, pady=(0, 4))
        views = ttk.Frame(root); views.pack(fill='both', expand=True, padx=12, pady=8)
        self.camlabel = ttk.Label(views, text='Physical camera + overlay', anchor='center'); self.camlabel.pack(side='left', fill='both', expand=True)
        self.twinlabel = ttk.Label(views, text='MuJoCo twin (matched camera)', anchor='center'); self.twinlabel.pack(side='left', fill='both', expand=True)
        ttk.Label(root, textvariable=self.last_msg, anchor='w').pack(fill='x', padx=12, pady=4)
        root.bind('<Left>', lambda e: self.move(-1) if not isinstance(e.widget, (tk.Entry, ttk.Entry, ttk.Combobox)) else None)
        root.bind('<Right>', lambda e: self.move(1) if not isinstance(e.widget, (tk.Entry, ttk.Entry, ttk.Combobox)) else None)
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(GUI_REFRESH_MS, self.tick)

    def port_names(self): return [p.device for p in list_ports.comports()]
    def connect(self):
        try: self.link.connect(self.ports.get()); self.status.set('CONNECTED')
        except Exception as e: messagebox.showerror('Serial', str(e))
    def disconnect(self):
        if self.link.port and self.link.port.is_open:
            self.link.send('STOP')
        self.cal_state = None
        self.auto_enabled = False
        self.link.close()
        self.telemetry.clear()
        self.clear_goto('Disconnected')
        self.status.set('DISCONNECTED')
        self.speed.set('0')
        for key in ('END LIMIT D2', 'HOME LIMIT D3', 'PHYSICAL E-STOP A1', 'SOFTWARE E-STOP', 'MOTOR'):
            self.labels[key].configure(text='UNKNOWN', bg='#777')
        self.last_msg.set('Serial disconnected. Motor state cannot be verified until reconnected.')
    def send(self, cmd):
        if not self.link.send(cmd): self.last_msg.set('Serial not connected')
    def stop_all(self):
        self.clear_goto('Stopped by user')
        self.send('STOP')
    def move(self, direction):
        try:
            n = int(self.steps.get()); assert 0 < n <= 1000000
            self.send(f'M {direction*n}')
        except (ValueError, AssertionError): messagebox.showerror('Steps', 'Enter 1..1000000 steps')
    def set_freq(self):
        try:
            f = int(self.freq.get()); assert 1 <= f <= 15000
            self.send(f'F {f}')
        except (ValueError, AssertionError): messagebox.showerror('Speed', 'Enter 1..15000 pulses/s')
    def clear_stop(self):
        if messagebox.askyesno('Reset software stop', 'Confirm physical area is clear and the hardware E-stop is released?'):
            self.send('SCLEAR')
    def clear_goto(self, text='Target: none'):
        self.goto_pulses = None
        self.goto_mm = None
        self.goto_seen_running = False
        self.goto_text.set(text)
    def set_target_and_go(self, mm):
        self.target_mm.set(str(mm))
        self.go_to_position()
    def go_to_position(self):
        try:
            target = float(self.target_mm.get())
            assert 0.0 <= target <= TRAVEL_MM
        except (ValueError, AssertionError):
            messagebox.showerror('Target', f'Enter a position from 0 to {TRAVEL_MM:.0f} mm'); return
        t = self.telemetry
        if not t: messagebox.showwarning('Controller', 'Connect the ESP32 first (no telemetry yet)'); return
        if self.cal_state: messagebox.showwarning('Busy', 'Calibration is running'); return
        if t.get('running') or self.goto_pulses is not None or self.shot is not None:
            messagebox.showwarning('Busy', 'Motor is already moving or a throw is in flight. Press STOP first.'); return
        if t.get('estop') or t.get('swstop'):
            messagebox.showwarning('E-stop', 'Clear the E-stop first (hardware released, then Reset SW E-stop)'); return
        if not t.get('known'):
            messagebox.showwarning('Not homed', 'Position unknown. Run HOME then END (or Calibrate) first.'); return
        target_pulses = round(target * FULL_PULSES / TRAVEL_MM)
        delta = t['pulses'] - target_pulses
        if abs(delta) <= 2:
            self.goto_text.set(f'Already at {target:.1f} mm')
            return
        self.goto_pulses = target_pulses
        self.goto_mm = target
        self.goto_t0 = time.monotonic()
        self.goto_seen_running = False
        self.goto_text.set(f'Moving to {target:.1f} mm ...')
        self.send(f'M {delta}')
        self.last_msg.set(f'Go: target {target:.1f} mm = {target_pulses} pulses, sent M {delta}')

    def finish_goto(self, t):
        if t.get('estop') or t.get('swstop'):
            self.clear_goto('Aborted: E-stop')
            return
        if t['running']:
            self.goto_seen_running = True
            return
        if not self.goto_seen_running:
            if time.monotonic() - self.goto_t0 > GOTO_START_TIMEOUT_S:
                self.clear_goto('Did not start')
                self.last_msg.set('Motor did not start. Check the last serial message and limit switches.')
            return
        error_mm = (t['pulses'] - self.goto_pulses) * TRAVEL_MM / FULL_PULSES
        final_mm = t['pulses'] * TRAVEL_MM / FULL_PULSES
        target = self.goto_mm
        if abs(error_mm) <= GOTO_TOLERANCE_MM:
            self.goto_text.set(f'ARRIVED at {final_mm:.1f} mm (error {error_mm:+.2f} mm)')
        else:
            self.goto_text.set(f'NOT REACHED: stopped at {final_mm:.1f} mm, error {error_mm:+.1f} mm')
        self.last_msg.set(f'Target {target:.1f} mm, board stopped at {final_mm:.1f} mm. Reason: {t["reason"]}')
        self.clear_goto(self.goto_text.get())

    def throw_ball(self):
        try:
            T = float(self.throw_T.get()); assert THROW_MIN_T <= T <= THROW_MAX_T
            landing_mm = float(self.throw_x.get()); assert 0.0 <= landing_mm <= TRAVEL_MM
        except (ValueError, AssertionError):
            messagebox.showerror('Throw', f'Flight time {THROW_MIN_T}..{THROW_MAX_T} s, landing 0..{TRAVEL_MM:.0f} mm'); return
        t = self.telemetry
        if not t: messagebox.showwarning('Controller', 'Connect the ESP32 first (no telemetry yet)'); return
        if self.cal_state: messagebox.showwarning('Busy', 'Calibration is running'); return
        if t.get('running') or self.goto_pulses is not None or self.shot is not None:
            messagebox.showwarning('Busy', 'Motor is moving or a throw is in flight'); return
        if t.get('estop') or t.get('swstop'):
            messagebox.showwarning('E-stop', 'Clear the E-stop first'); return
        if not t.get('known'):
            messagebox.showwarning('Not homed', 'Position unknown. Run HOME then END first.'); return

        xL = BUCKET_X_HOME_M + landing_mm / 1000.0
        # Side launch: starts SIDE_OFFSET_M beside the bucket, SIDE_HEIGHT_ABOVE_RIM_M above the rim
        z0 = BUCKET_RIM_Z_M + SIDE_HEIGHT_ABOVE_RIM_M
        p0 = np.array([xL + SIDE_OFFSET_M, 0.0, z0], dtype=float)
        v0 = np.array([-SIDE_OFFSET_M / T, 0.0, (BUCKET_RIM_Z_M - z0 + 0.5 * GRAVITY * T * T) / T])

        # The ball is launched now and flies with physics for T seconds; the catcher moves at the same time
        target_pulses = round(landing_mm * FULL_PULSES / TRAVEL_MM)
        delta = t['pulses'] - target_pulses
        steps = abs(delta)
        est = estimate_travel_time(steps, t.get('freq', 3000)) if steps > 2 else 0.0
        if steps > 2:
            self.send(f'M {delta}')
        # Launch now, or after the catcher has moved (if 'Move catcher before launch' is ticked)
        t0 = time.monotonic() + ((est + LAUNCH_MARGIN_S) if self.prepos.get() else 0.0)
        self.shot = dict(t0=t0, T=T, landing_mm=landing_mm, est=est, p0=p0, v0=v0)
        if self.vision:
            with self.vision.state_lock:
                self.vision.ball_shot = self.shot
        verdict = 'catcher can arrive in time' if est <= T else f'catcher needs ~{est:.2f} s, may be late'
        self.throw_text.set(f'Flying: {T:.2f} s, {SIDE_OFFSET_M*100:.0f} cm side launch; {verdict}')
        self.last_msg.set(f'Thrown: landing {landing_mm:.1f} mm, flight {T:.2f} s, catcher {verdict}')

    def evaluate_shot(self):
        """Judge the catch at impact time. Uses the camera carriage position if available, else the board's."""
        s = self.shot
        self.shot = None
        if self.vision:
            with self.vision.state_lock:
                self.vision.ball_shot = None
        tc = s['t0'] + s['T']
        pulse_pts = [(tt, mm, rr) for (tt, mm, rr) in self.pos_hist if tt >= s['t0'] - 0.3]
        cam_pts = []
        if self.vision:
            with self.vision.state_lock:
                cam_pts = [(tt, mm) for (tt, mm) in self.vision.cam_hist if tt >= s['t0'] - 0.3]
        pulse_mm = None
        if len(pulse_pts) >= 2:
            pulse_mm = float(np.interp(tc, [p[0] for p in pulse_pts], [p[1] for p in pulse_pts]))
        cam_mm = None
        if len(cam_pts) >= 3:
            cam_mm = float(np.interp(tc, [p[0] for p in cam_pts], [p[1] for p in cam_pts]))
        catcher_mm = cam_mm if cam_mm is not None else pulse_mm
        source = 'camera' if cam_mm is not None else 'board'
        if catcher_mm is None:
            self.throw_text.set('MISSED: no position data (camera and board)'); return
        err = catcher_mm - s['landing_mm']
        moving = any(rr for (tt, _, rr) in pulse_pts if abs(tt - tc) < 0.15)
        ok, why = catch_result(s, BUCKET_X_HOME_M + catcher_mm / 1000.0)
        if ok:
            text = f'CAUGHT ({source}): catcher {catcher_mm:.1f} mm, ball {s["landing_mm"]:.1f} mm, offset {err:+.1f} mm'
        elif moving:
            text = f'MISSED ({source}): catcher still moving at impact (offset {err:+.1f} mm)'
        else:
            text = f'MISSED ({source}): {why} (offset {err:+.1f} mm)'
        if pulse_mm is not None and cam_mm is not None:
            text += f'  [board {pulse_mm:.1f} mm, camera {cam_mm:.1f} mm]'
        self.throw_text.set(text)
        self.last_msg.set(text)

    def start_vision(self):
        if self.vision or (self.vision_thread and self.vision_thread.is_alive()):
            self.last_msg.set('Camera is already starting or running (it takes a few seconds)')
            return
        self.vision_stop.clear()
        self.last_msg.set('Starting camera, pose tracking and MuJoCo in background...')
        self.vision_thread = threading.Thread(target=self.vision_worker, daemon=True)
        self.vision_thread.start()
    def vision_worker(self):
        vision = None
        try:
            vision = Vision()
            self.vision = vision
            last_id = -1
            while not self.vision_stop.is_set():
                if not vision.camera.new_frame.wait(0.1): continue
                frame, fid = vision.camera.latest(last_id)
                if frame is None: continue
                last_id = fid
                result = vision.process(frame)
                try: self.vision_queue.get_nowait()
                except queue.Empty: pass
                try: self.vision_queue.put_nowait(result)
                except queue.Full: pass
        except Exception as exc:
            self.vision_error = str(exc)
            traceback.print_exc()          # full error in the console
        finally:
            if vision is not None: vision.close()
            self.vision = None
    def calibrate(self):
        if not self.vision: messagebox.showwarning('Camera', 'Start camera first'); return
        if not self.link.port: messagebox.showwarning('Controller', 'Connect ESP32 first'); return
        if not messagebox.askyesno('Automatic calibration', 'The catcher will move to HOME then END. Clear the entire rail and verify both limit switches and hardware E-stop. Continue?'): return
        self.cal_state = 'home'; self.cal_deadline = time.monotonic()+15
        self.send('HOME'); self.last_msg.set('Calibration: moving HOME')
    def handle_cal(self):
        if not self.cal_state: return
        t = self.telemetry
        if time.monotonic() > self.cal_deadline or t.get('estop') or t.get('swstop'):
            self.send('STOP'); self.cal_state = None; self.last_msg.set('Calibration aborted'); return
        if self.cal_state == 'home' and t.get('fwd') and not t.get('running'):
            try: self.vision.calibrate_point('home')
            except ValueError: self.last_msg.set('HOME reached but marker missing; calibration aborted'); self.cal_state = None; return
            self.cal_state = 'end'; self.cal_deadline = time.monotonic()+15; self.send('END'); self.last_msg.set('Calibration: moving END')
        elif self.cal_state == 'end' and t.get('rev') and not t.get('running'):
            try: self.vision.calibrate_point('end')
            except ValueError: self.last_msg.set('END reached but marker missing'); self.cal_state = None; return
            self.cal_state = None; self.last_msg.set(f'Calibration complete: HOME={self.vision.home_ref:.5f} END={self.vision.end_ref:.5f}')
    def set_light(self, key, active, text=None):
        self.labels[key].configure(text=text or ('ACTIVE' if active else 'OK'), bg=('#ba2020' if active else '#19734a'))
    def toggle_auto(self):
        self.auto_enabled = not self.auto_enabled
        self.last_msg.set('Autonomous commands ' + ('enabled' if self.auto_enabled else 'disabled'))
    def parse_line(self, line):
        if line.startswith('TEL,'):
            a = line.split(',', 11)
            if len(a) < 12: return
            try:
                self.telemetry = dict(running=bool(int(a[1])), rev=bool(int(a[2])), fwd=bool(int(a[3])), estop=bool(int(a[4])), swstop=bool(int(a[5])), known=bool(int(a[6])), freq=int(a[7]), current=float(a[8]), pulses=int(a[9]), remaining=int(a[10]), reason=a[11])
            except ValueError: return
            t = self.telemetry
            self.set_light('END LIMIT D2', t['rev'])
            self.set_light('HOME LIMIT D3', t['fwd'])
            self.set_light('PHYSICAL E-STOP A1', t['estop'])
            self.set_light('SOFTWARE E-STOP', t['swstop'])
            self.set_light('MOTOR', t['running'], 'RUNNING' if t['running'] else 'IDLE')
            self.speed.set(f"{t['current']:.0f} / {t['freq']} pulses/s")
            self.handle_cal()
            if t['known']:
                self.pos_hist.append((time.monotonic(), t['pulses'] * TRAVEL_MM / FULL_PULSES, t['running']))
            if self.goto_pulses is not None:
                self.finish_goto(t)
        elif line == 'DISCONNECTED': self.status.set('DISCONNECTED'); self.cal_state = None; self.clear_goto('Disconnected')
        else: self.last_msg.set(line)
    def show_image(self, label, rgb):
        photo = ImageTk.PhotoImage(Image.fromarray(rgb)); label.configure(image=photo, text=''); label.image = photo
    def tick(self):
        while not self.link.lines.empty(): self.parse_line(self.link.lines.get_nowait())
        now = time.monotonic()
        if now-self.last_poll > TELEM_POLL_S and self.link.port:
            self.link.send('TELEM'); self.last_poll = now
        if self.shot is not None and now >= self.shot['t0'] + self.shot['T'] + SHOT_TAIL_S:
            self.evaluate_shot()
        if self.vision_error:
            self.last_msg.set(f'Vision error: {self.vision_error}')
            self.vision_error = None
        result = None
        try:
            while True: result = self.vision_queue.get_nowait()
        except queue.Empty: pass
        if result is not None:
            self.show_image(self.camlabel, result[0])
            self.show_image(self.twinlabel, result[1])
            vision = self.vision
            if vision:
                with vision.state_lock:
                    tracked = vision.pose_tracked
                    cam_mm = vision.cam_mm
                    calibrated = vision.home_ref is not None and vision.end_ref is not None
                    pos = vision.position_mm
                    metrics = vision.metrics.copy()
                cam = vision.camera
                with cam.lock:
                    cam_fps, cam_ms = cam.read_fps, cam.read_ms
                self.performance.set(
                    f"Camera FPS: {cam_fps:.1f} (read {cam_ms:.1f} ms)  |  Vision FPS: {metrics['vision_fps']:.1f}  |  "
                    f"Detect: {metrics['detect_ms']:.1f} ms  |  MuJoCo: {metrics['render_ms']:.1f} ms  |  "
                    f"Render FPS: {metrics['render_fps']:.1f}  |  Markers: {metrics['markers']}")
                self.set_light('CAMERA POSE', False, 'TRACKING' if tracked else 'LOST')
                self.set_light('CALIBRATION', False, 'DONE' if calibrated or cam_mm is not None else 'PENDING')
                self.position.set(f'{pos:.1f} mm' if pos is not None else 'Unknown')
        self.root.after(GUI_REFRESH_MS, self.tick)
    def close(self):
        self.send('STOP'); self.link.alive = False; self.link.close()
        self.vision_stop.set()
        if self.vision_thread and self.vision_thread.is_alive():
            self.vision_thread.join(timeout=1.0)
        self.root.destroy()


if __name__ == '__main__':
    root = tk.Tk(); App(root); root.mainloop()
