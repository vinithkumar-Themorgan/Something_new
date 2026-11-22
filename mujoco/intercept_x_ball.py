"""INTERCEPT-X | USB serial motor controller + physical ArUco + red-ball tracking + unchanged MuJoCo XML.

Vision pipeline:
  * Camera reads run in their own thread and only the NEWEST frame is kept.
  * Red ball only: HSV mask keeps red pixels; the most circular red blob is the ball.
  * Ball 3D position from the blob's apparent size (pinhole model), mapped to the MuJoCo world
    with a calibration file (camera_calibration.json) recorded from the GUI.
  * The MuJoCo 'virtual_ball' body follows the measured position and velocity, and MuJoCo
    integrates gravity between camera frames. Without calibration the ball stays hidden.
  * ArUco rail tracking is unchanged.

Install: pip install pyserial opencv-contrib-python mujoco numpy pillow
Run beside catch_x.xml and meshes/. Always use a hardware E-stop.
"""
from pathlib import Path
import json, queue, threading, time, tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import numpy as np
import cv2
import mujoco
import serial
from serial.tools import list_ports
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parent
CAMERA_INDEX = 0
CAPTURE_W, CAPTURE_H = 640, 480
CAPTURE_FPS = 30
USE_CLAHE = False
RENDER_HZ = 15
DISPLAY_W, DISPLAY_H = 580, 330
GUI_REFRESH_MS = 15
MARKERS = (23, 10, 42)
TRAVEL_MM = 432.0
FULL_PULSES = 34192

# ---- Red ball tracking ----
BALL_DIAMETER_M = 0.040            # measure your real ball (sim ball radius is 0.02 m)
BALL_MIN_RADIUS_PX = 6             # ignore blobs smaller than this
BALL_MIN_CIRCULARITY = 0.6         # 1.0 = perfect circle
BALL_HSV_RED_LOW = ((0, 120, 70), (170, 120, 70))      # two hue ranges wrap around red
BALL_HSV_RED_HIGH = ((10, 255, 255), (180, 255, 255))
BALL_LOST_PREDICT_S = 0.15         # predict with gravity this long after losing the ball, then hold
BALL_VELOCITY_MAX_GAP_S = 0.2      # no velocity estimate across longer gaps
CALIB_FILE = ROOT / 'camera_calibration.json'
MORPH_KERNEL = np.ones((5, 5), np.uint8)


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
        time.sleep(1.8)  # board may reboot on opening COM
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
        """Return (frame, id) if a frame newer than last_id exists, else (None, last_id)."""
        with self.lock:
            self.new_frame.clear()
            if self.frame_id == last_id:
                return None, last_id
            return self.frame, self.frame_id
    def close(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


def detect_red_ball(frame_bgr):
    """Return (u, v, radius_px) of the most circular red blob, or None. Non-red colours are ignored."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = None
    for low, high in zip(BALL_HSV_RED_LOW, BALL_HSV_RED_HIGH):
        part = cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8))
        mask = part if mask is None else cv2.bitwise_or(mask, part)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0.0
    min_area = np.pi * BALL_MIN_RADIUS_PX ** 2
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area <= best_area:
            continue
        perimeter = cv2.arcLength(c, True)
        circularity = 4.0 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0
        if circularity < BALL_MIN_CIRCULARITY:
            continue
        best, best_area = c, area
    if best is None:
        return None
    (u, v), r = cv2.minEnclosingCircle(best)
    return float(u), float(v), float(r)


def ball_camera_point(u, v, r, width, height):
    """Ball centre in the camera frame (metres), pinhole model.

    X and Y do not depend on the focal length. Z does, but the affine calibration absorbs that scale.
    """
    d = 2.0 * r
    cx, cy = width / 2.0, height / 2.0
    X = (u - cx) * BALL_DIAMETER_M / d
    Y = (v - cy) * BALL_DIAMETER_M / d
    Z = 500.0 * BALL_DIAMETER_M / d   # 500 px ~ focal length at 640 px width (approximate)
    return np.array([X, Y, Z])


class BallCalibration:
    """Affine map camera frame -> MuJoCo world, fitted from recorded (camera point, world point) pairs."""
    def __init__(self):
        self.M = None          # 4x3 matrix: world = [cam, 1] @ M
        self.points = []       # list of {'cam': [x,y,z], 'world': [x,y,z]}
        self.load()
    def load(self):
        try:
            data = json.loads(CALIB_FILE.read_text())
            self.points = data.get('points', [])
            self.M = np.array(data['M'], dtype=float) if data.get('M') else None
        except (OSError, ValueError, KeyError):
            self.M = None
    def save(self):
        CALIB_FILE.write_text(json.dumps({'M': self.M.tolist(), 'points': self.points}, indent=2))
    def fit(self):
        if len(self.points) < 4:
            raise ValueError('Need at least 4 points (8+ recommended)')
        A = np.array([[*p['cam'], 1.0] for p in self.points])
        B = np.array([p['world'] for p in self.points], dtype=float)
        if np.linalg.matrix_rank(A) < 4:
            raise ValueError('Points are too coplanar. Hold the ball at different heights, depths and sides, '
                             'not all in one plane or along one line.')
        M = np.linalg.lstsq(A, B, rcond=None)[0]
        err = np.linalg.norm(A @ M - B, axis=1)
        self.M = M
        return float(err.mean()), float(err.max())
    def to_world(self, cam_point):
        return np.append(cam_point, 1.0) @ self.M


class BallSim:
    """Drives the MuJoCo 'virtual_ball' body from measurements, with MuJoCo gravity between frames."""
    def __init__(self, model, data):
        self.model = model
        self.data = data
        jid = model.joint('ball_free').id
        self.qpos_adr = int(model.jnt_qposadr[jid])
        self.qvel_adr = int(model.jnt_dofadr[jid])
        geom = model.geom('ball_geom').id
        model.geom_rgba[geom] = [1.0, 0.0, 0.0, 1.0]     # red ball in the twin
        self.visible = False
        self.prev_pos = None
        self.prev_t = None
        self.vel = np.zeros(3)
        self.last_seen = None
        self.synced_at = None
        self.frozen = False
        self._hide()
    def _hide(self):
        self._set(np.array([0.0, 0.0, -10.0]), np.zeros(3))
    def _set(self, pos, vel):
        self.data.qpos[self.qpos_adr:self.qpos_adr + 3] = pos
        self.data.qpos[self.qpos_adr + 3:self.qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[self.qvel_adr:self.qvel_adr + 3] = vel
        self.data.qvel[self.qvel_adr + 3:self.qvel_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
    def update(self, now, pos_world):
        if pos_world is not None:
            if self.prev_pos is not None and 0 < now - self.prev_t < BALL_VELOCITY_MAX_GAP_S:
                v_new = (pos_world - self.prev_pos) / (now - self.prev_t)
                self.vel = 0.5 * v_new + 0.5 * self.vel
            else:
                self.vel = np.zeros(3)
            self.prev_pos = pos_world.copy()
            self.prev_t = now
            self.last_seen = now
            self.synced_at = now
            self.frozen = False
            self.visible = True
            self._set(pos_world, self.vel)
        elif self.last_seen is not None and not self.frozen:
            if now - self.last_seen < BALL_LOST_PREDICT_S:
                dt = now - self.synced_at
                steps = min(int(dt / self.model.opt.timestep), 200)
                for _ in range(steps):
                    mujoco.mj_step(self.model, self.data)
                self.synced_at = now
            else:
                # Lost for too long: hold where it was last seen (likely caught or out of view).
                self.frozen = True
                pos = self.data.qpos[self.qpos_adr:self.qpos_adr + 3].copy()
                self._set(pos, np.zeros(3))


class Vision:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(ROOT/'catch_x.xml'))
        self.data = mujoco.MjData(self.model)
        j = self.model.joint('carriage_slide').id
        self.qaddr = int(self.model.jnt_qposadr[j]); self.daddr = int(self.model.jnt_dofadr[j])
        self.low, self.high = map(float, self.model.jnt_range[j])
        self.data.qpos[self.qaddr] = self.low
        mujoco.mj_forward(self.model, self.data)
        self.cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(self.model, self.cam)
        self.cam.lookat[:] = [0, 0.5, 1.2]
        self.cam.distance = 1.; self.cam.azimuth = -90; self.cam.elevation = -15
        self.renderer = mujoco.Renderer(self.model,
                                        width=min(640, int(self.model.vis.global_.offwidth)),
                                        height=min(360, int(self.model.vis.global_.offheight)))
        self.camera = Camera(CAMERA_INDEX)
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                                                cv2.aruco.DetectorParameters())
        self.clahe = cv2.createCLAHE(2.0, (8, 8))
        self.state_lock = threading.Lock()
        self.home_ref = None; self.end_ref = None; self.measured = None
        self.position_mm = None
        self.last_twin = None
        self.last_render_at = 0.0
        self._last_render_perf = None
        self.calib = BallCalibration()
        self.ball = BallSim(self.model, self.data)
        self.last_ball_cam = None        # camera-frame ball centre when detected now, else None
        self.ball_radius_px = 0.0
        self.metrics = dict(vision_fps=0.0, detect_ms=0.0, render_ms=0.0, render_fps=0.0, markers=0)
        self._last_vision_perf = None

    def process(self, frame):
        """Run ball + ArUco detection and the twin on one camera frame. Returns (camera_rgb, twin_rgb)."""
        t0 = time.perf_counter()
        h, w = frame.shape[:2]

        # Red ball (runs on the raw colour frame, before any overlays are drawn)
        ball = detect_red_ball(frame)
        ball_cam = None
        ball_world = None
        if ball is not None:
            u, v, r = ball
            ball_cam = ball_camera_point(u, v, r, w, h)
            if self.calib.M is not None:
                ball_world = self.calib.to_world(ball_cam)
            cv2.circle(frame, (int(u), int(v)), max(int(r), 3), (0, 0, 255), 3)

        # ArUco rail tracking (unchanged logic)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if USE_CLAHE:
            gray = self.clahe.apply(gray)
        corners, ids, _ = self.detector.detectMarkers(gray)
        t1 = time.perf_counter()

        pos = {}
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for c, i in zip(corners, ids.flatten()):
                pos[int(i)] = c.reshape(4, 2).mean(axis=0)
        measured = None
        if all(i in pos for i in MARKERS):
            a, b, p = (pos[i] for i in MARKERS)
            vv = b - a; den = float(np.dot(vv, vv))
            if den > 1.: measured = float(np.dot(p - a, vv) / den)

        with self.state_lock:
            self.measured = measured
            if measured is not None and self.home_ref is not None and self.end_ref is not None and abs(self.end_ref - self.home_ref) > 1e-5:
                progress = float(np.clip((measured - self.home_ref) / (self.end_ref - self.home_ref), 0, 1))
                self.position_mm = progress * TRAVEL_MM
                self.data.qpos[self.qaddr] = self.low + progress * (self.high - self.low)
            display_position = self.position_mm
            self.last_ball_cam = ball_cam
            self.ball_radius_px = ball[2] if ball is not None else 0.0

        # Ball state in MuJoCo (gravity between frames, measurement when visible)
        self.ball.update(time.monotonic(), ball_world)

        # MuJoCo render at RENDER_HZ; reuse the last twin image otherwise.
        now = time.monotonic()
        if self.last_twin is None or now - self.last_render_at >= 1.0 / RENDER_HZ:
            r0 = time.perf_counter()
            self.data.qvel[self.daddr] = 0
            mujoco.mj_forward(self.model, self.data)
            self.renderer.update_scene(self.data, camera=self.cam)
            self.last_twin = cv2.resize(self.renderer.render(), (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_LINEAR)
            r1 = time.perf_counter()
            self.last_render_at = now
            with self.state_lock:
                self.metrics['render_ms'] = self._ema(self.metrics['render_ms'], (r1 - r0) * 1000.0)
                if self._last_render_perf is not None and r1 > self._last_render_perf:
                    self.metrics['render_fps'] = self._ema(self.metrics['render_fps'], 1.0 / (r1 - self._last_render_perf))
            self._last_render_perf = r1

        cv2.putText(frame, f'Markers: {sorted(pos)} | Camera pos: {f"{display_position:.1f} mm" if display_position is not None else "uncalibrated"}',
                    (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cam_rgb = cv2.cvtColor(cv2.resize(frame, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)

        t2 = time.perf_counter()
        with self.state_lock:
            m = self.metrics
            m['detect_ms'] = self._ema(m['detect_ms'], (t1 - t0) * 1000.0)
            if self._last_vision_perf is not None and t2 > self._last_vision_perf:
                m['vision_fps'] = self._ema(m['vision_fps'], 1.0 / (t2 - self._last_vision_perf))
            self._last_vision_perf = t2
            m['markers'] = len(pos)
        return cam_rgb, self.last_twin

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
        self.root = root; root.title('INTERCEPT-X | Control & Digital Twin'); root.geometry('1250x900')
        self.link = SerialLink(); self.vision = None
        self.vision_queue = queue.Queue(maxsize=1)
        self.vision_stop = threading.Event()
        self.vision_thread = None
        self.telemetry = {}; self.last_poll = 0; self.cal_state = None; self.cal_deadline = 0
        self.auto_enabled = False; self.pending_auto = None; self.vision_error = None
        self.status = tk.StringVar(value='DISCONNECTED'); self.freq = tk.StringVar(value='3000'); self.steps = tk.StringVar(value='1000')
        self.position = tk.StringVar(value='Unknown'); self.speed = tk.StringVar(value='0')
        self.performance = tk.StringVar(value='Camera FPS: --  |  Vision FPS: --  |  Detect: -- ms  |  MuJoCo: -- ms  |  Render FPS: --  |  Markers: --  |  Ball r: --')
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
        for name, cmd in [('◀ Reverse', lambda: self.move(-1)), ('Forward ▶', lambda: self.move(1)), ('HOME', lambda: self.send('HOME')), ('END', lambda: self.send('END')), ('STOP', lambda: self.send('STOP'))]:
            ttk.Button(controls, text=name, command=cmd).pack(side='left', padx=4)
        tk.Button(controls, text='SOFTWARE E-STOP', bg='#ba2020', fg='white', font=('Segoe UI', 11, 'bold'), command=lambda: self.send('SESTOP')).pack(side='left', padx=10)
        ttk.Button(controls, text='Reset SW E-stop', command=self.clear_stop).pack(side='left')
        settings = ttk.Frame(root, padding=8); settings.pack(fill='x')
        ttk.Label(settings, text='Step count').pack(side='left'); ttk.Entry(settings, textvariable=self.steps, width=10).pack(side='left', padx=4)
        ttk.Label(settings, text='Target pulses/s').pack(side='left', padx=(18, 0)); ttk.Entry(settings, textvariable=self.freq, width=10).pack(side='left', padx=4)
        ttk.Button(settings, text='Apply speed (idle only)', command=self.set_freq).pack(side='left', padx=6)
        ttk.Button(settings, text='Calibrate HOME → END', command=self.calibrate).pack(side='left', padx=10)
        ttk.Button(settings, text='Enable autonomous input', command=self.toggle_auto).pack(side='left')
        ballbar = ttk.LabelFrame(root, text='Red ball calibration (hold ball still in catcher at known world position)', padding=8)
        ballbar.pack(fill='x', padx=12, pady=2)
        ttk.Button(ballbar, text='Add calibration point', command=self.add_ball_point).pack(side='left', padx=4)
        ttk.Button(ballbar, text='Fit & save calibration', command=self.save_ball_calibration).pack(side='left', padx=4)
        ttk.Button(ballbar, text='Clear points', command=self.clear_ball_points).pack(side='left', padx=4)
        info = ttk.LabelFrame(root, text='Live status', padding=10); info.pack(fill='x', padx=12, pady=6)
        self.labels = {}
        for name in ('END LIMIT D2', 'HOME LIMIT D3', 'PHYSICAL E-STOP A1', 'SOFTWARE E-STOP', 'MOTOR', 'CAMERA', 'CALIBRATION', 'BALL'):
            f = ttk.Frame(info); f.pack(side='left', padx=6)
            ttk.Label(f, text=name).pack(); lab = tk.Label(f, text='UNKNOWN', width=13, bg='#777', fg='white'); lab.pack(); self.labels[name] = lab
        vals = ttk.Frame(root, padding=10); vals.pack(fill='x')
        ttk.Label(vals, text='Camera position:').pack(side='left'); ttk.Label(vals, textvariable=self.position, font=('Segoe UI', 12, 'bold')).pack(side='left', padx=6)
        ttk.Label(vals, text='Motor frequency:').pack(side='left', padx=(25, 0)); ttk.Label(vals, textvariable=self.speed, font=('Segoe UI', 12, 'bold')).pack(side='left', padx=6)
        ttk.Label(vals, text='Travel: 432 mm / 34192 pulses nominal').pack(side='right')
        ttk.Label(root, textvariable=self.performance, font=('Consolas', 10), anchor='w').pack(fill='x', padx=12, pady=(0, 4))
        views = ttk.Frame(root); views.pack(fill='both', expand=True, padx=12, pady=8)
        self.camlabel = ttk.Label(views, text='Physical camera', anchor='center'); self.camlabel.pack(side='left', fill='both', expand=True)
        self.twinlabel = ttk.Label(views, text='MuJoCo twin', anchor='center'); self.twinlabel.pack(side='left', fill='both', expand=True)
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
        # Request a stop while the serial connection is still open.
        # A disconnected PC cannot guarantee the motor stopped: hardware safety remains essential.
        if self.link.port and self.link.port.is_open:
            self.link.send('STOP')
        self.cal_state = None
        self.auto_enabled = False
        self.link.close()
        self.telemetry.clear()
        self.status.set('DISCONNECTED')
        self.speed.set('0')
        for key in ('END LIMIT D2', 'HOME LIMIT D3', 'PHYSICAL E-STOP A1', 'SOFTWARE E-STOP', 'MOTOR'):
            self.labels[key].configure(text='UNKNOWN', bg='#777')
        self.last_msg.set('Serial disconnected. Motor state cannot be verified until reconnected.')
    def send(self, cmd):
        if not self.link.send(cmd): self.last_msg.set('Serial not connected')
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
    def start_vision(self):
        if self.vision: return
        self.vision_stop.clear()
        self.last_msg.set('Starting camera and MuJoCo in background...')
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
        finally:
            if vision is not None: vision.close()
            self.vision = None
    def add_ball_point(self):
        v = self.vision
        if not v: messagebox.showwarning('Camera', 'Start camera first'); return
        with v.state_lock:
            cam = None if v.last_ball_cam is None else v.last_ball_cam.copy()
        if cam is None:
            messagebox.showwarning('Ball', 'Red ball not visible right now. Hold it still where the camera can see it.')
            return
        s = simpledialog.askstring('Ball world position',
            'Ball centre in MuJoCo world metres: x y z\n\n'
            'Catcher centre = (-0.219 + carriage_m, 0, 0.844)\n'
            'carriage_m = carriage slide in metres (0 to 0.432, from the XML).\n'
            'Use points with different heights and depths (not all in one plane).',
            parent=self.root)
        if not s: return
        try:
            xyz = [float(t) for t in s.replace(',', ' ').split()]
            assert len(xyz) == 3
        except (ValueError, AssertionError):
            messagebox.showerror('Ball', 'Enter exactly three numbers: x y z'); return
        v.calib.points.append({'cam': cam.tolist(), 'world': xyz})
        self.last_msg.set(f'Ball calibration points recorded: {len(v.calib.points)} (fit & save when at least 8)')
    def save_ball_calibration(self):
        v = self.vision
        if not v: messagebox.showwarning('Camera', 'Start camera first'); return
        try:
            mean_err, max_err = v.calib.fit()
            v.calib.save()
            messagebox.showinfo('Ball calibration',
                f'Saved to {CALIB_FILE.name}\nMean error {mean_err*1000:.1f} mm, max error {max_err*1000:.1f} mm')
        except ValueError as e:
            messagebox.showerror('Ball calibration', str(e))
    def clear_ball_points(self):
        if not self.vision: messagebox.showwarning('Camera', 'Start camera first'); return
        if messagebox.askyesno('Ball calibration', 'Delete all recorded ball calibration points?'):
            self.vision.calib.points = []
            self.vision.calib.M = None
            self.last_msg.set('Ball calibration cleared (save to write the empty file)')
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
    def autonomous_move_mm(self, target_mm):
        """Call from a planning algorithm on the Tk main thread. Requires homed pulse position."""
        if not self.auto_enabled: return False
        t = self.telemetry
        if not t or t.get('running') or t.get('estop') or t.get('swstop') or not t.get('known'): return False
        target_mm = float(np.clip(target_mm, 0, TRAVEL_MM))
        delta = t['pulses'] - round(target_mm*FULL_PULSES/TRAVEL_MM)
        if delta: self.send(f'M {delta}')
        return True
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
        elif line == 'DISCONNECTED': self.status.set('DISCONNECTED'); self.cal_state = None
        else: self.last_msg.set(line)
    def show_image(self, label, rgb):
        photo = ImageTk.PhotoImage(Image.fromarray(rgb)); label.configure(image=photo, text=''); label.image = photo
    def tick(self):
        while not self.link.lines.empty(): self.parse_line(self.link.lines.get_nowait())
        now = time.monotonic()
        if now-self.last_poll > 0.25 and self.link.port:
            self.link.send('TELEM'); self.last_poll = now
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
                    measured = vision.measured
                    calibrated = vision.home_ref is not None and vision.end_ref is not None
                    pos = vision.position_mm
                    metrics = vision.metrics.copy()
                    ball_seen = vision.last_ball_cam is not None
                    ball_r = vision.ball_radius_px
                cam = vision.camera
                with cam.lock:
                    cam_fps, cam_ms = cam.read_fps, cam.read_ms
                self.performance.set(
                    f"Camera FPS: {cam_fps:.1f} (read {cam_ms:.1f} ms)  |  Vision FPS: {metrics['vision_fps']:.1f}  |  "
                    f"Detect: {metrics['detect_ms']:.1f} ms  |  MuJoCo: {metrics['render_ms']:.1f} ms  |  "
                    f"Render FPS: {metrics['render_fps']:.1f}  |  Markers: {metrics['markers']}/3  |  Ball r: {ball_r:.0f}px")
                self.set_light('CAMERA', False, 'TRACKING' if measured is not None else 'MARKER LOST')
                self.set_light('CALIBRATION', False, 'DONE' if calibrated else 'PENDING')
                if vision.calib.M is None:
                    self.set_light('BALL', True, 'NEEDS CALIB')
                elif ball_seen:
                    self.set_light('BALL', False, 'TRACKED')
                else:
                    self.set_light('BALL', False, 'NO BALL')
                self.position.set(f'{pos:.1f} mm' if pos is not None else 'Not calibrated')
        self.root.after(GUI_REFRESH_MS, self.tick)
    def close(self):
        self.send('STOP'); self.link.alive = False; self.link.close()
        self.vision_stop.set()
        if self.vision_thread and self.vision_thread.is_alive():
            self.vision_thread.join(timeout=1.0)
        self.root.destroy()


if __name__ == '__main__':
    root = tk.Tk(); App(root); root.mainloop()
