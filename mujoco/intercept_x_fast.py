"""INTERCEPT-X | USB serial motor controller + physical ArUco + unchanged MuJoCo XML.

Vision pipeline:
  * Camera reads run in their own thread and only the NEWEST frame is kept.
  * ArUco rail tracking (markers 23, 10, 42) gives the carriage position.
  * MuJoCo twin follows the carriage position at RENDER_HZ.
  * Target-position box: type a position in mm and press Go. The controller moves the
    carriage there using pulse counts after HOME/END are known, and reports arrival.

Install: pip install pyserial opencv-contrib-python mujoco numpy pillow
Run beside catch_x.xml and meshes/. Always use a hardware E-stop.
"""
from pathlib import Path
import queue, threading, time, tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import cv2
import mujoco
import serial
from serial.tools import list_ports
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parent
CAMERA_INDEX = 0
CAPTURE_W, CAPTURE_H = 640, 480      # lower = faster; 1280x720 was the main cost
CAPTURE_FPS = 30
USE_CLAHE = False                    # True only if lighting is poor (costly per frame)
RENDER_HZ = 15                       # MuJoCo twin refresh rate
DISPLAY_W, DISPLAY_H = 580, 330      # size of the two preview panes
GUI_REFRESH_MS = 15                  # Tk polling interval
MARKERS = (23, 10, 42)
TRAVEL_MM = 432.0
FULL_PULSES = 34192
GOTO_TOLERANCE_MM = 1.0              # arrival tolerance
GOTO_START_TIMEOUT_S = 2.0           # motor must start within this time after Go
ACCEL_PULSES_S2 = 20000.0            # must match firmware ACCEL
START_PULSES_S = 500.0               # must match firmware START_FREQ


def estimate_travel_time(steps, vmax):
    """Estimated seconds for a trapezoidal move of `steps` pulses (same ramp as the firmware)."""
    a, v0 = ACCEL_PULSES_S2, START_PULSES_S
    vmax = max(float(vmax), v0)
    d_ramp = (vmax ** 2 - v0 ** 2) / (2.0 * a)          # distance spent accelerating (one side)
    if steps >= 2.0 * d_ramp:                           # full trapezoid: ramp up, cruise, ramp down
        return 2.0 * (vmax - v0) / a + (steps - 2.0 * d_ramp) / vmax
    vp = (a * steps + v0 ** 2) ** 0.5                   # triangle: never reaches vmax
    return 2.0 * (vp - v0) / a


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
        self.metrics = dict(vision_fps=0.0, detect_ms=0.0, render_ms=0.0, render_fps=0.0, markers=0)
        self._last_vision_perf = None

    def process(self, frame):
        """Run ArUco tracking and the twin on one camera frame. Returns (camera_rgb, twin_rgb)."""
        t0 = time.perf_counter()
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
            v = b - a; den = float(np.dot(v, v))
            if den > 1.: measured = float(np.dot(p - a, v) / den)

        with self.state_lock:
            self.measured = measured
            if measured is not None and self.home_ref is not None and self.end_ref is not None and abs(self.end_ref - self.home_ref) > 1e-5:
                progress = float(np.clip((measured - self.home_ref) / (self.end_ref - self.home_ref), 0, 1))
                self.position_mm = progress * TRAVEL_MM
                self.data.qpos[self.qaddr] = self.low + progress * (self.high - self.low)
            display_position = self.position_mm

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
        self.auto_enabled = False; self.vision_error = None
        # Go-to-position state
        self.goto_pulses = None; self.goto_mm = None; self.goto_t0 = 0.0; self.goto_seen_running = False
        self.goto_run_t0 = None          # PC time when the motor was first seen running (fallback timing)
        self.last_elapsed_s = None       # firmware-measured move time from the 'Elapsed:' line
        self.travel_text = tk.StringVar(value='Last travel time: --')
        self.status = tk.StringVar(value='DISCONNECTED'); self.freq = tk.StringVar(value='3000'); self.steps = tk.StringVar(value='1000')
        self.target_mm = tk.StringVar(value='200')
        self.goto_text = tk.StringVar(value='Target: none')
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

        target = ttk.LabelFrame(root, text='Go to position (test before autonomous use)', padding=10)
        target.pack(fill='x', padx=12, pady=4)
        ttk.Label(target, text='Target position (mm, 0 = forward limit, 432 = reverse/END):').pack(side='left')
        entry = ttk.Entry(target, textvariable=self.target_mm, width=10); entry.pack(side='left', padx=6)
        entry.bind('<Return>', lambda e: self.go_to_position())
        ttk.Button(target, text='Go to position', command=self.go_to_position).pack(side='left', padx=4)
        ttk.Button(target, text='Position 0 mm', command=lambda: self.set_target_and_go(0)).pack(side='left', padx=4)
        ttk.Button(target, text='Position 216 mm', command=lambda: self.set_target_and_go(216)).pack(side='left', padx=4)
        ttk.Button(target, text='Position 432 mm', command=lambda: self.set_target_and_go(432)).pack(side='left', padx=4)
        ttk.Label(target, textvariable=self.goto_text, font=('Segoe UI', 10, 'bold')).pack(side='left', padx=12)
        ttk.Label(target, textvariable=self.travel_text, font=('Segoe UI', 10)).pack(side='left', padx=12)

        settings = ttk.Frame(root, padding=8); settings.pack(fill='x')
        ttk.Label(settings, text='Step count').pack(side='left'); ttk.Entry(settings, textvariable=self.steps, width=10).pack(side='left', padx=4)
        ttk.Label(settings, text='Target pulses/s').pack(side='left', padx=(18, 0)); ttk.Entry(settings, textvariable=self.freq, width=10).pack(side='left', padx=4)
        ttk.Button(settings, text='Apply speed (idle only)', command=self.set_freq).pack(side='left', padx=6)
        ttk.Button(settings, text='Calibrate HOME → END', command=self.calibrate).pack(side='left', padx=10)
        ttk.Button(settings, text='Enable autonomous input', command=self.toggle_auto).pack(side='left')

        info = ttk.LabelFrame(root, text='Live status', padding=10); info.pack(fill='x', padx=12, pady=6)
        self.labels = {}
        for name in ('END LIMIT D2', 'HOME LIMIT D3', 'PHYSICAL E-STOP A1', 'SOFTWARE E-STOP', 'MOTOR', 'CAMERA', 'CALIBRATION'):
            f = ttk.Frame(info); f.pack(side='left', padx=8)
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
        self.goto_run_t0 = None
        self.goto_text.set(text)
    def set_target_and_go(self, mm):
        self.target_mm.set(str(mm))
        self.go_to_position()
    def go_to_position(self):
        """Move the carriage to the typed position (mm) using pulse counts. Needs HOME/END done first."""
        try:
            target = float(self.target_mm.get())
            assert 0.0 <= target <= TRAVEL_MM
        except (ValueError, AssertionError):
            messagebox.showerror('Target', f'Enter a position from 0 to {TRAVEL_MM:.0f} mm'); return
        t = self.telemetry
        if not t: messagebox.showwarning('Controller', 'Connect the ESP32 first (no telemetry yet)'); return
        if self.cal_state: messagebox.showwarning('Busy', 'Calibration is running'); return
        if t.get('running') or self.goto_pulses is not None:
            messagebox.showwarning('Busy', 'Motor is already moving. Press STOP first.'); return
        if t.get('estop') or t.get('swstop'):
            messagebox.showwarning('E-stop', 'Clear the E-stop first (hardware released, then Reset SW E-stop)'); return
        if not t.get('known'):
            messagebox.showwarning('Not homed', 'Position unknown. Run HOME then END (or Calibrate) first.'); return
        target_pulses = round(target * FULL_PULSES / TRAVEL_MM)
        delta = t['pulses'] - target_pulses          # M +n moves forward (pulses decrease)
        if abs(delta) <= 2:
            self.goto_text.set(f'Already at {target:.1f} mm')
            self.last_msg.set(f'Already within 2 pulses of {target:.1f} mm')
            return
        est = estimate_travel_time(abs(delta), t.get('freq', 3000))
        self.goto_pulses = target_pulses
        self.goto_mm = target
        self.goto_t0 = time.monotonic()
        self.goto_seen_running = False
        self.goto_run_t0 = None
        self.last_elapsed_s = None
        self.goto_text.set(f'Moving to {target:.1f} mm (est. {est:.2f} s)...')
        self.send(f'M {delta}')
        self.last_msg.set(f'Go: target {target:.1f} mm = {target_pulses} pulses, sent M {delta}')

    def finish_goto(self, t):
        """Called from parse_line on each telemetry line while a Go is pending."""
        if t.get('estop') or t.get('swstop'):
            self.clear_goto('Aborted: E-stop')
            self.last_msg.set('Go aborted by E-stop / software stop')
            return
        if t['running']:
            self.goto_seen_running = True
            if self.goto_run_t0 is None:
                self.goto_run_t0 = time.monotonic()
            return
        if not self.goto_seen_running:
            if time.monotonic() - self.goto_t0 > GOTO_START_TIMEOUT_S:
                self.clear_goto('Did not start')
                self.last_msg.set('Motor did not start. Check the last serial message and limit switches.')
            return
        # Motor was running and has now stopped.
        error_mm = (t['pulses'] - self.goto_pulses) * TRAVEL_MM / FULL_PULSES
        final_mm = t['pulses'] * TRAVEL_MM / FULL_PULSES
        target = self.goto_mm
        if self.last_elapsed_s is not None:
            travel_s, src = self.last_elapsed_s, 'firmware'
        else:
            travel_s = time.monotonic() - (self.goto_run_t0 or time.monotonic())
            src = 'PC, +/-0.25 s'
        travel_txt = f'{travel_s:.2f} s ({src})'
        self.travel_text.set(f'Last travel time: {travel_txt}')
        if abs(error_mm) <= GOTO_TOLERANCE_MM:
            self.goto_text.set(f'ARRIVED at {final_mm:.1f} mm (error {error_mm:+.2f} mm) in {travel_s:.2f} s')
            self.last_msg.set(f'Arrived at {final_mm:.1f} mm for target {target:.1f} mm in {travel_txt}. Reason: {t["reason"]}')
        else:
            self.goto_text.set(f'NOT REACHED: stopped at {final_mm:.1f} mm, error {error_mm:+.1f} mm after {travel_s:.2f} s')
            self.last_msg.set(f'Stopped at {final_mm:.1f} mm, target {target:.1f} mm after {travel_txt}. Reason: {t["reason"]}')
        self.clear_goto(self.goto_text.get())

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
        """For a planning algorithm on the Tk main thread. Same pulse maths as Go to position."""
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
            if self.goto_pulses is not None:
                self.finish_goto(t)
        elif line == 'DISCONNECTED': self.status.set('DISCONNECTED'); self.cal_state = None; self.clear_goto('Disconnected')
        elif line.startswith('Elapsed:'):
            # Firmware stop report: "Elapsed: 1.234 s (1234.00 ms)" = measured motion time
            try:
                self.last_elapsed_s = float(line.split()[1])
                if self.goto_pulses is None:
                    self.travel_text.set(f'Last travel time: {self.last_elapsed_s:.2f} s (firmware)')
            except (IndexError, ValueError): pass
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
                cam = vision.camera
                with cam.lock:
                    cam_fps, cam_ms = cam.read_fps, cam.read_ms
                self.performance.set(
                    f"Camera FPS: {cam_fps:.1f} (read {cam_ms:.1f} ms)  |  Vision FPS: {metrics['vision_fps']:.1f}  |  "
                    f"Detect: {metrics['detect_ms']:.1f} ms  |  MuJoCo: {metrics['render_ms']:.1f} ms  |  "
                    f"Render FPS: {metrics['render_fps']:.1f}  |  Markers: {metrics['markers']}/3")
                self.set_light('CAMERA', False, 'TRACKING' if measured is not None else 'MARKER LOST')
                self.set_light('CALIBRATION', False, 'DONE' if calibrated else 'PENDING')
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
