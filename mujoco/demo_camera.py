"""Catcher X desktop controller.
Run: pip install -r requirements.txt && python catcher_ui.py
UI / camera and localhost automation all share the same serial command channel.
"""
import json
import math
import queue
import socketserver
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

import cv2
import serial
from serial.tools import list_ports
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parent
CAL_FILE = ROOT / 'camera_calibration.json'
TRAVEL_MM = 432.0


class CameraWorker:
    def __init__(self):
        self.cap = None
        self.thread = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.frame = None
        self.marker_px = None
        self.last_seen = 0.0
        self.marker_id = 0
        self.index = 0

    def start(self, index, marker_id):
        self.stop()
        self.index = index
        self.marker_id = marker_id
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if hasattr(cv2, 'CAP_DSHOW') and __import__('sys').platform == 'win32' else cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f'Cannot open camera {index}')
        self.cap = cap
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        aruco = getattr(cv2, 'aruco', None)
        if aruco is None:
            self.stop_event.set()
            return
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters()) if hasattr(aruco, 'ArucoDetector') else None
        try:
            while not self.stop_event.is_set():
                ok, frame = self.cap.read()
                if not ok:
                    time.sleep(.05)
                    continue
                if detector:
                    corners, ids, _ = detector.detectMarkers(frame)
                else:
                    corners, ids, _ = aruco.detectMarkers(frame, dictionary)
                marker = None
                if ids is not None:
                    for k, marker_id in enumerate(ids.flatten()):
                        if int(marker_id) == self.marker_id:
                            pts = corners[k].reshape(-1, 2)
                            marker = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
                            cv2.polylines(frame, [pts.astype('int32')], True, (40, 220, 90), 2)
                            cv2.circle(frame, (int(marker[0]), int(marker[1])), 5, (0, 255, 0), -1)
                            break
                with self.lock:
                    self.frame = frame.copy()
                    if marker is not None:
                        self.marker_px = marker
                        self.last_seen = time.monotonic()
        finally:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

    def snapshot(self):
        with self.lock:
            return (None if self.frame is None else self.frame.copy(),
                    self.marker_px, self.last_seen)

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)
        self.thread = None


class SerialBridge:
    def __init__(self, on_line):
        self.conn = None
        self.on_line = on_line
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

    def connect(self, port):
        self.close()
        self.stop_event.clear()
        self.conn = serial.Serial(port, 115200, timeout=.1, write_timeout=.15)
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while not self.stop_event.is_set():
            try:
                line = self.conn.readline().decode('utf-8', 'replace').strip()
                if line:
                    self.on_line(line)
            except (serial.SerialException, OSError, AttributeError):
                self.on_line('EVENT,SERIAL_DISCONNECTED')
                break

    def send(self, command):
        with self.lock:
            if not self.conn or not self.conn.is_open:
                raise RuntimeError('Not connected to controller')
            self.conn.write((command + '\n').encode('ascii'))

    def close(self):
        self.stop_event.set()
        if self.conn:
            try:
                self.conn.close()
            except serial.SerialException:
                pass
            self.conn = None


class AutoServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class AutoHandler(socketserver.StreamRequestHandler):
    def handle(self):
        self.request.settimeout(2)
        try:
            obj = json.loads(self.rfile.readline(1024).decode())
            if not isinstance(obj, dict):
                raise ValueError('Expected JSON object')
            cmd = obj.get('cmd', '')
            allowed = {'GOTO', 'STOP', 'ESTOP', 'AUTOX', 'AUTOXY'}
            if cmd not in allowed:
                raise ValueError('Unknown autonomous command')
            if cmd in {'GOTO', 'AUTOX', 'AUTOXY'}:
                n = float(obj.get('mm') if cmd == 'GOTO' else obj.get('pixel_x'))
                if not math.isfinite(n):
                    raise ValueError('Non-finite number')
                obj['value'] = n
                if cmd == 'AUTOXY':
                    obj['pixel_y'] = float(obj.get('pixel_y'))
                    if not math.isfinite(obj['pixel_y']): raise ValueError('Non-finite pixel_y')
            self.server.app.auto_requests.put(obj)
            result = {'accepted_for_processing': True, 'note': 'Firmware may reject an unsafe command; monitor EVENT telemetry.'}
        except Exception as ex:
            result = {'accepted_for_processing': False, 'error': str(ex)}
        self.wfile.write((json.dumps(result) + '\n').encode())


class App:
    def __init__(self, root):
        self.root = root
        self.root.title('CATCHER X  |  ESP32-S3 Motion Controller')
        self.root.geometry('1100x840')
        self.root.minsize(890, 690)
        self.camera = CameraWorker()
        self.lines = queue.Queue()
        self.auto_requests = queue.Queue()
        self.serial = SerialBridge(self.lines.put)
        self.connected = False
        self.auto_mode = tk.StringVar(value='MANUAL')
        self.port = tk.StringVar()
        self.speed = tk.IntVar(value=3000)
        self.steps = tk.StringVar(value='1000')
        self.target = tk.StringVar(value='216')
        self.camera_index = tk.StringVar(value='0')
        self.marker_id = tk.StringVar(value='0')
        self.camera_status = tk.StringVar(value='Camera OFF')
        self.mode_label = tk.StringVar(value='DISCONNECTED')
        self.position_label = tk.StringVar(value='Position: uncalibrated')
        self.speed_label = tk.StringVar(value='Speed: 0 pulses/s')
        self.last_label = tk.StringVar(value='Last travel: --')
        self.steps_label = tk.StringVar(value='Stroke pulses: --')
        self.event_label = tk.StringVar(value='Ready')
        self.physical = tk.BooleanVar(value=False)
        self.software = tk.BooleanVar(value=False)
        self.home_limit = tk.BooleanVar(value=False)
        self.end_limit = tk.BooleanVar(value=False)
        self.valid = False
        self.calibrated = False
        self.running = False
        self.jog_held = None
        self.last_jog_send = 0
        self.cal_home_px = None
        self.cal_end_px = None
        self.camera_cal = self._load_cal()
        self.photo = None
        self._make_ui()
        self.refresh_ports()
        self.server = AutoServer(('127.0.0.1', 8765), AutoHandler)
        self.server.app = self
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.root.bind('<KeyPress-Left>', lambda e: self.key_jog('L', e))
        self.root.bind('<KeyRelease-Left>', lambda e: self.key_stop('L', e))
        self.root.bind('<KeyPress-Right>', lambda e: self.key_jog('R', e))
        self.root.bind('<KeyRelease-Right>', lambda e: self.key_stop('R', e))
        self.root.after(50, self.tick)
        self.root.protocol('WM_DELETE_WINDOW', self.close)

    def _load_cal(self):
        try:
            d = json.loads(CAL_FILE.read_text())
            if math.hypot(d['end_px'] - d['home_px'], d.get('end_py', 0) - d.get('home_py', 0)) < 30:
                return None
            return d
        except (OSError, KeyError, ValueError, TypeError):
            return None

    def _make_ui(self):
        shell = ttk.Frame(self.root, padding=14)
        shell.pack(fill='both', expand=True)
        title = ttk.Frame(shell); title.pack(fill='x')
        ttk.Label(title, text='CATCHER X', font=('Segoe UI', 22, 'bold')).pack(side='left')
        ttk.Label(title, text='  /  Motion + Safety Dashboard', font=('Segoe UI', 12)).pack(side='left')
        bar = ttk.Frame(shell); bar.pack(fill='x', pady=12)
        ttk.Label(bar, text='Serial port').pack(side='left')
        self.ports = ttk.Combobox(bar, textvariable=self.port, width=18, state='readonly')
        self.ports.pack(side='left', padx=6)
        ttk.Button(bar, text='Refresh', command=self.refresh_ports).pack(side='left', padx=3)
        ttk.Button(bar, text='Connect', command=self.connect).pack(side='left', padx=3)
        ttk.Button(bar, text='Disconnect', command=self.disconnect).pack(side='left', padx=3)
        ttk.Label(bar, textvariable=self.mode_label, font=('Segoe UI', 10, 'bold')).pack(side='right')

        status = ttk.LabelFrame(shell, text='Live safety inputs  •  HIGH = released / LOW = active', padding=10)
        status.pack(fill='x', pady=(0, 10))
        self.indicators = {}
        for label, key in [('HOME D3', 'home'), ('END D2', 'end'), ('PHYSICAL E-STOP A1', 'estop'), ('SOFTWARE E-STOP', 'soft')]:
            frame = ttk.Frame(status); frame.pack(side='left', expand=True)
            c = tk.Canvas(frame, width=18, height=18, highlightthickness=0)
            c.pack(side='left', padx=(3, 5))
            circle = c.create_oval(2, 2, 16, 16, fill='#9ca3af', outline='')
            ttk.Label(frame, text=label).pack(side='left')
            self.indicators[key] = (c, circle)

        ctrl = ttk.Frame(shell); ctrl.pack(fill='x', pady=4)
        left = ttk.LabelFrame(ctrl, text='Manual control', padding=12)
        left.pack(side='left', fill='both', expand=True, padx=(0, 8))
        modebar=ttk.Frame(left); modebar.pack(fill='x', pady=5)
        ttk.Radiobutton(modebar, text='Manual', variable=self.auto_mode, value='MANUAL', command=self.mode_change).pack(side='left', padx=5)
        ttk.Radiobutton(modebar, text='Autonomous', variable=self.auto_mode, value='AUTO', command=self.mode_change).pack(side='left', padx=5)
        ttk.Label(left, text='Hold arrow buttons / keyboard ← → to jog').pack(pady=7)
        arrows = ttk.Frame(left); arrows.pack(pady=7)
        for name, direction in [('◀  HOME side', 'L'), ('END side  ▶', 'R')]:
            b = ttk.Button(arrows, text=name, width=18)
            b.pack(side='left', padx=9)
            b.bind('<ButtonPress-1>', lambda e, d=direction: self.begin_jog(d))
            b.bind('<ButtonRelease-1>', lambda e, d=direction: self.end_jog(d))
        quick = ttk.Frame(left); quick.pack(fill='x', pady=8)
        for name, cmd in [('HOME (D3)', 'HOME'), ('CENTER', 'CENTER'), ('END (D2)', 'END')]:
            ttk.Button(quick, text=name, command=lambda c=cmd: self.command(c)).pack(side='left', expand=True, padx=4)
        row = ttk.Frame(left); row.pack(fill='x', pady=5)
        ttk.Label(row, text='Absolute target (mm)').pack(side='left')
        ttk.Entry(row, textvariable=self.target, width=10).pack(side='left', padx=7)
        ttk.Button(row, text='GOTO', command=self.goto).pack(side='left')
        row = ttk.Frame(left); row.pack(fill='x', pady=5)
        ttk.Label(row, text='Relative steps').pack(side='left')
        ttk.Entry(row, textvariable=self.steps, width=12).pack(side='left', padx=7)
        ttk.Button(row, text='STEP +', command=lambda: self.step(1)).pack(side='left', padx=4)
        ttk.Button(row, text='STEP −', command=lambda: self.step(-1)).pack(side='left')
        row = ttk.Frame(left); row.pack(fill='x', pady=10)
        ttk.Label(row, text='Target frequency (pulses/s)').pack(anchor='w')
        ttk.Scale(row, from_=100, to=15000, variable=self.speed, command=lambda x: self.speed_label_local()).pack(side='left', fill='x', expand=True)
        self.speed_number=ttk.Label(row, text='3000', width=6)
        self.speed_number.pack(side='left', padx=4)
        ttk.Button(row, text='Apply', command=self.set_speed).pack(side='left', padx=5)
        ttk.Label(left, text='400 driver pulses/rev  •  RPM = pulses/s × 60 / 400').pack(anchor='w')
        ctl=ttk.Frame(left);ctl.pack(fill='x', pady=14)
        ttk.Button(ctl, text='Controlled STOP', command=lambda: self.command('STOP')).pack(side='left', expand=True, padx=4)
        ttk.Button(ctl, text='Clear software E-STOP', command=lambda: self.command('CLEAR')).pack(side='left', expand=True, padx=4)
        tk.Button(left, text='SOFTWARE EMERGENCY STOP', bg='#b91c1c', fg='white', font=('Segoe UI', 13, 'bold'), height=2, command=lambda: self.command('ESTOP')).pack(fill='x', pady=4)

        right = ttk.LabelFrame(ctrl, text='Position and calibration', padding=12)
        right.pack(side='left', fill='both', expand=True)
        for var in [self.position_label,self.speed_label,self.steps_label,self.last_label,self.event_label]:
            ttk.Label(right, textvariable=var, font=('Consolas', 11)).pack(anchor='w', pady=4)
        ttk.Separator(right).pack(fill='x', pady=10)
        ttk.Button(right, text='CALIBRATE: HOME → END → CENTER', command=self.calibrate).pack(fill='x', pady=5)
        ttk.Label(right, text='Calibration pauses at HOME and END to record ArUco marker coordinates.', wraplength=350).pack(anchor='w')
        retry = ttk.Frame(right); retry.pack(fill='x', pady=3)
        ttk.Button(retry, text='Retry HOME capture', command=lambda: self.capture_cal_marker('HOME')).pack(side='left', padx=3)
        ttk.Button(retry, text='Retry END capture', command=lambda: self.capture_cal_marker('END')).pack(side='left', padx=3)
        cam_row=ttk.Frame(right);cam_row.pack(fill='x', pady=8)
        ttk.Label(cam_row,text='Camera #').pack(side='left')
        ttk.Entry(cam_row,textvariable=self.camera_index,width=4).pack(side='left',padx=5)
        ttk.Label(cam_row,text='Marker ID').pack(side='left',padx=5)
        ttk.Entry(cam_row,textvariable=self.marker_id,width=5).pack(side='left',padx=5)
        ttk.Button(cam_row,text='Start camera',command=self.start_camera).pack(side='left',padx=4)
        ttk.Label(right,textvariable=self.camera_status).pack(anchor='w')
        self.preview = ttk.Label(right, text='Camera preview', anchor='center')
        self.preview.pack(fill='both', expand=True, pady=5)

        logframe=ttk.LabelFrame(shell, text='Controller events and diagnostics', padding=8)
        logframe.pack(fill='both', expand=True, pady=10)
        self.logbox=tk.Text(logframe,height=8,wrap='word',font=('Consolas',9))
        self.logbox.pack(fill='both',expand=True)
        ttk.Label(shell, text='Autonomous interface: TCP JSON on 127.0.0.1:8765  •  GUI Manual/Autonomous mode arbitrates commands  •  D3=HOME, D2=END').pack(anchor='w')

    def refresh_ports(self):
        values = [p.device for p in list_ports.comports()]
        self.ports['values'] = values
        if values and self.port.get() not in values:
            self.port.set(values[0])

    def connect(self):
        try:
            self.serial.connect(self.port.get())
            self.connected = True
            self.log('Connected to '+self.port.get())
            self.command('STATUS')
        except Exception as e:
            messagebox.showerror('Serial connection', str(e))

    def disconnect(self):
        if self.connected:
            try: self.serial.send('STOP')
            except Exception: pass
        self.serial.close();self.connected=False
        self.mode_label.set('DISCONNECTED')

    def command(self, cmd):
        try:
            self.serial.send(cmd)
            self.log('> '+cmd)
        except Exception as e:
            self.log('Error: '+str(e))

    def mode_change(self):
        self.end_jog()
        self.command('STOP')
        self.log('Control mode: '+self.auto_mode.get())

    def begin_jog(self, direction):
        if self.auto_mode.get() != 'MANUAL':
            return
        if self.jog_held != direction:
            self.jog_held=direction
            self.command('JOG '+direction)
            self.last_jog_send=time.monotonic()

    def end_jog(self, direction=None):
        if self.jog_held is not None and (direction is None or self.jog_held == direction):
            self.jog_held=None
            self.command('JOGSTOP')

    def key_jog(self, d, e):
        if isinstance(e.widget,(tk.Entry,ttk.Entry,ttk.Combobox)):
            return
        self.begin_jog(d)

    def key_stop(self,d,e):
        self.end_jog(d)

    def speed_label_local(self):
        self.speed_number.configure(text=str(int(self.speed.get())))

    def set_speed(self):
        self.command('F '+str(int(self.speed.get())))

    def goto(self):
        try:
            mm=float(self.target.get())
            if not 0 <= mm <= TRAVEL_MM or not math.isfinite(mm): raise ValueError()
            self.command(f'GOTO {mm:.2f}')
        except ValueError:
            messagebox.showerror('Target','Use a number from 0 to 432 mm')

    def step(self,sign):
        try:
            n=int(self.steps.get())
            if n <= 0 or n > 1000000: raise ValueError()
            self.command(f'STEP {sign*n}')
        except ValueError:
            messagebox.showerror('Steps','Enter 1 to 1000000 steps')

    def start_camera(self):
        try:
            self.camera.start(int(self.camera_index.get()),int(self.marker_id.get()))
            self.camera_status.set('Camera running; looking for marker ID '+self.marker_id.get())
        except Exception as e:
            messagebox.showerror('Camera',str(e))

    def calibrate(self):
        if self.camera.cap is None:
            messagebox.showwarning('Calibration','Start camera before calibration; marker must remain visible at HOME and END.')
            return
        self.cal_home_px=None;self.cal_end_px=None
        self.command('CAL')
        self.log('Camera calibration started. Ensure rail is clear.')

    def capture_cal_marker(self, which):
        _, marker, last_seen = self.camera.snapshot()
        if marker is None or time.monotonic()-last_seen > .3:
            self.log(f'Cannot capture {which}: marker not visible. Controller is PAUSED; fix camera then press CALNEXT manually via diagnostic if needed.')
            return
        if self.mode_label.get() != ('CAL_HOLD_HOME' if which=='HOME' else 'CAL_HOLD_END'):
            self.log(f'Not waiting for {which} camera capture')
            return
        if which=='HOME':
            self.cal_home_px = marker
            self.log(f'Captured HOME marker pixel={marker}')
            self.command('CALNEXT')
        else:
            self.cal_end_px = marker
            self.log(f'Captured END marker pixel={marker}')
            if self.cal_home_px is None or math.dist(self.cal_end_px,self.cal_home_px)<30:
                self.log('Camera calibration invalid: markers overlap or HOME missing; do not use AUTOX.')
                return
            self.camera_cal = {'home_px':self.cal_home_px[0], 'home_py':self.cal_home_px[1],
                               'end_px':self.cal_end_px[0], 'end_py':self.cal_end_px[1],
                               'stroke_mm':TRAVEL_MM,'marker_id':int(self.marker_id.get()),
                               'camera_index':int(self.camera_index.get())}
            CAL_FILE.write_text(json.dumps(self.camera_cal,indent=2))
            self.log('Saved camera_calibration.json')
            self.command('CALNEXT')

    def pixel_to_mm(self, pixel_x, pixel_y=None):
        if not self.camera_cal: raise ValueError('Camera calibration required')
        d=self.camera_cal
        # Project the observed marker position onto the HOME->END vector.
        # Works for non-horizontal rail images (approx. planar, fixed camera).
        dx=d['end_px']-d['home_px']
        dy=d.get('end_py',0)-d.get('home_py',0)
        if pixel_y is None:
            pixel_y=d.get('home_py',0)
        denom=dx*dx+dy*dy
        if denom<1: raise ValueError('Invalid calibration')
        return ((pixel_x-d['home_px'])*dx+(pixel_y-d.get('home_py',0))*dy)/denom*TRAVEL_MM

    def process_event(self,name):
        self.event_label.set('Last event: '+name[:42])
        if name == 'CAL_AT_HOME':
            self.root.after(300, lambda:self.capture_cal_marker('HOME'))
        elif name == 'CAL_AT_END':
            self.root.after(300, lambda:self.capture_cal_marker('END'))
        elif name == 'SERIAL_DISCONNECTED':
            self.connected=False;self.mode_label.set('DISCONNECTED')

    def process_line(self,line):
        if line.startswith('T,'):
            p=line.split(',')
            if len(p) != 15: return
            try:
                mode,pe,se,hl,el=p[1],int(p[2]),int(p[3]),int(p[4]),int(p[5])
                mm,tgt,speed=float(p[6]),float(p[7]),float(p[8])
                f,valid,cal,stroke=int(p[9]),int(p[10]),int(p[11]),int(p[12])
                elapsed,issued=float(p[13]),int(p[14])
                self.physical.set(bool(pe));self.software.set(bool(se))
                self.home_limit.set(bool(hl));self.end_limit.set(bool(el))
                self.valid=bool(valid);self.calibrated=bool(cal)
                self.running=mode not in ('IDLE','CAL_HOLD_HOME','CAL_HOLD_END')
                self.mode_label.set(mode)
                self.position_label.set(f'Position: {mm:.1f} mm  /  Target: {tgt:.1f} mm' if valid and cal else 'Position: NOT REFERENCED')
                self.speed_label.set(f'Speed: {speed:,.0f} steps/s  ({speed*60/400:.0f} motor RPM)')
                self.steps_label.set(f'Stroke: {stroke:,} steps  |  Move: {issued:,} steps')
                self.last_label.set(f'Last travel: {elapsed:.3f} s')
            except (ValueError,IndexError):
                pass
        elif line.startswith('EVENT,'):
            name=line[6:]; self.process_event(name);self.log(line)
        elif line.startswith('CAL,') or line.startswith('RESULT,'):
            self.log(line)

    def tick(self):
        for _ in range(80):
            try:line=self.lines.get_nowait()
            except queue.Empty:break
            self.process_line(line)
        for _ in range(20):
            try:req=self.auto_requests.get_nowait()
            except queue.Empty:break
            if req['cmd']=='ESTOP': self.command('ESTOP')
            elif req['cmd']=='STOP': self.command('STOP')
            elif self.auto_mode.get()!='AUTO': self.log('Rejected autonomous command (switch to AUTO)')
            elif req['cmd']=='GOTO':
                mm=req['value']
                if 0<=mm<=432: self.command(f'GOTO {mm:.2f}')
            elif req['cmd'] in ('AUTOX','AUTOXY'):
                try:
                    mm=self.pixel_to_mm(req['value'],req.get('pixel_y'))
                    if 0<=mm<=432: self.command(f'GOTO {mm:.2f}')
                    else:self.log('AUTOX outside calibrated stroke')
                except ValueError as ex:self.log(str(ex))
        if self.jog_held and time.monotonic()-self.last_jog_send>.12:
            self.command('JOG '+self.jog_held)
            self.last_jog_send=time.monotonic()
        for key,state in [('home',self.home_limit.get()),('end',self.end_limit.get()),('estop',self.physical.get()),('soft',self.software.get())]:
            c,ci=self.indicators[key]
            c.itemconfigure(ci,fill='#9ca3af' if not self.connected else ('#dc2626' if state else '#16a34a'))
        frame,marker,seen=self.camera.snapshot()
        if frame is not None:
            height,width=frame.shape[:2]
            thumb=cv2.resize(frame,(350,200))
            pic=Image.fromarray(cv2.cvtColor(thumb,cv2.COLOR_BGR2RGB))
            self.photo=ImageTk.PhotoImage(pic)
            self.preview.configure(image=self.photo,text='')
            if marker and time.monotonic()-seen<.4:
                try:self.camera_status.set(f'Marker x={marker[0]:.1f}px; visual position {self.pixel_to_mm(*marker):.1f} mm')
                except ValueError:self.camera_status.set(f'Marker x={marker[0]:.1f}px (camera calibration pending)')
            else:self.camera_status.set('Marker not visible')
        self.root.after(50,self.tick)

    def log(self, s):
        self.logbox.insert('end', time.strftime('%H:%M:%S')+'  '+s+'\n')
        self.logbox.see('end')
        if float(self.logbox.index('end-1c').split('.')[0])>350:
            self.logbox.delete('1.0','80.0')

    def close(self):
        try:self.serial.send('STOP')
        except Exception:pass
        self.serial.close()
        self.camera.stop()
        self.server.shutdown();self.server.server_close()
        self.root.destroy()


if __name__=='__main__':
    root=tk.Tk()
    App(root)
    root.mainloop()
