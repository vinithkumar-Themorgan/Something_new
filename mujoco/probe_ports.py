"""Find which COM port is the INTERCEPT-X board. Read-only: sends only TELEM status requests.

Usage:  python probe_ports.py
"""
import time
import serial
from serial.tools import list_ports

for p in list_ports.comports():
    name = p.device
    try:
        ser = serial.Serial(name, 115200, timeout=0.3, write_timeout=0.5)
    except serial.SerialException as exc:
        print(f'{name:6s} cannot open: {exc}')
        continue
    try:
        time.sleep(1.8)                      # board may reboot when the port opens
        ser.reset_input_buffer()
        ser.write(b'TELEM\n')
        reply = None
        deadline = time.time() + 1.0
        while time.time() < deadline:
            line = ser.readline().decode('utf-8', 'replace').strip()
            if line.startswith('TEL,'):
                reply = line
                break
        print(f'{name:6s} {p.description}: ' + (f'BOARD -> {reply}' if reply else 'no TEL reply'))
    except serial.SerialException as exc:
        print(f'{name:6s} {p.description}: error {exc}')
    finally:
        ser.close()
