"""Mirror the physical catcher in the MuJoCo viewer.

Reads the board's position (pulse count from TELEM) and moves the simulated carriage to match.
Read-only: it only sends TELEM queries and never sends a motion command, so you can run it
while another tool or the GUI is moving the hardware.

Usage:  python mirror_physical.py COM5
Close the MuJoCo window to stop.
"""
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import serial

XML = Path(r'C:\Users\vinith\AppData\Local\Temp\claude\5b5ff089e563181a\380e24e4-3b72-4177-84a5-ec9c5a570d75\scratchpad\catch_x_user.xml')
TRAVEL_MM = 432.0
FULL_PULSES = 34192
POLL_S = 0.05
PRINT_EVERY_S = 1.0


def read_tel(ser):
    """Ask for telemetry and return the parsed TEL fields, or None if no reply."""
    ser.write(b'TELEM\n')
    deadline = time.time() + 0.2
    while time.time() < deadline:
        line = ser.readline().decode('utf-8', 'replace').strip()
        if line.startswith('TEL,'):
            a = line.split(',', 11)
            if len(a) == 12:
                return dict(running=int(a[1]), estop=int(a[4]), swstop=int(a[5]),
                            known=int(a[6]), pulses=int(a[9]), reason=a[11])
    return None


def main():
    if len(sys.argv) < 2:
        sys.exit('Usage: python mirror_physical.py COMx')
    ser = serial.Serial(sys.argv[1], 115200, timeout=0.05, write_timeout=0.2)
    time.sleep(1.8)                      # board may reboot when the port opens
    ser.reset_input_buffer()

    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    qa = int(model.jnt_qposadr[model.joint('carriage_slide').id])
    ball = int(model.jnt_qposadr[model.joint('ball_free').id])
    data.qpos[ball:ball + 3] = [0.0, 0.0, -10.0]     # hide the XML ball
    data.qpos[qa] = 0.0
    mujoco.mj_forward(model, data)

    last_print = 0.0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            tel = read_tel(ser)
            if tel is not None and tel['known']:
                mm = tel['pulses'] * TRAVEL_MM / FULL_PULSES
                data.qpos[qa] = mm / 1000.0          # sim joint in metres, 0 = HOME
                mujoco.mj_forward(model, data)
            now = time.time()
            if now - last_print >= PRINT_EVERY_S:
                last_print = now
                if tel is None:
                    print('No reply from board')
                elif not tel['known']:
                    print('Position unknown: run HOME then END on the board')
                else:
                    mm = tel['pulses'] * TRAVEL_MM / FULL_PULSES
                    state = 'E-STOP' if tel['estop'] or tel['swstop'] else ('RUN' if tel['running'] else 'IDLE')
                    print(f'physical {mm:7.1f} mm  |  {state}  |  {tel["reason"]}')
            viewer.sync()
            time.sleep(POLL_S)
    ser.close()


if __name__ == '__main__':
    main()
