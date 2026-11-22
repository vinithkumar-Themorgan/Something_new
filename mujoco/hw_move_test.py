"""Hardware move test for INTERCEPT-X. Moves the physical carriage through test positions.

Usage:  python hw_move_test.py COM5
Keep the hardware E-stop in reach. The script asks for confirmation before any motion,
and sends STOP on Ctrl+C.

Steps: check E-stop and limits -> HOME -> END (sets the position reference) ->
go to each target in TARGETS_MM -> log the result to a CSV file next to this script.
"""
import csv
import sys
import time
from pathlib import Path

import serial

TARGETS_MM = [50.0, 150.0, 216.0, 300.0, 400.0]
TRAVEL_MM = 432.0
FULL_PULSES = 34192
SETTLE_S = 0.5               # pause after each move before reading the final position
START_TIMEOUT_S = 2.0        # motor must start within this time of a command
MOVE_TIMEOUT_S = 20.0        # max time for one move
LOG = Path(__file__).resolve().parent / 'hw_move_log.csv'


def parse_tel(line):
    a = line.split(',', 11)
    if len(a) < 12:
        return None
    return dict(running=int(a[1]), rev=int(a[2]), fwd=int(a[3]), estop=int(a[4]),
                swstop=int(a[5]), known=int(a[6]), freq=int(a[7]), current=float(a[8]),
                pulses=int(a[9]), remaining=int(a[10]), reason=a[11])


def read_tel(ser, timeout=1.0):
    ser.write(b'TELEM\n')
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = ser.readline().decode('utf-8', 'replace').strip()
        if line.startswith('TEL,'):
            return parse_tel(line)
    return None


def wait_for_move(ser, label):
    """Wait until the motor has run and stopped. Returns (firmware_elapsed_s, final telemetry)."""
    elapsed = None
    start = time.time()
    seen_running = False
    while True:
        line = ser.readline().decode('utf-8', 'replace').strip()
        if line.startswith('Elapsed:'):
            try:
                elapsed = float(line.split()[1])
            except (IndexError, ValueError):
                pass
        if line.startswith('STOP:') or line.startswith('MOVE '):
            print('   ', line)
        if time.time() - start > MOVE_TIMEOUT_S:
            raise RuntimeError(f'{label}: timed out')
        tel = read_tel(ser, timeout=0.3)
        if tel is None:
            continue
        if tel['running']:
            seen_running = True
            continue
        if seen_running or time.time() - start > START_TIMEOUT_S:
            if not seen_running:
                raise RuntimeError(f'{label}: motor did not start (reason: {tel["reason"]})')
            time.sleep(SETTLE_S)
            return elapsed, read_tel(ser) or tel


def send_and_wait(ser, cmd, label):
    ser.write((cmd + '\n').encode('ascii'))
    return wait_for_move(ser, label)


def main():
    if len(sys.argv) < 2:
        sys.exit('Usage: python hw_move_test.py COMx')
    port = sys.argv[1]
    ser = serial.Serial(port, 115200, timeout=0.05, write_timeout=0.5)
    time.sleep(1.8)                         # board may reboot when the port opens
    ser.reset_input_buffer()

    tel = read_tel(ser)
    if tel is None:
        sys.exit('No telemetry from the board. Check the port and firmware.')
    print('Start state:', tel)
    if tel['estop'] or tel['swstop']:
        sys.exit('E-stop or software stop is active. Release it and send SCLEAR first.')

    print('\nThe carriage will HOME, then END, then move to:', TARGETS_MM, 'mm')
    print('Check: the rail is clear, the hardware E-stop is in reach, someone is watching.')
    if input("Type GO to start the motion test: ").strip() != 'GO':
        print('Cancelled. No motion.')
        ser.close()
        return

    rows = []
    try:
        if not tel['known']:
            print('\nHOME ...')
            elapsed, tel = send_and_wait(ser, 'HOME', 'HOME')
            print(f'   HOME done, firmware time {elapsed} s')
            print('END ...')
            elapsed, tel = send_and_wait(ser, 'END', 'END')
            print(f'   END done, firmware time {elapsed} s, known={tel["known"]}')
        if not tel['known']:
            sys.exit('Position still unknown after HOME/END. Check the limit switches.')

        for target in TARGETS_MM:
            target_pulses = round(target * FULL_PULSES / TRAVEL_MM)
            delta = tel['pulses'] - target_pulses        # M +n moves forward (pulses decrease)
            if abs(delta) <= 2:
                print(f'Already at {target:.0f} mm')
                continue
            print(f'\nGo to {target:.0f} mm (target {target_pulses} pulses, M {delta}) ...')
            t0 = time.time()
            elapsed, tel = send_and_wait(ser, f'M {delta}', f'{target:.0f} mm')
            wall = time.time() - t0
            final_mm = tel['pulses'] * TRAVEL_MM / FULL_PULSES
            error = final_mm - target
            print(f'   final {final_mm:.2f} mm, error {error:+.2f} mm, '
                  f'firmware time {elapsed} s, wall time {wall:.2f} s')
            rows.append(dict(target_mm=target, final_mm=round(final_mm, 2), error_mm=round(error, 2),
                             firmware_s=elapsed, wall_s=round(wall, 2), reason=tel['reason']))
    except KeyboardInterrupt:
        print('\nInterrupted. Sending STOP.')
    except RuntimeError as exc:
        print('\nERROR:', exc)
    finally:
        ser.write(b'STOP\n')
        ser.close()

    if rows:
        with LOG.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'\nLog written to {LOG}')
        worst = max(abs(r['error_mm']) for r in rows)
        print(f'Worst position error: {worst:.2f} mm over {len(rows)} moves')


if __name__ == '__main__':
    main()
