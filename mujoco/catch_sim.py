"""Headless catch check for INTERCEPT-X.

A virtual ping-pong ball is launched in MuJoCo. At launch the planner predicts where the ball
will cross the bucket rim, starts the carriage toward that point, and the check asks whether the
bucket is under the ball when it crosses the rim.

Carriage motion copies the firmware ramp (ACCEL 20000, START 500, VMAX 15000 pulses/s).
Geometry comes from catch_x.xml and meshes/catcher.stl, with the opening from your measurements.
"""
from pathlib import Path
import csv, sys
import numpy as np
import mujoco

XML = Path(r'C:\Users\vinith\AppData\Local\Temp\claude\5b5ff089e563181a\380e24e4-3b72-4177-84a5-ec9c5a570d75\scratchpad\catch_x_user.xml')  # your tested XML (0..0.432 m travel)
OUT_CSV = Path(r'C:\Users\vinith\AppData\Local\Temp\claude\5b5ff089e563181a\380e24e4-3b72-4177-84a5-ec9c5a570d75\scratchpad\catch_results.csv')

# ---- Hardware (firmware and your measurements) ----
RAIL_MM = 432.0                  # physical end-to-end travel (HOME = 0 mm, D3 side)
FULL_PULSES = 34192
PULSES_PER_MM = FULL_PULSES / RAIL_MM
ACCEL = 20000.0                  # pulses/s^2, firmware
START = 500.0                    # pulses/s, firmware START_FREQ
VMAX = 15000.0                   # pulses/s, your limit
CMD_LATENCY = 0.05               # s from launch to motor command (serial + firmware), adjust after measuring

# ---- Geometry (XML + catcher.stl) ----
BUCKET_X_AT_HOME = -0.219        # bucket centre x relative to carriage origin (XML catcher_1 pos)
BUCKET_Y = 0.0                   # bucket centre y (XML -0.00045 m)
BUCKET_TOP_Z = 0.755 + 0.1901    # carriage z + catcher.stl top: rim where the ball enters
OPEN_X_M, OPEN_Y_M = 0.170, 0.173  # measured opening
BALL_R = 0.020                   # ping-pong ball, 40 mm diameter (XML ball_geom size)
WIN_X = OPEN_X_M / 2 - BALL_R    # ball centre must be within this of bucket centre
WIN_Y = OPEN_Y_M / 2 - BALL_R

LAUNCH_POS = np.array([0.0, 0.6, 1.1])    # launcher on the camera side (+y), where the markers face


def firmware_profile(steps, vmax=VMAX):
    """Time and pulse count for a move of `steps` pulses, same update as firmware updateMotionSpeed."""
    t = 0.0
    cur = START
    remaining = steps
    done = 0
    times = [0.0]
    pos = [0.0]
    while remaining > 0:
        t += 1.0 / cur                       # time until the next pulse
        done += 1
        remaining -= 1
        times.append(t)
        pos.append(done)
        stop = (cur * cur - START * START) / (2.0 * ACCEL) if cur > START else 0.0
        if remaining <= stop:
            cur = max(START, cur - ACCEL / cur)
        elif cur < vmax:
            cur = min(vmax, cur + ACCEL / cur)
    return np.array(times), np.array(pos, dtype=float)


def simulate_ball(model, data, ball_jid, p0, v0, t_max=2.0):
    """Free-flight ball in MuJoCo (gravity from the XML). Returns list of (t, pos, vel)."""
    qadr = int(model.jnt_qposadr[ball_jid])
    dadr = int(model.jnt_dofadr[ball_jid])
    data.qpos[qadr:qadr + 3] = p0
    data.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[dadr:dadr + 3] = v0
    data.qvel[dadr + 3:dadr + 6] = 0.0
    mujoco.mj_forward(model, data)
    traj = []
    t = 0.0
    while t < t_max:
        traj.append((t, data.qpos[qadr:qadr + 3].copy(), data.qvel[dadr:dadr + 3].copy()))
        mujoco.mj_step(model, data)
        t += model.opt.timestep
    return traj


def rim_crossing(traj):
    """First downward crossing of the bucket rim plane: (time, x, y) or None."""
    for (t0, p0, _), (t1, p1, v1) in zip(traj, traj[1:]):
        if p0[2] > BUCKET_TOP_Z >= p1[2] and v1[2] < 0:
            a = (BUCKET_TOP_Z - p0[2]) / (p1[2] - p0[2])
            return t0 + a * (t1 - t0), p0[0] + a * (p1[0] - p0[0]), p0[1] + a * (p1[1] - p0[1])
    return None


def run_shot(model, data, ball_jid, T, landing_x, landing_y=0.0, start_mm=216.0,
             latency=CMD_LATENCY, rng=None):
    """Launch a ball that reaches (landing_x, landing_y, rim) at time T; plan and check the catch."""
    rng = rng or np.random.default_rng()
    x0 = rng.uniform(-0.10, 0.10)
    p0 = np.array([x0, LAUNCH_POS[1], LAUNCH_POS[2]])
    g = 9.81
    v0 = np.array([
        (landing_x - x0) / T,
        (landing_y - p0[1]) / T,
        (BUCKET_TOP_Z - p0[2] + 0.5 * g * T * T) / T,   # ballistic solution for the rim at time T
    ])
    traj = simulate_ball(model, data, ball_jid, p0, v0)
    cross = rim_crossing(traj)
    result = dict(T=T, landing_x_target=landing_x, launch_vz=v0[2])
    if cross is None:
        result.update(caught=False, reason='no rim crossing')
        return result
    tc, xc, yc = cross
    target_mm = (landing_x - BUCKET_X_AT_HOME) * 1000.0
    result.update(t_cross=tc, landing_x=xc, landing_y=yc, target_mm=target_mm)
    if not 0.0 <= target_mm <= RAIL_MM:
        result.update(caught=False, reason='landing point off the rail')
        return result

    steps = round(abs(target_mm - start_mm) * PULSES_PER_MM)
    sign = 1.0 if target_mm >= start_mm else -1.0
    times, pos = firmware_profile(steps)
    move_time = times[-1]
    t_into_move = tc - latency
    moved = np.interp(t_into_move, times, pos) if t_into_move > 0 else 0.0
    carriage_mm = start_mm + sign * moved / PULSES_PER_MM
    bucket_x = BUCKET_X_AT_HOME + carriage_mm / 1000.0
    err_mm = (bucket_x - xc) * 1000.0
    in_window = abs(bucket_x - xc) <= WIN_X and abs(BUCKET_Y - yc) <= WIN_Y
    still_moving = t_into_move < move_time
    if in_window:
        reason = 'caught'
    elif still_moving:
        reason = 'too slow (still moving at rim crossing)'
    else:
        reason = 'positioning error'
    result.update(move_time=move_time, arrival_time=move_time + latency, bucket_x=bucket_x,
                  err_mm=err_mm, caught=in_window, reason=reason)
    return result


def main():
    if not XML.exists():
        sys.exit(f'Missing {XML}')
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    ball_jid = model.joint('ball_free').id
    rng = np.random.default_rng(1)

    # 1. Single demo shot with a timeline
    demo = run_shot(model, data, ball_jid, T=1.2, landing_x=0.10, rng=rng)
    print('Demo shot: T=1.2 s, landing x=+100 mm')
    for k in ('launch_vz', 'target_mm', 't_cross', 'landing_x', 'move_time', 'arrival_time', 'bucket_x', 'err_mm', 'caught', 'reason'):
        if k in demo:
            v = demo[k]
            print(f'  {k:14s} {v:.4f}' if isinstance(v, float) else f'  {k:14s} {v}')

    # 2. Batch: random flight times and landing points across the rail
    N = 400
    rows = []
    x_lo = BUCKET_X_AT_HOME                     # bucket at HOME
    x_hi = BUCKET_X_AT_HOME + RAIL_MM / 1000.0  # bucket at END
    for _ in range(N):
        T = rng.uniform(0.8, 1.6)
        xL = rng.uniform(x_lo, x_hi)
        rows.append(run_shot(model, data, ball_jid, T, xL, rng=rng))

    caught = [r for r in rows if r.get('caught')]
    print(f'\nBatch: {len(caught)}/{N} caught ({100*len(caught)/N:.1f}%), '
          f'window x = +/-{WIN_X*1000:.0f} mm, y = +/-{WIN_Y*1000:.0f} mm, latency {CMD_LATENCY*1000:.0f} ms')
    print('Flight time bins (caught / total):')
    for lo in np.arange(0.8, 1.6, 0.2):
        hi = lo + 0.2
        sub = [r for r in rows if lo <= r['T'] < hi]
        ok = sum(1 for r in sub if r.get('caught'))
        if sub:
            print(f'  {lo:.1f}-{hi:.1f} s: {ok}/{len(sub)}')
    reasons = {}
    for r in rows:
        reasons[r.get('reason', '?')] = reasons.get(r.get('reason', '?'), 0) + 1
    print('Outcomes:', reasons)

    keys = sorted({k for r in rows for k in r})
    with OUT_CSV.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f'Results written to {OUT_CSV}')


if __name__ == '__main__':
    main()
