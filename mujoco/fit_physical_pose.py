"""Fit the physical camera's pose from captured marker corners, using the chessboard intrinsics.

Usage:  python fit_physical_pose.py [intrinsics.json] [capture_markers.json]
Defaults: C:\\robotics\\mujoko_projects\\camera_intrinsics.json and capture_markers.json here.
Output: camera_pose.json next to this script.

The focal length and distortion are fixed (from the chessboard). The fit finds the camera's
position and angle, the principal point and the two pattern sizes.
"""
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from camera_fit import fit_camera, rvec_tvec_from_pose

HERE = Path(__file__).resolve().parent
DEFAULT_INTRINSICS = Path(r'C:\robotics\mujoko_projects\camera_intrinsics.json')
DEFAULT_CAPTURE = HERE / 'capture_markers.json'
OUT = HERE / 'camera_pose.json'


def load_obs(capture, K, dist):
    """Undistort captured corners to ideal pixel coordinates; carriage position in metres."""
    obs = []
    for e in capture['entries']:
        q = e['q_mm'] / 1000.0
        for mid, corners in e['markers'].items():
            pts = np.array(corners, dtype=np.float64).reshape(-1, 1, 2)
            und = cv2.undistortPoints(pts, K, dist, P=K).reshape(4, 2)
            obs.append((int(mid), q, und))
    return obs


def start_guesses(W, H, f, cx, cy):
    """Coarse grid of camera positions in front of the rail, looking down at the markers."""
    look = np.array([0.0, 0.0, 0.85])
    for az_deg in (-100, -90, -80):
        for el_deg in (-30, -20, -10):
            for dist in (1.0, 1.5, 2.0):
                az, el = math.radians(az_deg), math.radians(el_deg)
                fwd_dir = np.array([math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)])
                pos = look - dist * fwd_dir
                fwd = look - pos
                fwd /= np.linalg.norm(fwd)
                rvec, tvec = rvec_tvec_from_pose(pos, fwd, np.array([0.0, 0.0, 1.0]))
                yield np.concatenate([[f, cx, cy], rvec, tvec, [0.03, 0.04]])


def main():
    intr_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INTRINSICS
    cap_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CAPTURE
    intr = json.loads(intr_path.read_text())
    capture = json.loads(cap_path.read_text())
    K = np.array(intr['K'], dtype=np.float64)
    dist = np.array(intr['dist'], dtype=np.float64)
    W, H = intr['image_size']
    f = float((K[0, 0] + K[1, 1]) / 2)
    cx, cy = float(K[0, 2]), float(K[1, 2])

    obs = load_obs(capture, K, dist)
    print(f'{len(obs)} marker observations from {len(capture["entries"])} positions; '
          f'focal fixed at {f:.1f} px')

    best = None
    for init in start_guesses(W, H, f, cx, cy):
        params, rms, options = fit_camera(obs, W, H, init, fixed_f=f, fixed_pp=(cx, cy))
        if best is None or rms < best[1]:
            best = (params, rms, options)
    params, rms, options = best

    R, _ = cv2.Rodrigues(params[3:6])
    cam_pos = -R.T @ params[6:9]
    print(f'Reprojection RMS: {rms:.3f} px  (good if below about 1 px)')
    print(f'Camera position (MuJoCo world, m): {np.round(cam_pos, 3)}')
    print(f'Principal point: ({params[1]:.1f}, {params[2]:.1f})')
    print(f'Pattern half-size: 10/23 {params[9]*1000:.1f} mm, 42 {params[10]*1000:.1f} mm')
    print(f'Marker orientation options: {options}')
    d = np.linalg.norm(np.array([-0.2637, 0.1063, 0.7925]) - np.array([0.2637, 0.1063, 0.7925]))
    print(f'Marker 10-23 spacing in model: {d:.4f} m (your measurement: 0.5275 m)')
    if rms > 1.0:
        print('RMS is high: check the marker capture and the chessboard calibration.')

    OUT.write_text(json.dumps(dict(
        created=datetime.now().isoformat(timespec='seconds'),
        K=[[params[0], 0, params[1]], [0, params[0], params[2]], [0, 0, 1]],
        rvec=params[3:6].tolist(), tvec=params[6:9].tolist(),
        camera_position_m=cam_pos.tolist(), rms_px=float(rms),
        pattern_half_m={'10_23': float(params[9]), '42': float(params[10])},
        orientation_options={str(k): list(v) for k, v in options.items()},
        image_size=[W, H], distortion=dist.tolist(),
    ), indent=2))
    print(f'Saved {OUT}')


if __name__ == '__main__':
    main()
