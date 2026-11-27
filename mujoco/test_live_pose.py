"""Test the live pose without the camera.

1. Runs the live pose on the saved captures (fixed camera): the camera position should match the fit.
2. Simulates the camera being moved (rotated and shifted), projects markers 10 and 23 with
   distortion, and checks that the live pose recovers the new camera position.
"""
import json
from pathlib import Path

import cv2
import numpy as np

from live_pose import LivePose

HERE = Path(__file__).resolve().parent


def main():
    lp = LivePose()
    cap = json.loads((HERE / 'capture_markers.json').read_text())
    print('1. Saved captures (camera fixed):')
    for e in cap['entries']:
        det = {int(k): np.array(v) for k, v in e['markers'].items()}
        ok = lp.update(det)
        print(f"   q={e['q_mm']:6.1f} mm  updated={ok}  rms={lp.rms:.3f}px  "
              f"camera={np.round(lp.camera_position(), 3)}")

    print('2. Simulated camera move (rotated 3 deg, shifted 5 cm sideways and 3 cm closer):')
    base = LivePose()
    base.update({int(k): np.array(v) for k, v in cap['entries'][0]['markers'].items()})
    true_pos = base.camera_position()
    # New camera pose: rotate about the vertical axis by 3 degrees and move the camera
    rot_z, _ = cv2.Rodrigues(np.array([0.0, 0.0, np.deg2rad(3.0)]))
    R0, _ = cv2.Rodrigues(base.rvec)
    R_new = rot_z @ R0
    cam_new = true_pos + np.array([0.05, 0.0, 0.0]) + np.array([0.0, -0.03, 0.0])
    rvec_new, _ = cv2.Rodrigues(R_new)
    tvec_new = -R_new @ cam_new
    # Project the true markers 10 and 23 into the moved camera, with distortion
    detected = {}
    for mid in (10, 23):
        world = cv2.Rodrigues  # placeholder to keep names clear
        pts = base.obj[0:4] if mid == 10 else base.obj[4:8]
        img, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), rvec_new, tvec_new, base.K, base.dist)
        detected[mid] = img.reshape(4, 2) + np.random.default_rng(mid).normal(0, 0.2, (4, 2))
    sim = LivePose()
    sim.update(detected)
    err = np.linalg.norm(sim.camera_position() - cam_new) * 1000
    print(f'   true camera {np.round(cam_new, 3)}  recovered {np.round(sim.camera_position(), 3)}  '
          f'error {err:.1f} mm  rms {sim.rms:.3f}px')


if __name__ == '__main__':
    main()
