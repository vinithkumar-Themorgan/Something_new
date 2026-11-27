"""Match the MuJoCo camera to the physical camera, using markers 10 and 23 as the shared reference.

1. Reads the physical camera pose (camera_pose_10_23.json) and intrinsics.
2. Sets the MuJoCo free camera to the same position, direction and field of view.
3. Shifts the render so its optical centre matches the physical principal point.
4. Renders the rig with the carriage at 432 mm, detects the markers in the render, and compares them
   with the real detections from capture_markers.json at the same carriage position.

MuJoCo cameras have no roll (the up direction is fixed to world Z), so the roll of the physical
camera is printed as a check.
"""
import json
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
XML = Path('C:/Users/vinith/AppData/Local/Temp/claude/5b5ff089e563181a/380e24e4-3b72-4177-84a5-ec9c5a570d75/scratchpad/catch_x_user.xml')
POSE = HERE / 'camera_pose_10_23.json'
INTR = HERE / 'intrinsics_run2.json'
CAPTURE = HERE / 'capture_markers.json'
W, H = 640, 480
CHECK_MM = 432.0


def mujoco_camera_from_pose(pose, K):
    R, _ = cv2.Rodrigues(np.array(pose['rvec'], float).reshape(3, 1))
    t = np.array(pose['tvec'], float).reshape(3)
    pos = -R.T @ t                       # camera position in world
    fwd = R.T @ np.array([0.0, 0.0, 1.0])  # camera viewing direction in world
    right = R.T @ np.array([1.0, 0.0, 0.0])
    up = -R.T @ np.array([0.0, 1.0, 0.0])
    az = math.degrees(math.atan2(fwd[1], fwd[0]))
    el = math.degrees(math.asin(np.clip(fwd[2], -1, 1)))
    fovy = math.degrees(2 * math.atan((H / 2) / K[1, 1]))
    roll = math.degrees(math.atan2(right[2], math.hypot(right[0], right[1])))
    dx = K[0, 2] - W / 2
    dy = K[1, 2] - H / 2
    return dict(pos=pos, fwd=fwd, az=az, el=el, fovy=fovy, roll=roll, dx=dx, dy=dy)


def main():
    pose = json.loads(POSE.read_text())
    intr = json.loads(INTR.read_text())
    K = np.array(intr['K'], float)
    dist = np.array(intr['dist'], float)
    cam = mujoco_camera_from_pose(pose, K)
    print(f"camera position {np.round(cam['pos'], 3)} m, azimuth {cam['az']:.1f} deg, "
          f"elevation {cam['el']:.1f} deg, fovy {cam['fovy']:.1f} deg")
    print(f"roll of physical camera: {cam['roll']:.2f} deg (MuJoCo cannot represent roll)")
    print(f"principal point offset: dx {cam['dx']:+.1f} px, dy {cam['dy']:+.1f} px (applied as an image shift)")

    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    model.vis.global_.fovy = cam['fovy']
    qa = int(model.jnt_qposadr[model.joint('carriage_slide').id])
    ball = int(model.jnt_qposadr[model.joint('ball_free').id])
    data.qpos[ball:ball + 3] = [0.0, 0.0, -10.0]
    data.qpos[qa] = CHECK_MM / 1000.0
    mujoco.mj_forward(model, data)

    free = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, free)
    free.distance = 1.0
    free.lookat[:] = cam['pos'] + free.distance * cam['fwd']
    free.azimuth = cam['az']
    free.elevation = cam['el']

    rend = mujoco.Renderer(model, height=H, width=W)
    rend.update_scene(data, camera=free)
    rgb = rend.render()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    shift = np.float32([[1, 0, cam['dx']], [0, 1, cam['dy']]])
    matched = cv2.warpAffine(bgr, shift, (W, H))
    cv2.imwrite(str(HERE / 'virtual_matched_432mm.png'), matched)

    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                                       cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(matched, cv2.COLOR_BGR2GRAY))
    virtual = {}
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            virtual[int(i)] = c.reshape(4, 2)

    cap = json.loads(CAPTURE.read_text())
    entry = min(cap['entries'], key=lambda e: abs(e['q_mm'] - CHECK_MM))
    print(f"comparing with the real capture at {entry['q_mm']:.0f} mm")
    print('marker  mean corner error (px)')
    for mid in (10, 23, 42):
        if mid not in virtual or str(mid) not in entry['markers']:
            print(f'  {mid:3d}  not found in virtual or real image')
            continue
        real = np.array(entry['markers'][str(mid)], float).reshape(-1, 1, 2)
        real_u = cv2.undistortPoints(real, K, dist, P=K).reshape(4, 2)
        err = np.linalg.norm(virtual[mid] - real_u, axis=1).mean()
        print(f'  {mid:3d}  {err:.2f}')
    print('saved virtual_matched_432mm.png (the render with the matched camera)')


if __name__ == '__main__':
    main()
