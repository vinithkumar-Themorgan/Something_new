"""Calibrate a camera from ArUco marker corners whose world positions are known.

Known world geometry (from catch_x.xml / meshes, checked against your 0.5275 m):
  * marker 10 and 23: fixed base plates, facing +y
  * marker 42: on the carriage, moves along x with the carriage position q

Unknowns fitted by least squares:
  f (focal, px), cx, cy, camera rotation (rvec), camera position (tvec),
  pattern half-size for 10/23 and for 42 (the printed pattern is smaller than the white plate).

Marker corner order is unknown from the image alone, so each marker's orientation is chosen
by the best fit at the start (8 options: 4 rotations x 2 directions).

Synthetic test: run this file directly. It renders the virtual camera at several carriage
positions, detects the markers, and checks the fit recovers the known camera.
"""
from pathlib import Path
import itertools
import json
import math
import numpy as np
import cv2
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
LAYOUT = Path(r'C:\Users\vinith\Downloads\CATCH-X_Corrected_Marker_and_Camera\marker_layout.json')
PLATE_HALF_10_23 = 0.0375          # white plate 75 mm (pattern unknown, fitted)
PLATE_HALF_42 = 0.050              # white plate 100 mm
CARRIAGE_X0 = -0.219               # catcher centre x at HOME (XML)
CARRIAGE_Y = 0.0896                # catcher marker y (XML)
CARRIAGE_Z = 0.8945                # catcher marker z (XML)
BASE_Y = 0.1063
BASE_Z = 0.7925
BASE_X = 0.2637                    # marker 10 at -BASE_X, marker 23 at +BASE_X

ORDER = [(-1, +1), (+1, +1), (+1, -1), (-1, -1)]   # TL, TR, BR, BL in the marker's own frame


def marker_centre(mid, q):
    if mid == 10:
        return np.array([-BASE_X, BASE_Y, BASE_Z])
    if mid == 23:
        return np.array([+BASE_X, BASE_Y, BASE_Z])
    if mid == 42:
        return np.array([CARRIAGE_X0 + q, CARRIAGE_Y, CARRIAGE_Z])
    raise ValueError(mid)


def option_corners(centre, half, option):
    """World corners for one orientation option. Plates face +y, so their axes are x and z."""
    rot, flip = option
    order = ORDER if not flip else ORDER[::-1]
    order = order[rot:] + order[:rot]
    return np.array([centre + half * (a * np.array([1.0, 0, 0]) + b * np.array([0, 0, 1.0]))
                     for a, b in order])


def project(params, pts3d, W, H):
    f, cx, cy = params[0], params[1], params[2]
    rvec, tvec = params[3:6], params[6:9]
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    img, _ = cv2.projectPoints(pts3d.reshape(-1, 1, 3), rvec, tvec, K, None)
    return img.reshape(-1, 2)


def residuals(params, obs, options, W, H):
    res = []
    h10, h42 = params[9], params[10]
    for mid, q, corners in obs:
        half = h42 if mid == 42 else h10
        world = option_corners(marker_centre(mid, q), half, options[mid])
        res.append((project(params, world, W, H) - corners).ravel())
    return np.concatenate(res)


def fit_camera(obs, W, H, init_params, iters=2000, fixed_f=None, fixed_pp=None):
    """obs: list of (marker_id, carriage_q_m, corners[4,2]) in pixels. Returns (params, rms_px, options).

    fixed_f: if given (e.g. from a chessboard calibration), the focal length is not fitted.
    fixed_pp: if given as (cx, cy), the principal point is not fitted.
    """
    # 1. orientation per marker id from the initial pose
    options = {}
    for mid in (10, 23, 42):
        if not any(o[0] == mid for o in obs):
            continue                      # marker not used in this fit
        best, best_err = None, np.inf
        for rot, flip in itertools.product(range(4), (False, True)):
            opt = (rot, flip)
            trial = dict(options); trial[mid] = opt
            r = residuals(init_params, [o for o in obs if o[0] == mid], trial, W, H)
            err = float(np.sum(r ** 2))
            if err < best_err:
                best, best_err = opt, err
        options[mid] = best
    # 2. least squares over camera + pattern sizes
    fixed = {}
    if fixed_f is not None:
        fixed[0] = fixed_f
    if fixed_pp is not None:
        fixed[1], fixed[2] = fixed_pp
    free_idx = [i for i in range(len(init_params)) if i not in fixed]
    base = np.array(init_params, dtype=float).copy()
    for i, v in fixed.items():
        base[i] = v

    def reduced(p):
        full = base.copy()
        full[free_idx] = p
        return residuals(full, obs, options, W, H)

    sol = least_squares(reduced, base[free_idx], max_nfev=iters)
    params = base.copy()
    params[free_idx] = sol.x
    rms = float(np.sqrt(np.mean(sol.fun ** 2)))
    return params, rms, options


def rvec_tvec_from_pose(pos, forward, up):
    """OpenCV rvec/tvec for a camera at pos with given forward and up vectors (world coords)."""
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, forward)
    R = np.vstack([right, -cam_up, forward])   # OpenCV camera axes: x right, y down, z forward
    rvec, _ = cv2.Rodrigues(R)
    tvec = -R @ pos
    return rvec.ravel(), tvec


def detect(bgr, detector, clahe):
    corners, ids, _ = detector.detectMarkers(clahe.apply(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)))
    out = {}
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            if int(i) in (10, 23, 42):
                out[int(i)] = c.reshape(4, 2)
    return out


def synthetic_test():
    import mujoco
    import catch_sim as cs
    model = mujoco.MjModel.from_xml_path(str(HERE / 'catch_x_user.xml'))
    data = mujoco.MjData(model)
    qa = int(model.jnt_qposadr[model.joint('carriage_slide').id])
    W = min(1280, int(model.vis.global_.offwidth))
    H = min(720, int(model.vis.global_.offheight))
    # Ground truth: the sim camera (same as your tested script)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat[:] = [0.0, 0.5, 1.2]; cam.distance = 1.0; cam.azimuth = -90.0; cam.elevation = -15.0
    az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
    fwd_dir = np.array([math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)])
    pos_true = np.array(cam.lookat) - cam.distance * fwd_dir
    fwd = np.array(cam.lookat) - pos_true; fwd /= np.linalg.norm(fwd)
    f_true = (H / 2) / math.tan(math.radians(model.vis.global_.fovy) / 2)
    rvec_true, tvec_true = rvec_tvec_from_pose(pos_true, fwd, np.array([0, 0, 1.0]))
    true_params = np.concatenate([[f_true, W / 2, H / 2], rvec_true, tvec_true, [0.0375, 0.050]])

    rend = mujoco.Renderer(model, height=H, width=W)
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                                       cv2.aruco.DetectorParameters())
    clahe = cv2.createCLAHE(2.0, (8, 8))
    obs = []
    for q in [0.0, 0.108, 0.216, 0.324, 0.432]:
        data.qpos[qa] = q
        mujoco.mj_forward(model, data)
        rend.update_scene(data, camera=cam)
        bgr = cv2.cvtColor(rend.render(), cv2.COLOR_RGB2BGR)
        for mid, corners in detect(bgr, detector, clahe).items():
            obs.append((mid, q, corners))
    print(f'Detections: {len(obs)} marker observations at 5 carriage positions')

    # Start from a deliberately wrong guess: focal 20 % off, camera rotated and moved
    init = true_params.copy()
    init[0] *= 0.8
    init[3:6] += [0.05, -0.04, 0.03]
    init[6:9] += [0.10, -0.05, 0.10]
    init[9:11] = [0.030, 0.040]
    params, rms, options = fit_camera(obs, W, H, init)
    print('--- Fit 1: focal length free')
    print(f'Fit RMS reprojection error: {rms:.2f} px')
    print(f'focal  true {f_true:.1f}  fitted {params[0]:.1f}')
    print(f'cx, cy true ({W/2:.1f}, {H/2:.1f})  fitted ({params[1]:.1f}, {params[2]:.1f})')
    print(f'camera position true {np.round(pos_true, 3)}  fitted {np.round(-np.linalg.inv(cv2.Rodrigues(params[3:6])[0]) @ params[6:9], 3)}')
    print(f'pattern half-size 10/23 fitted {params[9]*1000:.1f} mm, 42 fitted {params[10]*1000:.1f} mm')
    print('orientation options chosen:', options)
    # Distance check against your measurement: markers 10 and 23 are 0.5275 m apart
    d = np.linalg.norm(marker_centre(23, 0) - marker_centre(10, 0))
    print(f'Known marker spacing {d:.4f} m (your measurement: 0.5275 m)')

    # Fit 2: focal length fixed to a value from a chessboard calibration (here: the true value)
    params2, rms2, _ = fit_camera(obs, W, H, init, fixed_f=f_true)
    print('--- Fit 2: focal length fixed (as if from a chessboard)')
    print(f'Fit RMS reprojection error: {rms2:.2f} px')
    print(f'camera position true {np.round(pos_true, 3)}  fitted {np.round(-np.linalg.inv(cv2.Rodrigues(params2[3:6])[0]) @ params2[6:9], 3)}')
    print(f'pattern half-size 10/23 fitted {params2[9]*1000:.1f} mm, 42 fitted {params2[10]*1000:.1f} mm')
    return params2, rms2


if __name__ == '__main__':
    synthetic_test()
