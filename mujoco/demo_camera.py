"""INTERCEPT-X: manually move the physical catcher; MuJoCo follows ArUco ID 42.

Requires: pip install mujoco opencv-contrib-python numpy
Place beside catch_x.xml and meshes/. Camera IDs: fixed 10,23; moving 42.
Calibration: move to physical home, press H; move to far end, press E.
Press R to reset calibration, Q/Esc to quit.
"""
from pathlib import Path
import time
import cv2
import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'catch_x.xml'
CAMERA_INDEX = 0  # Change to 1 or 2 if your USB camera is not index 0
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
JOINT_NAME = 'carriage_slide'
FIXED_A, FIXED_B, MOVING = 10, 23, 42
SMOOTHING = 0.35  # 1.0 = no smoothing; smaller = smoother, more latency


def marker_centers(corners, ids):
    if ids is None:
        return {}
    return {int(mid): pts.reshape(4, 2).mean(axis=0)
            for pts, mid in zip(corners, ids.flatten())}


def projected_coordinate(centers):
    """Fraction along fixed-marker baseline, using a 2D dot product.

    This is perspective-approximate, not a calibrated 3D pose estimate.
    """
    if not all(mid in centers for mid in (FIXED_A, FIXED_B, MOVING)):
        return None
    a, b, p = (centers[k] for k in (FIXED_A, FIXED_B, MOVING))
    axis = b - a
    length_sq = float(np.dot(axis, axis))
    if length_sq < 100.0:  # avoid unstable reference baseline
        return None
    return float(np.dot(p - a, axis) / length_sq)


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    jid = model.joint(JOINT_NAME).id
    qaddr = int(model.jnt_qposadr[jid])
    daddr = int(model.jnt_dofadr[jid])
    low, high = map(float, model.jnt_range[jid])
    travel_mm = (high - low) * 1000.0
    data.qpos[qaddr] = low
    mujoco.mj_forward(model, data)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open camera index {CAMERA_INDEX}. Try 1 or 2.')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    home_fraction = None
    end_fraction = None
    filtered_mm = None
    last_good = None
    print(f'Camera opened. Slide travel: {travel_mm:.1f} mm')
    print('Move catcher to HOME -> press H. Move to END -> press E. R reset. Q quit.')

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = [0, 0, 0.12]
            viewer.cam.distance = 1.1
            viewer.cam.azimuth = -90
            viewer.cam.elevation = -25
            while viewer.is_running():
                ok, frame = cap.read()
                if not ok:
                    print('Camera read failed; retrying...')
                    time.sleep(0.05)
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                enhanced = clahe.apply(gray)
                corners, ids, _ = detector.detectMarkers(enhanced)
                centers = marker_centers(corners, ids)
                fraction = projected_coordinate(centers)
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    for mid, center in centers.items():
                        x, y = np.rint(center).astype(int)
                        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
                        cv2.putText(frame, f'ID {mid}', (x + 8, y - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

                calibrated = (home_fraction is not None and end_fraction is not None
                              and abs(end_fraction - home_fraction) > 0.02)
                if fraction is not None and calibrated:
                    # Home and end measurements automatically establish travel direction.
                    measured_mm = travel_mm * ((fraction - home_fraction) /
                                               (end_fraction - home_fraction))
                    measured_mm = float(np.clip(measured_mm, 0, travel_mm))
                    filtered_mm = (measured_mm if filtered_mm is None else
                                   SMOOTHING * measured_mm + (1 - SMOOTHING) * filtered_mm)
                    data.qpos[qaddr] = low + filtered_mm / 1000.0
                    data.qvel[daddr] = 0.0
                    mujoco.mj_forward(model, data)
                    last_good = time.monotonic()
                    status = f'Physical: {measured_mm:6.1f} mm | Twin: {filtered_mm:6.1f} mm'
                elif not calibrated:
                    status = 'CALIBRATE: home -> H, end -> E (all 3 IDs must be visible)'
                else:
                    age = time.monotonic() - last_good if last_good is not None else 0
                    status = f'MARKER LOST: holding last position ({age:.1f}s)'

                cv2.putText(frame, status, (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Detected: {sorted(centers)}  |  H home  E end  R reset  Q quit',
                            (15, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('INTERCEPT-X | Physical camera and MuJoCo sync', frame)
                viewer.sync()
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                if key == ord('h'):
                    if fraction is None:
                        print('HOME calibration failed: need all three markers visible.')
                    else:
                        home_fraction = fraction
                        filtered_mm = None
                        print(f'Home captured: {home_fraction:.5f}')
                elif key == ord('e'):
                    if fraction is None:
                        print('END calibration failed: need all three markers visible.')
                    else:
                        end_fraction = fraction
                        filtered_mm = None
                        print(f'End captured: {end_fraction:.5f}')
                elif key == ord('r'):
                    home_fraction = end_fraction = filtered_mm = None
                    print('Calibration reset')
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
