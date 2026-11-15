"""INTERCEPT-X: Manually move the real catcher; MuJoCo follows via ArUco.

Place next to catch_x.xml and meshes/. Install mujoco, numpy, opencv-contrib-python.
Keys (in camera window): H=home, E=end, R=reset, S=save, Q/Esc=quit.
"""
from pathlib import Path
import time
import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
XML_PATH = ROOT / "catch_x.xml"
CAMERA_INDEX = 0  # Change to 1 if your USB camera is the second camera.
CAMERA_WIDTH, CAMERA_HEIGHT = 1280, 720
FIXED_A, FIXED_B, MOVING = 23, 10, 42

model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data = mujoco.MjData(model)
joint_id = model.joint("carriage_slide").id
qaddr = int(model.jnt_qposadr[joint_id])
daddr = int(model.jnt_dofadr[joint_id])
low, high = map(float, model.jnt_range[joint_id])
if high <= low:
    raise RuntimeError(f"Invalid carriage_slide range: {low}, {high}")
print(f"MuJoCo slide range: {low:.3f} to {high:.3f} m ({(high-low)*1000:.1f} mm)")

# The physical camera observes the real markers; the MuJoCo renderer shows the twin.
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)  # Windows
if not cap.isOpened():
    cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, params)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

cam = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model, cam)
cam.lookat[:] = [0.0, 0.5, 1.2]
cam.distance = 1.0
cam.azimuth = -90
cam.elevation = -15
render_width = min(960, int(model.vis.global_.offwidth))
render_height = min(540, int(model.vis.global_.offheight))

home_ref = None
end_ref = None
last_measurement = None
last_position_m = low
data.qpos[qaddr] = low
mujoco.mj_forward(model, data)

def center(marker_corners):
    return np.asarray(marker_corners, dtype=np.float64).reshape(4, 2).mean(axis=0)

def measure_fraction(positions):
    """Project moving marker onto fixed-marker line; result is dimensionless."""
    a, b, p = positions[FIXED_A], positions[FIXED_B], positions[MOVING]
    axis = b - a
    denom = float(np.dot(axis, axis))
    if denom < 1.0:
        return None
    return float(np.dot(p - a, axis) / denom)

print("Move real catcher to HOME and press H; to END and press E.")
print("R=reset calibration, S=save annotated images, Q/Esc=quit")
try:
    with mujoco.Renderer(model, width=render_width, height=render_height) as renderer:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame unavailable; retrying...")
                time.sleep(0.05)
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(clahe.apply(gray))
            positions = {}
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                for marker_corners, marker_id in zip(corners, ids.flatten()):
                    positions[int(marker_id)] = center(marker_corners)
                    cx, cy = np.rint(positions[int(marker_id)]).astype(int)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                    cv2.putText(frame, f"ID {marker_id}", (cx+8, cy-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            measured = None
            if all(mid in positions for mid in (FIXED_A, FIXED_B, MOVING)):
                measured = measure_fraction(positions)
                if measured is not None:
                    last_measurement = measured
                    a = tuple(np.rint(positions[FIXED_A]).astype(int))
                    b = tuple(np.rint(positions[FIXED_B]).astype(int))
                    p = tuple(np.rint(positions[MOVING]).astype(int))
                    cv2.line(frame, a, b, (0, 255, 0), 2)
                    cv2.line(frame, b, p, (0, 255, 0), 2)

            calibrated = (home_ref is not None and end_ref is not None
                          and abs(end_ref-home_ref) > 1e-6)
            if calibrated and measured is not None:
                progress = np.clip((measured-home_ref)/(end_ref-home_ref), 0.0, 1.0)
                last_position_m = low + float(progress)*(high-low)

            # Directly set digital twin position, no dynamic stepping.
            data.qpos[qaddr] = last_position_m
            data.qvel[daddr] = 0.0
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            twin_bgr = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)

            found = sorted(positions)
            status = "TRACKING" if measured is not None else "MARKER LOST - HOLDING"
            cv2.putText(frame, f"IDs: {found}  {status}", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(frame, f"Twin joint: {last_position_m*1000:.1f} mm", (12, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if measured is not None:
                cv2.putText(frame, f"Reference: {measured:.5f}", (12, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cal_text = (f"HOME={home_ref:.5f}" if home_ref is not None else "HOME=unset")
            cal_text += "  " + (f"END={end_ref:.5f}" if end_ref is not None else "END=unset")
            cv2.putText(frame, cal_text, (12, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, "H home | E end | R reset | S save | Q quit", (12, frame.shape[0]-18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("INTERCEPT-X | Physical camera", frame)
            cv2.imshow("INTERCEPT-X | MuJoCo digital twin", twin_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('h'):
                if measured is not None:
                    home_ref = measured
                    print(f"Home captured: {home_ref:.5f}")
                else:
                    print("Cannot capture HOME: all 3 markers must be visible")
            elif key == ord('e'):
                if measured is not None:
                    end_ref = measured
                    print(f"End captured: {end_ref:.5f}")
                else:
                    print("Cannot capture END: all 3 markers must be visible")
            elif key == ord('r'):
                home_ref = end_ref = None
                last_position_m = low
                print("Calibration reset")
            elif key == ord('s'):
                cv2.imwrite(str(ROOT / "intercept_x_physical.png"), frame)
                cv2.imwrite(str(ROOT / "intercept_x_twin.png"), twin_bgr)
                print("Saved physical and twin frames")
finally:
    cap.release()
    cv2.destroyAllWindows()
