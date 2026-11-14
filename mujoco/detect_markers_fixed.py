"""Live MuJoCo virtual camera with ArUco outlines and IDs.
Place beside catch_x.xml and meshes/. Requires mujoco, opencv-contrib-python, numpy.
Keys: Q/Esc quit, A/D move carriage, S save annotated frame.
"""
from pathlib import Path
import time
import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'catch_x.xml'
model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

# Respect the framebuffer limit set in catch_x.xml.
width = min(1280, int(model.vis.global_.offwidth))
height = min(720, int(model.vis.global_.offheight))
print(f'Render size: {width}x{height}. For 1280x720 add '
      '<visual><global offwidth="1280" offheight="720"/></visual> to XML.')

camera = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model, camera)
camera.lookat[:] = [0.0, 0.5, 1.2]
camera.distance = 1.0
camera.azimuth = -90.0  # Change to -90 if viewing the back of markers
camera.elevation = -15.0

joint_id = model.joint('carriage_slide').id
qaddr = int(model.jnt_qposadr[joint_id])
daddr = int(model.jnt_dofadr[joint_id])
low, high = map(float, model.jnt_range[joint_id])
data.qpos[qaddr] = 0.0
mujoco.mj_forward(model, data)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, params)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

print('Live detection started. A/D: move catcher 5 mm, S: save frame, Q/Esc: quit.')
point_1 = 0
point_2 = 0
known_distance=0.5275

def get_center(corners):
    points = corners.reshape(4, 2)
    return np.mean(points, axis=0)

try:
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        while True:
            # Direct position control: do not step dynamics or move the carriage via gravity.
            data.qvel[daddr] = 0.0
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            rgb = renderer.render()
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            enhanced = clahe.apply(gray)
            corners, ids, rejected = detector.detectMarkers(enhanced)
            positions = {}
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                found = ids.flatten().tolist()
                for marker_corners, marker_id in zip(corners, ids.flatten()):

                            center = get_center(marker_corners)
                            cx, cy = center.astype(int)
                
                            positions[int(marker_id)] = center                
                            cv2.circle(
                                frame, (cx, cy), 3, (0, 255, 0), -1
                            )
                
                            cv2.putText(
                                frame,
                                f"ID {marker_id}: ({cx}, {cy})",
                                (cx + 10, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2
                            )
            else:
                found = []

            id_10 = 10
            id_23 = 23
            id_42 = 42

            if id_10 in positions and id_23 in positions and id_42:
                pt1 = (int(positions[23][0]) , int(positions[23][1]) )
                pt2 = (int(positions[10][0]) , int(positions[10][1]) )
                pt3 = (int(positions[42][0]) , int(positions[42][1]) )
                dx1 = pt2[0]-pt1[0]
                dy1 = pt2[1]-pt1[1]
                cx2 = int(pt2[0]+pt3[0])//2
                cy2 = int(pt2[1]+pt3[1])//2

                pixel_distance = np.hypot(dx1,dy1)
                pixel_meter = known_distance/pixel_distance
                dx2 = pt3[0]-pt2[0]
                dy2 = pt3[1]-pt2[1]
                pixel_distance_a1 = np.hypot(dx2,dy2)
                pixel_distance_a2 = pt3[0] - pt2[0]
                original_distance_a1 = pixel_meter * pixel_distance_a1
                original_distance_a2 = pixel_meter * pixel_distance_a2
                print(original_distance_a2)

                # Draw a green line with thickness 3
                cv2.line(frame, pt1, pt2, color=(0, 255, 0), thickness=2)

                cv2.line(frame, pt2, pt3, color=(0, 255, 0), thickness=2)
                cv2.circle(frame, (cx2, cy2), radius=3, color = (0, 255, 0), thickness = -1)

            cv2.putText(frame, f'Detected IDs: {found}', (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
            position_mm = (data.qpos[qaddr] - low) * 1000
            cv2.putText(frame, f'Carriage: {position_mm:.0f} mm from joint minimum',
                        (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow('CATCH-X | MuJoCo live ArUco detection', frame)
            key = cv2.waitKey(1) & 0xFF
            """print("current_position :", data.qpos[qaddr])"""
            if key in (ord('q'), 27):
                break
            elif key == ord('a'):
                data.qpos[qaddr] = np.clip(data.qpos[qaddr] - 0.005, low, high)
            elif key == ord('d'):
                data.qpos[qaddr] = np.clip(data.qpos[qaddr] + 0.005, low, high)
            elif key == ord('s'):
                out = ROOT / 'aruco_live_annotated.png'
                
                if cv2.imwrite(str(out), frame):
                    print('Saved:', out)
                else:
                    print('Failed to save:', out)
finally:
    cv2.destroyAllWindows()
