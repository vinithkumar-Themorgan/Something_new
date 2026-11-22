"""Virtual-camera check: project known 3D points into a MuJoCo render and compare with ArUco detections.

1. Renders the scene from a virtual camera (same pose as your tested live script).
2. Detects ArUco markers 10, 23 and 42 in that render.
3. Projects the known marker centres (marker_layout.json) with the camera model and prints the pixel error.
4. Projects a simulated ball flight onto the same frame and saves the annotated image.
"""
from pathlib import Path
import json, math
import numpy as np
import cv2
import mujoco
import catch_sim as cs

HERE = Path(__file__).resolve().parent
LAYOUT = Path(r'C:\Users\vinith\Downloads\CATCH-X_Corrected_Marker_and_Camera\marker_layout.json')
OUT_PNG = HERE / 'virtual_cam_check.png'

CARRIAGE_MM = 319.0                # catcher position for this check (bucket x = -0.219 + mm/1000)
LANDING_X = 0.10                   # where the simulated ball will cross the rim
T_FLIGHT = 1.2

model = mujoco.MjModel.from_xml_path(str(cs.XML))
data = mujoco.MjData(model)
# Render size limited by the XML framebuffer (<visual><global offwidth/offheight>)
W = min(1280, int(model.vis.global_.offwidth))
H = min(720, int(model.vis.global_.offheight))
print(f'Render size {W}x{H}')
carriage_adr = int(model.jnt_qposadr[model.joint('carriage_slide').id])
ball_jid = model.joint('ball_free').id

# Same virtual camera as your tested script
cam = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model, cam)
cam.lookat[:] = [0.0, 0.5, 1.2]
cam.distance = 1.0
cam.azimuth = -90.0
cam.elevation = -15.0


def make_projector(cam, fovy_deg):
    """World point -> pixel (u, v), MuJoCo free-camera convention with vertical field of view fovy."""
    az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
    look = np.array(cam.lookat, float)
    fwd_dir = np.array([math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)])
    pos = look - cam.distance * fwd_dir              # camera sits behind the lookat point
    f = look - pos
    f /= np.linalg.norm(f)
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    u_axis = np.cross(r, f)
    focal = (H / 2) / math.tan(math.radians(fovy_deg) / 2)

    def project(p):
        d = np.asarray(p, float) - pos
        x, y, z = d @ r, d @ u_axis, d @ f
        if z <= 0:
            return None
        return np.array([W / 2 + focal * x / z, H / 2 - focal * y / z])
    return project


project = make_projector(cam, float(model.vis.global_.fovy))

# Carriage in place for the check
data.qpos[carriage_adr] = CARRIAGE_MM / 1000.0
data.qpos[model.jnt_qposadr[ball_jid]:model.jnt_qposadr[ball_jid] + 3] = [0.0, 0.0, -10.0]   # hide the XML ball
mujoco.mj_forward(model, data)

# Render + ArUco detection, same settings as your live script
renderer = mujoco.Renderer(model, height=H, width=W)
renderer.update_scene(data, camera=cam)
rgb = renderer.render()
frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                                   cv2.aruco.DetectorParameters())
corners, ids, _ = detector.detectMarkers(clahe.apply(gray))
detected = {}
if ids is not None:
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    for c, i in zip(corners, ids.flatten()):
        detected[int(i)] = c.reshape(4, 2).mean(axis=0)

# Known marker centres (world)
layout = json.loads(LAYOUT.read_text())
centres = {
    10: np.array(layout['base_markers_world']['id10_home']['centre']),
    23: np.array(layout['base_markers_world']['id23_end']['centre']),
    42: np.array(layout['catcher_id42_at_home']['centre']) + [CARRIAGE_MM / 1000.0, 0.0, 0.0],
}

print('Marker  projected (px)        detected (px)        error (px)')
for mid in (10, 23, 42):
    p = project(centres[mid])
    d = detected.get(mid)
    if p is None or d is None:
        print(f'  {mid:3d}   not visible (projected={p is not None}, detected={d is not None})')
        continue
    err = np.linalg.norm(p - d)
    print(f'  {mid:3d}   ({p[0]:7.1f}, {p[1]:6.1f})   ({d[0]:7.1f}, {d[1]:6.1f})   {err:6.1f}')
    cv2.circle(frame, tuple(int(v) for v in p), 6, (0, 0, 255), 2)        # red: projected centre

# Simulated ball flight projected into the same frame
g = 9.81
p0 = cs.LAUNCH_POS.copy()
v0 = np.array([(LANDING_X - p0[0]) / T_FLIGHT,
               (0.0 - p0[1]) / T_FLIGHT,
               (cs.BUCKET_TOP_Z - p0[2] + 0.5 * g * T_FLIGHT ** 2) / T_FLIGHT])
traj = cs.simulate_ball(model, data, ball_jid, p0, v0, t_max=T_FLIGHT + 0.2)
pts = [project(p) for _, p, _ in traj]
pts = [p for p in pts if p is not None]
for a, b in zip(pts, pts[1:]):
    cv2.line(frame, tuple(int(v) for v in a), tuple(int(v) for v in b), (255, 200, 0), 2)
cross = cs.rim_crossing(traj)
if cross is not None:
    _, xc, yc = cross
    q = project([xc, yc, cs.BUCKET_TOP_Z])
    if q is not None:
        cv2.circle(frame, tuple(int(v) for v in q), 8, (0, 255, 255), 2)
        cv2.putText(frame, 'rim crossing', (int(q[0]) + 10, int(q[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

cv2.putText(frame, f'Carriage {CARRIAGE_MM:.0f} mm | ball path (blue) | red = projected marker centre',
            (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imwrite(str(OUT_PNG), frame)
print(f'Saved {OUT_PNG}')
