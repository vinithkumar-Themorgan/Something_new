"""Live camera pose from the fixed markers 10 and 23, with noise filtering.

Recomputed on every frame, so the overlay follows the camera if it is moved. Marker 42 is not used
here because it moves with the carriage.

Noise filtering:
  * A frame is rejected if its corners don't fit the model (reprojection error too large).
  * A frame is rejected if the camera position or angle jumps far from the recent poses.
    After several rejections in a row the filter accepts the new pose again, so a real move is followed.
  * The reported pose is the median of the last SMOOTH_N accepted poses, which ignores single bad frames.

Needs camera_pose_10_23.json (intrinsics and pattern size from the calibration fit) and camera_fit.py.
"""
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np

import camera_fit as cf

HERE = Path(__file__).resolve().parent
POSE_FILE = HERE / 'camera_pose_10_23.json'
MAX_REPROJ_PX = 3.0        # reject a frame whose 10/23 corners don't fit the model this well
SMOOTH_N = 21              # median window, in accepted frames
JUMP_POS_M = 0.10          # a single-frame camera position jump larger than this is rejected
JUMP_ANG_DEG = 4.0         # ... or an angle jump larger than this
MAX_REJECTS_IN_A_ROW = 3   # after this many rejections, accept the new pose (real camera move)
EMA_ALPHA = 0.3            # extra exponential smoothing of the pose (lower = steadier, more lag)
HOLD_FRAMES = 15           # keep the last good pose for this many frames when markers drop out


def _cam_pose(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    pos = (-R.T @ tvec).ravel()
    return pos, R


class LivePose:
    def __init__(self, pose_path=POSE_FILE):
        p = json.loads(Path(pose_path).read_text())
        self.K = np.array(p['K'], dtype=np.float64)
        self.dist = np.array(p['distortion'], dtype=np.float64)
        self.half = float(p['pattern_half_m']['10_23'])
        self.opts = {int(k): tuple(v) for k, v in p['orientation_options'].items()}
        # Model corners of markers 10 and 23 (fixed world positions)
        self.obj = np.vstack([cf.option_corners(cf.marker_centre(m, 0.0), self.half, self.opts[m])
                              for m in (10, 23)]).astype(np.float64)
        # Start from the calibrated pose, so the first frame converges to the right solution
        self.rvec = np.array(p['rvec'], dtype=np.float64).reshape(3, 1)
        self.tvec = np.array(p['tvec'], dtype=np.float64).reshape(3, 1)
        self.rms = None                    # reprojection error of the latest accepted frame
        self.raw_rvec = self.rvec.copy()   # latest unfiltered pose
        self.raw_tvec = self.tvec.copy()
        self.half42 = float(p.get('pattern_half_m', {}).get('42', 0.0442))
        self._buf = deque(maxlen=SMOOTH_N)
        self._rejects = 0
        self._missed = HOLD_FRAMES + 1     # no accepted frame yet
        self._sm = None
        self.accepted = 0
        self.rejected = 0

    def update(self, detected):
        """detected: dict marker_id -> raw corners (4,2). Returns True if the pose was updated."""
        if 10 not in detected or 23 not in detected:
            self._missed += 1
            return False
        img = np.vstack([np.asarray(detected[10], np.float64), np.asarray(detected[23], np.float64)])
        ok, rv, tv = cv2.solvePnP(self.obj, img, self.K, self.dist,
                                  self.rvec.copy(), self.tvec.copy(),
                                  useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            self.rejected += 1
            self._missed += 1
            return False
        proj, _ = cv2.projectPoints(self.obj, rv, tv, self.K, self.dist)
        rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
        if rms > MAX_REPROJ_PX:
            self.rejected += 1
            self._missed += 1
            return False

        # Jump check against the recent accepted poses
        self.raw_rvec, self.raw_tvec = rv, tv
        if self._buf:
            med_r = np.median(np.array([b[0] for b in self._buf]), axis=0)
            med_t = np.median(np.array([b[1] for b in self._buf]), axis=0)
            pos_new, R_new = _cam_pose(rv, tv)
            pos_old, R_old = _cam_pose(med_r, med_t)
            dpos = np.linalg.norm(pos_new - pos_old)
            dang = np.degrees(np.linalg.norm(cv2.Rodrigues(R_new @ R_old.T)[0]))
            if (dpos > JUMP_POS_M or dang > JUMP_ANG_DEG) and self._rejects < MAX_REJECTS_IN_A_ROW:
                self._rejects += 1
                self.rejected += 1
                self._missed += 1
                return False
        self._rejects = 0
        self._missed = 0
        self._buf.append((rv.ravel().copy(), tv.ravel().copy()))
        self.accepted += 1
        self.rms = rms
        # Reported pose: median of the window (robust to single bad frames)
        new_r = np.median(np.array([b[0] for b in self._buf]), axis=0).reshape(3, 1)
        new_t = np.median(np.array([b[1] for b in self._buf]), axis=0).reshape(3, 1)
        if self._sm is None:
            self._sm = (new_r, new_t)
        else:
            self._sm = (EMA_ALPHA * new_r + (1 - EMA_ALPHA) * self._sm[0],
                        EMA_ALPHA * new_t + (1 - EMA_ALPHA) * self._sm[1])
        self.rvec, self.tvec = self._sm
        return True

    def carriage_mm_from_42(self, corners42):
        """Carriage position (mm from HOME) from the moving marker 42 in the camera image.

        Finds the carriage position q whose projected marker-42 corners best match the detected corners,
        using the current camera pose. Independent of the serial link.
        """
        from scipy.optimize import minimize_scalar
        c = np.asarray(corners42, np.float64).reshape(4, 2)
        opt = self.opts.get(42, (2, True))

        def cost(q):
            world = cf.option_corners(cf.marker_centre(42, q), self.half42, opt)
            proj, _ = cv2.projectPoints(world.reshape(-1, 1, 3), self.rvec, self.tvec, self.K, self.dist)
            return float(np.sum((proj.reshape(-1, 2) - c) ** 2))

        grid = np.linspace(0.0, 0.432, 217)
        best = grid[int(np.argmin([cost(q) for q in grid]))]
        res = minimize_scalar(cost, bounds=(max(0.0, best - 0.003), min(0.432, best + 0.003)), method='bounded')
        return float(res.x * 1000.0)

    @property
    def valid(self):
        """True while the pose is recent enough to use (holds the last good pose briefly)."""
        return self.rvec is not None and self._missed <= HOLD_FRAMES

    def camera_position(self):
        pos, _ = _cam_pose(self.rvec, self.tvec)
        return pos

    def project(self, pts_world):
        """World points (N,3) to pixels (N,2) using the filtered pose. Points behind the camera are dropped."""
        pts = np.asarray(pts_world, np.float64).reshape(-1, 3)
        R, _ = cv2.Rodrigues(self.rvec)
        cam = (R @ pts.T + self.tvec).T
        front = cam[:, 2] > 0
        img, _ = cv2.projectPoints(pts[front].reshape(-1, 1, 3), self.rvec, self.tvec, self.K, self.dist)
        return img.reshape(-1, 2)
