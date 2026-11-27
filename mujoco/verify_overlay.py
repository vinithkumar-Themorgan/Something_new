
"""
Verify physical camera pose using fixed ArUco markers 10 and 23.

Camera input: eMeet C960 webcam via FFmpeg (DirectShow).
Does not move the carriage.

Green = detected marker corners
Red   = projected model corners

Outputs:
    camera_pose_10_23.json
    overlay_check.png

Usage:
    python verify_overlay.py
"""

import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

import camera_fit as cf


# ============================================================
# CONFIGURATION
# ============================================================

HERE = Path(__file__).resolve().parent

INTRINSICS = HERE / "intrinsics_run2.json"
CAPTURE = HERE / "capture_markers.json"

POSE_OUT = HERE / "camera_pose_10_23.json"
OUT_PNG = HERE / "overlay_check.png"

CAMERA_NAME = "HD Webcam eMeet C960"

W = 640
H = 480
FPS = 30

# Current physical carriage position.
# Do not change unless the carriage has moved.
CARRIAGE_MM_NOW = 432.0

ARUCO_DICT = cv2.aruco.DICT_4X4_50


# ============================================================
# CAMERA CAPTURE USING FFMPEG
# ============================================================

def grab_ffmpeg_frame():
    """
    Capture one camera frame using FFmpeg.

    Avoids OpenCV VideoCapture DirectShow/MSMF errors.
    Returns a BGR NumPy image.
    """

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-f", "dshow",
        "-video_size", f"{W}x{H}",
        "-framerate", str(FPS),
        "-i", f"video={CAMERA_NAME}",
        "-frames:v", "1",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1",
    ]

    print("\nOpening camera using FFmpeg...")
    print(f"Camera: {CAMERA_NAME}")
    print(f"Resolution: {W}x{H}")
    print(f"Requested FPS: {FPS}")

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False
        )

    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "FFmpeg camera capture timed out after 30 seconds."
        )

    except FileNotFoundError:
        raise RuntimeError(
            "FFmpeg executable not found. Check system PATH."
        )

    expected_bytes = W * H * 3

    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg failed:\n" +
            result.stderr.decode("utf-8", errors="replace")
        )

    if len(result.stdout) != expected_bytes:
        raise RuntimeError(
            f"Invalid frame size: {len(result.stdout)} bytes. "
            f"Expected {expected_bytes} bytes.\n"
            + result.stderr.decode("utf-8", errors="replace")
        )

    frame = np.frombuffer(
        result.stdout,
        dtype=np.uint8
    ).reshape(H, W, 3).copy()

    print("Camera frame captured successfully.")

    return frame


# ============================================================
# CAMERA POSE FIT
# ============================================================

def fit_pose_10_23():
    """
    Fit camera pose using fixed markers 10 and 23.

    Marker 42 is excluded from the fit.
    """

    print("\nLoading camera calibration...")

    intr = json.loads(INTRINSICS.read_text())

    K = np.array(
        intr["K"],
        dtype=np.float64
    )

    dist = np.array(
        intr["dist"],
        dtype=np.float64
    )

    cap_data = json.loads(CAPTURE.read_text())

    f = float((K[0, 0] + K[1, 1]) / 2)

    cx = float(K[0, 2])
    cy = float(K[1, 2])

    obs = []

    for entry in cap_data["entries"]:

        q = float(entry["q_mm"]) / 1000.0

        for marker_id, corners in entry["markers"].items():

            mid = int(marker_id)

            # Only fixed markers 10 and 23.
            if mid not in (10, 23):
                continue

            corners_np = np.array(
                corners,
                dtype=np.float64
            ).reshape(-1, 1, 2)

            undistorted = cv2.undistortPoints(
                corners_np,
                K,
                dist,
                P=K
            ).reshape(4, 2)

            obs.append(
                (mid, q, undistorted)
            )

    if not obs:
        raise RuntimeError(
            "No fixed-marker observations found in capture_markers.json."
        )

    print(f"Loaded {len(obs)} fixed-marker observations.")
    print("Fitting camera pose...")

    best = None

    look = np.array(
        [0.0, 0.0, 0.85],
        dtype=np.float64
    )

    for az_deg in (-100, -90, -80):

        for el_deg in (-30, -20, -10):

            for dist_m in (1.0, 1.5, 2.0):

                az = math.radians(az_deg)
                el = math.radians(el_deg)

                forward_direction = np.array([
                    math.cos(az) * math.cos(el),
                    math.sin(az) * math.cos(el),
                    math.sin(el)
                ])

                pos = look - dist_m * forward_direction

                fwd = look - pos
                fwd /= np.linalg.norm(fwd)

                rvec, tvec = cf.rvec_tvec_from_pose(
                    pos,
                    fwd,
                    np.array([0.0, 0.0, 1.0])
                )

                init = np.concatenate([
                    [f, cx, cy],
                    rvec,
                    tvec,
                    [0.03, 0.04]
                ])

                params, rms, options = cf.fit_camera(
                    obs,
                    W,
                    H,
                    init,
                    fixed_f=f,
                    fixed_pp=(cx, cy)
                )

                if best is None or rms < best[1]:
                    best = (
                        params,
                        rms,
                        options
                    )

    if best is None:
        raise RuntimeError("Camera pose fitting failed.")

    params, rms, options = best

    R, _ = cv2.Rodrigues(
        np.array(params[3:6], dtype=np.float64)
    )

    camera_position = -R.T @ params[6:9]

    pose_data = {
        "K": [
            [float(params[0]), 0.0, float(params[1])],
            [0.0, float(params[0]), float(params[2])],
            [0.0, 0.0, 1.0]
        ],
        "rvec": params[3:6].tolist(),
        "tvec": params[6:9].tolist(),
        "camera_position_m": camera_position.tolist(),
        "rms_px": float(rms),
        "pattern_half_m": {
            "10_23": float(params[9])
        },
        "orientation_options": {
            str(k): list(v)
            for k, v in options.items()
        },
        "image_size": [W, H],
        "distortion": dist.tolist(),
        "focal_px": f,
        "note": (
            "Camera pose fitted from fixed markers "
            "10 and 23 only using intrinsics_run2.json"
        )
    }

    POSE_OUT.write_text(
        json.dumps(pose_data, indent=2)
    )

    print("\nCamera pose fitting completed.")
    print(f"RMS error: {rms:.3f} px")
    print(
        "Camera position (m):",
        np.round(camera_position, 4)
    )
    print(f"Saved pose: {POSE_OUT}")

    return params, options, K, dist


# ============================================================
# ARUCO DETECTION
# ============================================================

def detect_aruco(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    dictionary = cv2.aruco.getPredefinedDictionary(
        ARUCO_DICT
    )

    detector_params = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(
        dictionary,
        detector_params
    )

    corners, ids, _ = detector.detectMarkers(gray)

    detected = {}

    if ids is not None:

        for corner, marker_id in zip(
            corners,
            ids.flatten()
        ):

            mid = int(marker_id)

            if mid in (10, 23, 42):
                detected[mid] = corner.reshape(4, 2)

    print("\nDetected markers:", sorted(detected.keys()))

    return detected


# ============================================================
# VISUAL OVERLAY
# ============================================================

def create_overlay(frame, detected, params, options, K, dist):
    """
    Green: actual detected image corners.
    Red: projected model corners.

    The camera model uses undistorted points.
    Therefore both the image and detected points are
    undistorted before drawing.
    """

    vis = cv2.undistort(frame, K, dist, None, K)

    q_m = CARRIAGE_MM_NOW / 1000.0

    print("\nMarker    Mean Error (px)    Status")
    print("----------------------------------------")

    for mid in (10, 23, 42):

        if mid == 42:
            half = float(params[10])
        else:
            half = float(params[9])

        opt = options.get(mid)

        if opt is None:
            # Fallback orientation for moving marker 42.
            # Must match its actual mounted orientation.
            opt = (2, True)

        centre = cf.marker_centre(
            mid,
            q_m
        )

        world_corners = cf.option_corners(
            centre,
            half,
            opt
        )

        projected = np.asarray(
            cf.project(
                params,
                world_corners,
                W,
                H
            ),
            dtype=np.float64
        ).reshape(4, 2)

        # Draw projected model corners in red.
        for index, point in enumerate(projected):

            x = int(round(point[0]))
            y = int(round(point[1]))

            cv2.circle(
                vis,
                (x, y),
                5,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                vis,
                str(index),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1
            )

        if mid in detected:

            actual = np.asarray(
                detected[mid],
                dtype=np.float64
            ).reshape(4, 1, 2)

            actual = cv2.undistortPoints(
                actual,
                K,
                dist,
                P=K
            ).reshape(4, 2)

            for point in actual:

                x = int(round(point[0]))
                y = int(round(point[1]))

                cv2.circle(
                    vis,
                    (x, y),
                    5,
                    (0, 255, 0),
                    2
                )

            error = np.linalg.norm(
                projected - actual,
                axis=1
            ).mean()

            print(
                f"{mid:>4}          "
                f"{error:>8.3f}       "
                f"Detected"
            )

            cv2.putText(
                vis,
                f"ID {mid}: {error:.2f}px",
                (
                    10,
                    65 + (10, 23, 42).index(mid) * 25
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

        else:

            print(
                f"{mid:>4}             ---        "
                f"Not detected"
            )

    cv2.putText(
        vis,
        "Green: Detected  Red: Projected",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )

    cv2.imwrite(str(OUT_PNG), vis)

    print(f"\nOverlay saved: {OUT_PNG}")

    return vis


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("PHYSICAL CAMERA POSE VERIFICATION")
    print("=" * 60)

    print("\nMode: STATIC VERIFICATION")
    print("Carriage movement: DISABLED")
    print(f"Assumed carriage position: {CARRIAGE_MM_NOW} mm")

    # Step 1: Fit camera pose from saved observations.
    params, options, K, dist = fit_pose_10_23()

    # Step 2: Capture frame using FFmpeg.
    frame = grab_ffmpeg_frame()

    # Step 3: Detect ArUco markers.
    detected = detect_aruco(frame)

    # Step 4: Create projected model overlay.
    vis = create_overlay(
        frame,
        detected,
        params,
        options,
        K,
        dist
    )

    print("\nVerification completed.")
    print("No motor movement commands were executed.")

    # Preview result.
    try:
        cv2.imshow(
            "Camera Pose Verification",
            vis
        )

        print("Press any key to close preview.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except cv2.error:
        print(
            "Preview unavailable; check the saved overlay PNG."
        )


if __name__ == "__main__":

    try:
        main()

    except Exception as exc:
        print("\nERROR:", str(exc))
        raise
