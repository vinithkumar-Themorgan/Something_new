CATCH-X corrected marker position

1. Extract all files together, keeping the meshes folder.
2. Run demo_camera.py using your mujoco_env Python.
3. It writes mujoco_camera.png in this folder (if offscreen rendering succeeds).
4. Run detect_markers_fixed.py to write aruco_detection_result.png.
5. To change initial camera view, change cam.azimuth = 90 to -90 in demo_camera.py.

The catcher marker now uses the same position and quaternion as catcher_1.
The STEP-extracted white faces are visual geometry, not verified ArUco IDs.
