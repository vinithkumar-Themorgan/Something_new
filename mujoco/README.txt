CATCH-X MuJoCo assembly — generated from uploaded STEP assembly and six unique STL meshes.

Files: catch_x.xml, meshes/*.stl, demo.py

Requirements: pip install mujoco
Run from this folder: python demo.py

11 STEP component instances are reconstructed using their original assembly transformations, expressed in the base-part coordinate system.
Fixed: base x1, rod x2, rod support x4.
Moving: linear bearing x2, catcher x1, catcher holder x1.

IMPORTANT: The slide joint range is temporarily set to -225..+225 mm relative to the CAD-exported carriage pose. This is NOT a verified physical home/limit position. Calibrate actual home and mechanical limits before sending hardware commands.

The ball is simulated, but ball-to-catcher collisions are not yet implemented. All CAD meshes are visual-only to avoid complex triangle collision dynamics. A physical collision model, ArUco synchronization, and ESP32 communication remain to be implemented.

Mesh scale 0.001 converts the source millimetre geometry to metres. The STEP placements are preserved and the entire assembly is expressed relative to the CAD base coordinate system.
