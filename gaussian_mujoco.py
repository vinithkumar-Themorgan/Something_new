"""Lesson: Gaussian elimination applied to an actual MuJoCo 2R robot Jacobian.

Install: python -m pip install mujoco numpy matplotlib
Run:     python gaussian_mujoco.py
Optional: python gaussian_mujoco.py --view  (opens live MuJoCo viewer on desktop)
"""
from pathlib import Path
import argparse
import numpy as np
import mujoco
import matplotlib.pyplot as plt

XML_FILE = Path(__file__).with_name("two_link.xml")


def gaussian_elimination(A, b, explain=False):
    """Solve Ax=b with elimination + partial pivoting, NOT np.linalg.solve."""
    U = np.array(A, dtype=float, copy=True)
    rhs = np.array(b, dtype=float, copy=True)
    n = len(rhs)

    for k in range(n - 1):
        # Move largest available pivot into place to avoid divide-by-zero.
        pivot = k + int(np.argmax(np.abs(U[k:, k])))
        if abs(U[pivot, k]) < 1e-12:
            raise ValueError("Singular Jacobian: cannot solve with ordinary elimination")
        if pivot != k:
            U[[k, pivot]] = U[[pivot, k]]
            rhs[[k, pivot]] = rhs[[pivot, k]]
            if explain:
                print(f"Swap rows {k} and {pivot} for stable pivoting")

        for i in range(k + 1, n):
            m = U[i, k] / U[k, k]  # how much pivot row to subtract
            if explain:
                print(f"k={k}, i={i}, multiplier m={m:.6f}")
            U[i, k:] -= m * U[k, k:]  # eliminate element below pivot
            rhs[i] -= m * rhs[k]      # IMPORTANT: apply operation to RHS too
            if explain:
                print("Upper matrix now:\n", U)
                print("Updated RHS:", rhs)

    if abs(U[-1, -1]) < 1e-12:
        raise ValueError("Singular Jacobian: zero final pivot")

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (rhs[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]
    if explain:
        print("Joint velocity solution [rad/s]:", x)
    return x


def tool_xy_jacobian(model, data, site_id):
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return jacp[:2, :2].copy()  # planar X/Y vs shoulder/elbow joint velocities


def run_simulation(view=False):
    model = mujoco.MjModel.from_xml_path(str(XML_FILE))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("start").id)
    mujoco.mj_forward(model, data)
    site_id = model.site("tool").id

    # End-effector target velocity: +30 mm/s X, +20 mm/s Y
    desired = np.array([0.030, 0.020])
    J = tool_xy_jacobian(model, data, site_id)
    print("Initial joint positions [deg]:", np.rad2deg(data.qpos[:2]))
    print("Initial MuJoCo Jacobian [m/rad]:\n", J)
    print("Target tool velocity [m/s]:", desired)
    qdot = gaussian_elimination(J, desired, explain=True)
    print("Check J @ qdot [m/s]:", J @ qdot)
    print("Compare NumPy solution:", np.linalg.solve(J, desired))
    print("Initial end-effector position [m]:", data.site_xpos[site_id, :2])

    viewer = None
    if view:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(model, data)

    time_log, xy_log, vel_log, qdot_log = [], [], [], []
    steps = int(2.0 / model.opt.timestep)

    try:
        for step in range(steps):
            # Read current kinematics; the Jacobian changes as the arm moves.
            mujoco.mj_forward(model, data)
            J = tool_xy_jacobian(model, data, site_id)
            try:
                target_joint_velocity = gaussian_elimination(J, desired)
            except ValueError:
                print("Near a singular configuration: stopping velocity control")
                data.ctrl[:] = 0
                break

            # Velocity actuators have finite response: actual != commanded instantly.
            data.ctrl[:] = np.clip(target_joint_velocity, -3.0, 3.0)
            measured_tool_velocity = J @ data.qvel[:2]
            time_log.append(data.time)
            xy_log.append(data.site_xpos[site_id, :2].copy())
            vel_log.append(measured_tool_velocity.copy())
            qdot_log.append(data.qvel[:2].copy())
            mujoco.mj_step(model, data)
            if viewer is not None and step % 5 == 0:
                viewer.sync()
    finally:
        if viewer is not None:
            viewer.close()

    time_log = np.array(time_log)
    xy_log = np.array(xy_log)
    vel_log = np.array(vel_log)
    qdot_log = np.array(qdot_log)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(xy_log[:, 0], xy_log[:, 1], label="MuJoCo tool path")
    axes[0].plot(xy_log[0, 0] + desired[0]*time_log,
                 xy_log[0, 1] + desired[1]*time_log, "--", label="Ideal path")
    axes[0].set(xlabel="X (m)", ylabel="Y (m)", title="Tool path")
    axes[0].axis("equal")
    axes[0].legend()

    for j, name in enumerate(("X", "Y")):
        axes[1].plot(time_log, vel_log[:, j], label=f"Measured {name}")
        axes[1].axhline(desired[j], linestyle="--", label=f"Target {name}")
    axes[1].set(xlabel="Time (s)", ylabel="Tool velocity (m/s)", title="Tool velocity")
    axes[1].legend()

    for j, name in enumerate(("Shoulder", "Elbow")):
        axes[2].plot(time_log, qdot_log[:, j], label=name)
    axes[2].set(xlabel="Time (s)", ylabel="Joint velocity (rad/s)", title="Real joint velocity")
    axes[2].legend()
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = Path(__file__).with_name("gaussian_mujoco_results.png")
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    print("Final tool position [m]:", data.site_xpos[site_id, :2])
    print("Saved graph:", plot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", action="store_true", help="Open MuJoCo live viewer")
    args = parser.parse_args()
    run_simulation(view=args.view)
