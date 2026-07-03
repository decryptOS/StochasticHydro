#!/usr/bin/env python3
"""Animate the time evolution of jx^2+jy^2+jz^2 at z=0 as a 2D contour plot,
from jx.dat, jy.dat and jz.dat (row=time)."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import numpy as np


def load(path: Path) -> np.ndarray:
    return np.loadtxt(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",     type=Path, default=Path("out"), help="Directory containing jx.dat, jy.dat, jz.dat")
    parser.add_argument("--nx",           type=int, help="Number of lattice sites in x", required=True)
    parser.add_argument("--ny",           type=int, help="Number of lattice sites in y", required=True)
    parser.add_argument("--nz",           type=int, help="Number of lattice sites in z", required=True)
    parser.add_argument("--dt",           type=float, help="Time step used in the simulation", required=True)
    parser.add_argument("--interval",     type=float, default=50, help="Delay between animation frames in milliseconds")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Save the animation to this file instead of showing it")
    args = parser.parse_args()

    jx = load(args.data_dir / "jx.dat")
    jy = load(args.data_dir / "jy.dat")
    jz = load(args.data_dir / "jz.dat")

    j2 = jx #**2 + jy**2 + jz**2

    # idx = x*Ny*Nz + y*Nz + z (matches fftw_plan_dft_r2c_3d(Nx,Ny,Nz,...):
    # x slowest, z fastest), so the z=0 plane is the strided slice [:, :, 0],
    # transposed to (Ny, Nx) for contourf's expected (row=y, col=x) shape.
    n_steps = j2.shape[0]
    j2_z0 = j2.reshape(n_steps, args.nx, args.ny, args.nz)[:, :, :, 0].transpose(0, 2, 1)

    time = np.arange(n_steps) * args.dt

    vmin, vmax = j2_z0.min(), j2_z0.max()

    fig, ax = plt.subplots()
    x = np.arange(args.nx)
    y = np.arange(args.ny)

    mesh = ax.pcolormesh(x, y, j2_z0[0], vmin=vmin, vmax=vmax, shading="nearest")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("jx^2 + jy^2 + jz^2")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"z=0, t={time[0]:.4f}")

    def update(frame):
        frame = int(round(frame))
        mesh.set_array(j2_z0[frame])
        ax.set_title(f"z=0, t={time[frame]:.4f}")

    if args.output is not None:
        anim = animation.FuncAnimation(fig, update, frames=n_steps,
                                        interval=args.interval, blit=False)
        anim.save(args.output)
    else:
        fig.subplots_adjust(bottom=0.2)
        ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.03])
        slider = Slider(ax_slider, "frame", 0, n_steps - 1, valinit=0, valstep=1)
        slider.on_changed(update)
        plt.show()


if __name__ == "__main__":
    main()
