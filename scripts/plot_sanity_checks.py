#!/usr/bin/env python3
"""Plot sanity checks from the raw output of one or more run directories:

* total energy sum_x (jx^2 + jy^2 + jz^2)/(2*mass_density) over the lattice
  as a function of time,
* the squared divergence of j, sum_x (Dx_jx + Dy_jy + Dz_jz)^2, with the same
  central differences as djdt_ideal in main.cpp (should be ~0 after the
  transverse projection),
* the total momentum sum_x jx, jy, jz as a function of time (conserved, and
  exactly 0 for a cold start),
* the variance <jpx^2 + jpy^2 + jpz^2>/Nsites of the zero mode (kx=ky=kz=0),
  averaged over runs, compared with the equilibrium expectation
  mass_density*temp*(d-1), where d counts the lattice dimensions with more
  than one site.

Reads jx.dat, jy.dat, jz.dat (real space, one row per output time, column
layout idx = nx*Ny*Nz + ny*Nz + nz as written by main.cpp)."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_j(run_dir: Path, nx: int, ny: int, nz: int) -> np.ndarray:
    """Return j with shape (3, n_times, nx, ny, nz)."""
    components = []
    for name in ("jx.dat", "jy.dat", "jz.dat"):
        data = np.loadtxt(run_dir / name)
        data = np.atleast_2d(data)
        components.append(data.reshape(len(data), nx, ny, nz))
    return np.stack(components)


def divergence_squared(j: np.ndarray) -> np.ndarray:
    """sum_x (Dx_jx + Dy_jy + Dz_jz)^2 per time, with periodic central
    differences as in djdt_ideal."""
    div = np.zeros(j.shape[1:])
    for component, axis in zip(j, (1, 2, 3)):
        div += (np.roll(component, -1, axis=axis)
                - np.roll(component, 1, axis=axis)) / 2.0
    return np.sum(div**2, axis=(1, 2, 3))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                         help="Run directory, containing jx.dat, jy.dat, jz.dat, etc.")
    parser.add_argument("--nx", type=int, required=True, help="Number of lattice sites in x")
    parser.add_argument("--ny", type=int, required=True, help="Number of lattice sites in y")
    parser.add_argument("--nz", type=int, required=True, help="Number of lattice sites in z")
    parser.add_argument("--dt", type=float, required=True,
                         help="Time difference between output rows")
    parser.add_argument("--mass-density", type=float, default=1.0, help="Mass density")
    parser.add_argument("--temp", type=float, default=1.0, help="Temperature")
    parser.add_argument("-o", "--output", type=Path, default=None,
                         help="Save the plot to this file instead of showing it")
    args = parser.parse_args()

    nsites = args.nx * args.ny * args.nz
    ndim = sum(n > 1 for n in (args.nx, args.ny, args.nz))

    fig, ((ax_energy, ax_div), (ax_mom, ax_zero)) = plt.subplots(
        2, 2, figsize=(11, 8))

    time = None

    j = load_j(args.run_dir, args.nx, args.ny, args.nz)
    if time is None:
        time = np.arange(j.shape[1]) * args.dt
    label = args.run_dir.name

    energy = np.sum(j**2, axis=(0, 2, 3, 4)) / (2 * args.mass_density)
    ax_energy.plot(time, energy, alpha=0.7, label=label)

    ax_div.plot(time, divergence_squared(j), alpha=0.7, label=label)

    # Total momentum = zero mode of jp: jp(k=0) = sum_x j(x).
    total_momentum = np.sum(j, axis=(2, 3, 4))
    for component, name in zip(total_momentum, ("x", "y", "z")):
        ax_mom.plot(time, component, alpha=0.7,
                    label=f"{label} {name}")

    ax_energy.set_ylabel(r"$\sum_x \, (j_x^2+j_y^2+j_z^2)/(2\rho)$")
    theory = nsites * args.temp * (ndim - 1)
    ax_energy.axhline(theory, color="black", linestyle="--",
                   label=rf"$(d-1)N_s T = {theory:g}$")
    ax_energy.set_title("Total energy")

    ax_div.set_ylabel(r"$\sum_x \, (D_x j_x + D_y j_y + D_z j_z)^2$")
    ax_div.set_title("Squared divergence of j")

    ax_mom.set_ylabel(r"$\sum_x \, j_i$")
    ax_mom.set_title("Total momentum")

    for ax in (ax_energy, ax_div, ax_mom, ax_zero):
        ax.set_xlabel("time t")
        ax.legend(fontsize="x-small", ncol=2)

    fig.tight_layout()

    if args.output is not None:
        fig.savefig(args.output)
    else:
        plt.show()


if __name__ == "__main__":
    main()
