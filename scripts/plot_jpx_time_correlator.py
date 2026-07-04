#!/usr/bin/env python3
"""Plot the time correlator sum_T jpx(t)*jpx(t+T) as a function of the time
difference t, averaged over several runs with an error band from their
run-to-run variation.

Reads the per-run correlator files written by compute_jpx_time_correlator.py:
one <run>.dat file per run in --input-dir, with columns time_diff, Re, Im and
a header recording nx, ny, nz, nkx, nky, nkz, dt for the given mode."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_header(path: Path) -> dict:
    with open(path) as f:
        first_line = f.readline().lstrip("#").strip()
    return {key: float(value) for key, value in
            (item.split("=") for item in first_line.split())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("processed"),
                         help="Directory containing the *.dat correlator files "
                              "written by compute_jpx_time_correlator.py")
    parser.add_argument("--eta", type=float, default=1.0, help="Shear viscosity")
    parser.add_argument("--mass-density", type=float, default=1.0, help="Mass density")
    parser.add_argument("--temp", type=float, required=True, help="Temperature")
    parser.add_argument("-o", "--output", type=Path, default=None,
                         help="Save the plot to this file instead of showing it")
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob("*.dat"))
    if not paths:
        raise SystemExit(f"No correlator files found in {args.input_dir}")

    meta = read_header(paths[0])

    correlators = []
    time_diff = None
    for path in paths:
        t, re, im = np.loadtxt(path, unpack=True)
        if time_diff is None:
            time_diff = t
        correlators.append(re + 1j * im)

    n_min = min(len(c) for c in correlators)
    correlators = np.stack([c[:n_min] for c in correlators])
    time_diff = time_diff[:n_min]

    mean = correlators.mean(axis=0)
    n_runs = len(correlators)
    if n_runs > 1:
        # Standard error of the mean, not the run-to-run standard deviation:
        # the latter reflects the spread from the Langevin noise itself and
        # does not shrink as more runs are added, while the SEM quantifies
        # the uncertainty on the estimated mean correlator.
        sem_re = correlators.real.std(axis=0, ddof=1) / np.sqrt(n_runs)
        sem_im = correlators.imag.std(axis=0, ddof=1) / np.sqrt(n_runs)
    else:
        sem_re = np.zeros(n_min)
        sem_im = np.zeros(n_min)

    kx = np.sin(2 * np.pi * meta["nkx"] / meta["nx"])
    ky = np.sin(2 * np.pi * meta["nky"] / meta["ny"])
    kz = np.sin(2 * np.pi * meta["nkz"] / meta["nz"])
    damp = args.eta / args.mass_density * (kx**2 + ky**2 + kz**2)

    nsites = meta["nx"] * meta["ny"] * meta["nz"]
    theory = nsites * args.temp * args.mass_density * np.exp(-damp * time_diff)

    fig, ax = plt.subplots()
    for part, values, errors in (("Re", mean.real, sem_re), ("Im", mean.imag, sem_im)):
        line, = ax.plot(time_diff, values, label=part)
        ax.fill_between(time_diff, values - errors, values + errors,
                         color=line.get_color(), alpha=0.3)
    ax.plot(time_diff, theory, "--", color="black",
            label=r"$\exp(-\eta/\rho \, (\sin^2 k_x+\sin^2 k_y+\sin^2 k_z)\, t)$")
    ax.set_xlabel("time difference t")
    ax.set_ylabel("sum_T  <jpy*(T+t) jpy(T)>  "
                  f"(nkx={meta['nkx']:.0f}, nky={meta['nky']:.0f}, nkz={meta['nkz']:.0f})")
    ax.set_ylim(-0.2*theory[0], 1.2*theory[0])
    ax.legend()
    fig.tight_layout()

    if args.output is not None:
        fig.savefig(args.output)
    else:
        plt.show()


if __name__ == "__main__":
    main()
