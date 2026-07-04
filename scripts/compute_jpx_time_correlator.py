#!/usr/bin/env python3
"""Compute the time correlator sum_T jpx(t)*jpx(t+T), averaged over all t, for
a single lattice mode (nkx, nky, nkz) from the raw jpx_re.dat/jpx_im.dat
snapshot files of one or more run directories, where each row is one time
step and each column is a lattice mode idx = nkx*Ny*Nzh + nky*Nzh + nkz with
Nzh = Nz//2 + 1 (FFTW's r2c layout: nkx slowest, the halved dimension nkz
fastest).

For each run directory, writes a file <output-dir>/<run_name>.dat with
columns time_diff, Re, Im (and a header recording nx, ny, nz, nkx, nky, nkz,
dt), so that plot_jpx_time_correlator.py can read the correlators back
without recomputing them."""

import argparse
from pathlib import Path

import numpy as np


def load_mode(re_path: Path, im_path: Path, nkx: int, nky: int, nkz: int,
              ny: int, nzh: int) -> np.ndarray:
    idx = nkx * ny * nzh + nky * nzh + nkz
    re = np.loadtxt(re_path, usecols=idx)
    im = np.loadtxt(im_path, usecols=idx)
    return re + 1j * im


def time_correlator(jpx: np.ndarray) -> np.ndarray:
    n = len(jpx)
    correlator = np.empty(len(jpx), dtype=complex)
    for T in range(n):
        correlator[T] = np.mean(np.conj(jpx[:n - T]) * jpx[T:])
    # for n in range(len(jpx)):
    #     for m in range(len(jpx)-n):
    #         correlator[n] += np.conj(jpx[n+m]) * jpx[m]#
    #     correlator[n] /= (len(jpx)-n)
    return correlator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True,
                         help="One or more run directories, each containing its own "
                              "jpx_re.dat, jpx_im.dat (e.g. run1 run2 run3)")
    parser.add_argument("--nx", type=int, required=True, help="Number of lattice sites in x")
    parser.add_argument("--ny", type=int, required=True, help="Number of lattice sites in y")
    parser.add_argument("--nz", type=int, required=True, help="Number of lattice sites in z")
    parser.add_argument("--nkx", type=int, required=True, help="kx mode index")
    parser.add_argument("--nky", type=int, required=True, help="ky mode index")
    parser.add_argument("--nkz", type=int, required=True, help="kz mode index")
    parser.add_argument("--dt", type=float, required=True,
                         help="Time step used in the simulation")
    parser.add_argument("--output-dir", type=Path, default=Path("processed"),
                         help="Directory to write the computed correlators into")
    args = parser.parse_args()

    nzh = args.nz // 2 + 1
    args.output_dir.mkdir(parents=True, exist_ok=True)

    header = (f"nx={args.nx} ny={args.ny} nz={args.nz} "
              f"nkx={args.nkx} nky={args.nky} nkz={args.nkz} dt={args.dt}\n"
              "time_diff Re Im")

    for run_dir in args.run_dirs:
        jpx = load_mode(run_dir / "jpx_re.dat", run_dir / "jpx_im.dat",
                         args.nkx, args.nky, args.nkz, args.ny, nzh)
        correlator = time_correlator(jpx)
        time_diff = np.arange(len(correlator)) * args.dt

        out_path = args.output_dir / f"{run_dir.name}.dat"
        np.savetxt(out_path,
                   np.column_stack([time_diff, correlator.real, correlator.imag]),
                   header=header)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
