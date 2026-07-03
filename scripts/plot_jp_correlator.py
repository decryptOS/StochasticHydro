#!/usr/bin/env python3
"""Plot the row-averaged correlator |jpx|^2+|jpy|^2+|jpz|^2 at (kx, ky=0, kz=0)
from the raw jpx_re.dat/jpx_im.dat, jpy_re.dat/jpy_im.dat and jpz_re.dat/jpz_im.dat
snapshot files, where each row is one time step and each column is a lattice mode
idx = nx*Ny*(Nz/2+1) + ny*(Nz/2+1) + nz (FFTW's r2c layout: nx slowest, the
halved dimension nz fastest)."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_row_averaged(re_path: Path, im_path: Path) -> complex:
    re2 = (np.loadtxt(re_path)**2).mean(axis=0)
    im2 = (np.loadtxt(im_path)**2).mean(axis=0)
    return re2 + im2


def ky0_kz0_slice(jp: np.ndarray, Nx: int, Ny: int, Nz: int) -> np.ndarray:
    # idx = nx*Ny*Nzh + ny*Nzh + nz with Nzh = Nz/2+1, so ky=0 (ny=0) and
    # kz=0 (nz=0) is the strided slice jp[0], jp[Ny*Nzh], jp[2*Ny*Nzh], ...
    Nzh = Nz // 2 + 1
    return jp.reshape(Nx, Ny, Nzh)[:, 0, 0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("out"),
                         help="Directory containing jpx_re.dat, jpx_im.dat, jpy_re.dat, jpy_im.dat, jpz_re.dat, jpz_im.dat")
    parser.add_argument("--nx", type=int, default=16, help="Number of lattice sites in x")
    parser.add_argument("--ny", type=int, default=16, help="Number of lattice sites in y")
    parser.add_argument("--nz", type=int, default=16, help="Number of lattice sites in z")
    parser.add_argument("-o", "--output", type=Path, default=None,
                         help="Save the plot to this file instead of showing it")
    args = parser.parse_args()

    jpx = load_row_averaged(args.data_dir / "jpx_re.dat", args.data_dir / "jpx_im.dat")
    jpy = load_row_averaged(args.data_dir / "jpy_re.dat", args.data_dir / "jpy_im.dat")
    jpz = load_row_averaged(args.data_dir / "jpz_re.dat", args.data_dir / "jpz_im.dat")

    jpx2 = ky0_kz0_slice(jpx, args.nx, args.ny, args.nz)
    jpy2 = ky0_kz0_slice(jpy, args.nx, args.ny, args.nz)
    jpz2 = ky0_kz0_slice(jpz, args.nx, args.ny, args.nz)

    kx = 2 * np.pi * np.arange(args.nx) / args.nx
    correlator = (jpx2 + jpy2 + jpz2)/(args.nx*args.ny*args.nz)

    fig, ax = plt.subplots()
    ax.plot(np.arange(args.nx), correlator, marker="o")
    ax.set_xlabel("kx")
    ax.set_ylabel("|jpx|^2 + |jpy|^2 + |jpz|^2  (ky=0, kz=0)")
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    if args.output is not None:
        fig.savefig(args.output)
    else:
        plt.show()


if __name__ == "__main__":
    main()
