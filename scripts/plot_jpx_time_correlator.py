#!/usr/bin/env python3
"""Plot the time correlator sum_T jpx(t)*jpx(t+T), averaged over all t and
plotted as a function of the time difference T, for a single lattice mode
(nkx, nky, nkz) from the raw jpx_re.dat/jpx_im.dat snapshot files, where each
row is one time step and each column is a lattice mode
idx = nkx*Ny*Nzh + nky*Nzh + nkz with Nzh = Nz//2 + 1 (FFTW's r2c layout: nkx
slowest, the halved dimension nkz fastest)."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np



def load_mode(re_path: Path, im_path: Path, nkx: int, nky: int, nkz: int,
              ny: int, nzh: int) -> np.ndarray:
    idx = nkx * ny * nzh + nky * nzh + nkz
    re = np.loadtxt(re_path, usecols=idx)
    im = np.loadtxt(im_path, usecols=idx)
    return re + 1j * im


def time_correlator(jpx: np.ndarray) -> np.ndarray:
    n = len(jpx)
    correlator = np.empty(n, dtype=complex)

    for n in range(len(jpx)):
        #n: time diff
        
        for m in range(len(jpx)-n):
            correlator[n] += np.conj(jpx[m+n])*jpx[m]
        
        correlator[n]/=(len(jpx)-n)

    # for T in range(n):
    #     correlator[T] = np.mean(np.conj(jpx[:n - T]) * jpx[T:])
    return correlator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("out"),
                         help="Directory containing jpx_re.dat, jpx_im.dat")
    parser.add_argument("--nx", type=int, required=True, help="Number of lattice sites in x")
    parser.add_argument("--ny", type=int, required=True, help="Number of lattice sites in y")
    parser.add_argument("--nz", type=int, required=True, help="Number of lattice sites in z")
    parser.add_argument("--nkx", type=int, required=True, help="kx mode index")
    parser.add_argument("--nky", type=int, required=True, help="ky mode index")
    parser.add_argument("--nkz", type=int, required=True, help="kz mode index")
    parser.add_argument("--dt", type=float, required=True,
                         help="Time step used in the simulation")
    parser.add_argument("--eta", type=float, default=1.0, help="shear viscosity")
    parser.add_argument("--mass-density", type=float, default=1.0, help="mass density")
    parser.add_argument("-o", "--output", type=Path, default=None,
                         help="Save the plot to this file instead of showing it")
    args = parser.parse_args()

    nzh = args.nz // 2 + 1
    jpx = load_mode(args.data_dir / "jpx_re.dat", args.data_dir / "jpx_im.dat",
                     args.nkx, args.nky, args.nkz, args.ny, nzh)

    correlator = time_correlator(jpx)
    time_diff = np.arange(len(correlator)) * args.dt

    kx = np.sin(2 * np.pi * args.nkx / args.nx)
    ky = np.sin(2 * np.pi * args.nky / args.ny)
    kz = np.sin(2 * np.pi * args.nkz / args.nz)
    damp = args.eta / args.mass_density * (kx**2 + ky**2 + kz**2)
    theory = correlator[0].real * np.exp(-damp * time_diff)

    fig, ax = plt.subplots()
    ax.plot(time_diff, correlator.real, label="Re")
    ax.plot(time_diff, correlator.imag, label="Im")
    ax.plot(time_diff, theory, "--", color="black",
            label=r"$\exp(-\eta/\rho \, (k_x^2+k_y^2+k_z^2)\, t)$")
    ax.set_xlabel("time difference T")
    ax.set_ylabel(f"<jpx(t) jpx(t+T)>_t  (nkx={args.nkx}, nky={args.nky}, nkz={args.nkz})")
    ax.legend()
    fig.tight_layout()

    if args.output is not None:
        fig.savefig(args.output)
    else:
        plt.show()


if __name__ == "__main__":
    main()
