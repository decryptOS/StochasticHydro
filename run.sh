
./sim --dt 0.5 --output-every-n-steps 10 --sim-time 2000 --nx 32 --ny 32 --nz 32 --eta 0.01 --eta-uv-cutoff 3.0 --no-ideal-step --seed 1
./sim --dt 0.5 --output-every-n-steps 10 --sim-time 2000 --nx 32 --ny 32 --nz 32 --eta 0.01 --eta-uv-cutoff 3.0 --no-ideal-step --seed 2
./sim --dt 0.5 --output-every-n-steps 10 --sim-time 2000 --nx 32 --ny 32 --nz 32 --eta 0.01 --eta-uv-cutoff 3.0 --no-ideal-step --seed 3
./sim --dt 0.5 --output-every-n-steps 10 --sim-time 2000 --nx 32 --ny 32 --nz 32 --eta 0.01 --eta-uv-cutoff 3.0 --no-ideal-step --seed 4
./sim --dt 0.5 --output-every-n-steps 10 --sim-time 2000 --nx 32 --ny 32 --nz 32 --eta 0.01 --eta-uv-cutoff 3.0 --no-ideal-step --seed 5

python scripts/compute_jpx_time_correlator.py --nkx 0 --nky 1 --nkz 0 --run-dir sim-Nx32Ny32Nz32dt5eta0.01Lam3seed1x
python scripts/compute_jpx_time_correlator.py --nkx 0 --nky 1 --nkz 0 --run-dir sim-Nx32Ny32Nz32dt5eta0.01Lam3seed2x
python scripts/compute_jpx_time_correlator.py --nkx 0 --nky 1 --nkz 0 --run-dir sim-Nx32Ny32Nz32dt5eta0.01Lam3seed3x
python scripts/compute_jpx_time_correlator.py --nkx 0 --nky 1 --nkz 0 --run-dir sim-Nx32Ny32Nz32dt5eta0.01Lam3seed4x
python scripts/compute_jpx_time_correlator.py --nkx 0 --nky 1 --nkz 0 --run-dir sim-Nx32Ny32Nz32dt5eta0.01Lam3seed5x

python scripts/plot_jpx_time_correlator.py obs-Nx32Ny32Nz32dt5eta0.01Lam3x --ylim -0.2 1.2
