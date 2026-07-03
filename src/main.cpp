#include <iostream>
#include <fstream>
#include <complex>
#include <random>
#include <vector>
#include <span>

#include <gsl/gsl_pow_int.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>

#include <fftw3.h>

using namespace std;

using real_t = double;
using complex_t = complex<real_t>;

// Momentum density

vector<real_t> j_storage;

span<real_t> jx;
span<real_t> jy;
span<real_t> jz;

vector<complex_t> jp_storage;

span<complex_t> jpx;
span<complex_t> jpy;
span<complex_t> jpz;

fftw_plan plan_jx_to_jpx;
fftw_plan plan_jy_to_jpy;
fftw_plan plan_jz_to_jpz;

fftw_plan plan_jpx_to_jx;
fftw_plan plan_jpy_to_jy;
fftw_plan plan_jpz_to_jz;

// Langevin noise

vector<real_t> lx;
vector<real_t> ly;
vector<real_t> lz;

vector<complex_t> lpx;
vector<complex_t> lpy;
vector<complex_t> lpz;

fftw_plan plan_lx_to_lpx;
fftw_plan plan_ly_to_lpy;
fftw_plan plan_lz_to_lpz;

fftw_plan plan_lpx_to_lx;
fftw_plan plan_lpy_to_ly;
fftw_plan plan_lpz_to_lz;

int Nx = 16;
int Ny = 16;
int Nz = 16;

int Nsites = Nx*Ny*Nz;

double dt = 0.1;
double eta = 1.0;
double temp = 2.0;
double mass_density = 1.0;

// Random number generator

mt19937 mt{random_device{}()};
normal_distribution<> normal_dist(0, 1);

void print_ji(const vector<real_t>& ji)
{
    for (int nz = 0; nz < Nz; ++nz) {
        for (int ny = 0; ny < Ny; ++ny) {
            for (int nx = 0; nx < Nx; ++nx) {
                int idx = nz*Nx*Ny + ny*Nx + nx;

                cout << ji[idx] << " ";
            }
            cout << endl;
        }
    }
}

void print_jpi(const vector<complex_t>& jpi)
{
    int Nzh = Nz/2+1;

    for (int nz = 0; nz < Nzh; ++nz) {
        for (int ny = 0; ny < Ny; ++ny) {
            for (int nx = 0; nx < Nx; ++nx) {
                // FFTW r2c layout: nx slowest, halved dim (nz) fastest
                int idx = nx*Ny*Nzh + ny*Nzh + nz;

                cout << jpi[idx] << " ";
            }
            cout << endl;
        }
    }
}

template <typename T>
void write_row(ofstream& out, const span<T>& values)
{
    for (const auto& value : values) {
        out << value << ' ';
    }
    out << '\n';
}

void write_row(ofstream& out_re, ofstream& out_im, const span<complex_t>& values)
{
    for (const auto& value : values) {
        out_re << value.real() << ' ';
    }
    out_re << '\n';
    
    for (const auto& value : values) {
        out_im << value.imag() << ' ';
    }
    out_im << '\n';
}

// void write_lattice_avg(const string& path, const string& header, const vector<real_t>& avg)
// {
//     ofstream out(path);
//     out << "# x y z " << header << '\n';

//     for (int nz = 0; nz < Nz; ++nz) {
//         for (int ny = 0; ny < Ny; ++ny) {
//             for (int nx = 0; nx < Nx; ++nx) {
//                 int idx = nz*Nx*Ny + ny*Nx + nx;

//                 out << nx << ' ' << ny << ' ' << nz << ' ' << avg[idx] << '\n';
//             }
//         }
//     }
// }

// void write_kspace_avg(const std::string& path, const std::string& header, const vector<complex_t>& avg)
// {
//     ofstream out(path);
//     out << "# kx ky kz " << header << '\n';

//     for (int nz = 0; nz < Nz/2+1; ++nz) {
//         for (int ny = 0; ny < Ny; ++ny) {
//             for (int nx = 0; nx < Nx; ++nx) {
//                 int idx = nz*Nx*Ny + ny*Nx + nx;

//                 real_t kx = 2*M_PI*(nx/static_cast<real_t>(Nx)-0.5);
//                 real_t ky = 2*M_PI*(ny/static_cast<real_t>(Ny)-0.5);
//                 real_t kz = 2*M_PI*(nz/static_cast<real_t>(Nz)-0.5);

//                 out << kx << ' ' << ky << ' ' << kz << ' ' << avg[idx].real() << ' ' << avg[idx].imag() << '\n';
//             }
//         }
//     }
// }

void do_diss_step()
{
    for (int i=0; i<Nsites; ++i) {
        lx[i] = normal_dist(mt);
        ly[i] = normal_dist(mt);
        lz[i] = normal_dist(mt);
    }

    fftw_execute(plan_lx_to_lpx);
    fftw_execute(plan_ly_to_lpy);
    fftw_execute(plan_lz_to_lpz);

    int Nzh = Nz/2+1;

    for (int nz = 0; nz < Nzh; ++nz) {
        for (int ny = 0; ny < Ny; ++ny) {
            for (int nx = 0; nx < Nx; ++nx) {
                auto kx = 2*M_PI*(nx/static_cast<real_t>(Nx)-0.5);
                auto ky = 2*M_PI*(ny/static_cast<real_t>(Ny)-0.5);
                auto kz = 2*M_PI*(nz/static_cast<real_t>(Nz)-0.5);

                // lattice spacing a=1
                auto ktx = sin(kx);
                auto kty = sin(ky);
                auto ktz = sin(kz);

                auto damp = eta/mass_density*(ktx*ktx+kty*kty+ktz*ktz);

                // FFTW r2c layout: nx slowest, halved dim (nz) fastest
                int idx = nx*Ny*Nzh + ny*Nzh + nz;

                // Fluctuation-dissipation theorem: the exact OU update of a
                // mode damped at rate `damp` needs noise variance
                // (eta*T/damp)*(1-exp(-2*damp*dt)), not just temp*(1-exp(...)).
                auto coeff = sqrt(mass_density*temp*(1-exp(-2*damp*dt)));

                jpx[idx] = exp(-damp*dt)*jpx[idx] + coeff*lpx[idx];
                jpy[idx] = exp(-damp*dt)*jpy[idx] + coeff*lpy[idx];
                jpz[idx] = exp(-damp*dt)*jpz[idx] + coeff*lpz[idx];
            }
        }
    }
}

int djdt_ideal(double t, const double y[], double dydt[], void *params)
{
    auto *jx = &y[0*Nsites];
    auto *jy = &y[1*Nsites];
    auto *jz = &y[2*Nsites];

    // printf("%d %d\n", -1%8, (-1)&(8-1));
    // exit(0);

    for (int nz = 0; nz < Nz; ++nz) {
        for (int ny = 0; ny < Ny; ++ny) {
            for (int nx = 0; nx < Nx; ++nx) {
                int idx = nz*Nx*Ny + ny*Nx + nx;
                
                int idx_xm = nz*Nx*Ny + ny*Nx + ( (nx-1)&(Nx-1) );
                int idx_xp = nz*Nx*Ny + ny*Nx + ( (nx+1)&(Nx-1) );

                int idx_ym = nz*Nx*Ny + ((ny-1)&(Ny-1))*Nx + nx;
                int idx_yp = nz*Nx*Ny + ((ny+1)&(Ny-1))*Nx + nx;

                int idx_zm = ((nz-1)&(Nz-1))*Nx*Ny + ny*Nx + nx;
                int idx_zp = ((nz+1)&(Nz-1))*Nx*Ny + ny*Nx + nx;

                // no static regulator:
                auto vx = jx[idx]/mass_density;
                auto vy = jy[idx]/mass_density;
                auto vz = jz[idx]/mass_density;

                // central differences
                auto Dx_jx = (jx[idx_xp] - jx[idx_xm])/2.0;
                auto Dy_jx = (jx[idx_yp] - jx[idx_ym])/2.0;
                auto Dz_jx = (jx[idx_zp] - jx[idx_zm])/2.0;

                auto Dx_jy = (jy[idx_xp] - jy[idx_xm])/2.0;
                auto Dy_jy = (jy[idx_yp] - jy[idx_ym])/2.0;
                auto Dz_jy = (jy[idx_zp] - jy[idx_zm])/2.0;

                auto Dx_jz = (jz[idx_xp] - jz[idx_xm])/2.0;
                auto Dy_jz = (jz[idx_yp] - jz[idx_ym])/2.0;
                auto Dz_jz = (jz[idx_zp] - jz[idx_zm])/2.0;

                auto djxdt = -vx*( Dx_jx - Dx_jx ) -vy*( Dy_jx - Dx_jy ) -vz*( Dz_jx - Dx_jz );
                auto djydt = -vx*( Dx_jy - Dy_jx ) -vy*( Dy_jy - Dy_jy ) -vz*( Dz_jy - Dy_jz );
                auto djzdt = -vx*( Dx_jz - Dz_jx ) -vy*( Dy_jz - Dz_jy ) -vz*( Dz_jz - Dz_jz );

                dydt[idx+0*Nsites] = djxdt;
                dydt[idx+1*Nsites] = djydt;
                dydt[idx+2*Nsites] = djzdt;
            }
        }
    }

    return GSL_SUCCESS;
}

void perform_transverse_projection()
{
    int Nzh = Nz/2+1;

    for (int nz = 0; nz < Nzh; ++nz) {
        for (int ny = 0; ny < Ny; ++ny) {
            for (int nx = 0; nx < Nx; ++nx) {
                auto kx = 2*M_PI*(nx/static_cast<real_t>(Nx)-0.5);
                auto ky = 2*M_PI*(ny/static_cast<real_t>(Ny)-0.5);
                auto kz = 2*M_PI*(nz/static_cast<real_t>(Nz)-0.5);

                // lattice spacing a=1
                auto ktx = sin(kx);
                auto kty = sin(ky);
                auto ktz = sin(kz);

                auto kt2 = ktx*ktx + kty*kty + ktz*ktz;

                // k=0 and the Nyquist-degenerate points (sin(k_i)=0 in all
                // three directions) have no well-defined transverse direction.
                [[unlikely]]
                if (kt2 < 1e-14) continue;

                // FFTW r2c layout: nx slowest, halved dim (nz) fastest
                int idx = nx*Ny*Nzh + ny*Nzh + nz;

                auto kt_dot_jp = ktx*jpx[idx] + kty*jpy[idx] + ktz*jpz[idx];

                jpx[idx] = jpx[idx] - ktx*kt_dot_jp/kt2;
                jpy[idx] = jpy[idx] - kty*kt_dot_jp/kt2;
                jpz[idx] = jpz[idx] - ktz*kt_dot_jp/kt2;
            }
        }
    }
}

int main(int argc, const char *argv[])
{
    cout << "Nx, Ny, Nz MUST BE POWERS OF TWO" << endl;

    //WARNING:
//The fftw_plan_dft_c2r function destroys (overwrites) its input array (out) during execution to save memory.

    // Momentum density

    j_storage = vector<real_t>(3*Nsites, 0);

    jx = span<real_t>(j_storage.data()+0*Nsites, Nsites);
    jy = span<real_t>(j_storage.data()+1*Nsites, Nsites);
    jz = span<real_t>(j_storage.data()+2*Nsites, Nsites);
    
    auto Np = Nx*Ny * (Nz/2+1);

    jp_storage = vector<complex_t>(3*Np, complex_t{0, 0});

    jpx = span<complex_t>(jp_storage.data()+0*Np, Np); // The last dimension is cut in half + 1
    jpy = span<complex_t>(jp_storage.data()+1*Np, Np); // The last dimension is cut in half + 1
    jpz = span<complex_t>(jp_storage.data()+2*Np, Np); // The last dimension is cut in half + 1

    plan_jx_to_jpx = fftw_plan_dft_r2c_3d(Nx, Ny, Nz, &jx[0], reinterpret_cast<fftw_complex *>(&jpx[0]), FFTW_ESTIMATE);
    plan_jy_to_jpy = fftw_plan_dft_r2c_3d(Nx, Ny, Nz, &jy[0], reinterpret_cast<fftw_complex *>(&jpy[0]), FFTW_ESTIMATE);
    plan_jz_to_jpz = fftw_plan_dft_r2c_3d(Nx, Ny, Nz, &jz[0], reinterpret_cast<fftw_complex *>(&jpz[0]), FFTW_ESTIMATE);
    
    plan_jpx_to_jx = fftw_plan_dft_c2r_3d(Nx, Ny, Nz, reinterpret_cast<fftw_complex *>(&jpx[0]), &jx[0], FFTW_ESTIMATE);
    plan_jpy_to_jy = fftw_plan_dft_c2r_3d(Nx, Ny, Nz, reinterpret_cast<fftw_complex *>(&jpy[0]), &jy[0], FFTW_ESTIMATE);
    plan_jpz_to_jz = fftw_plan_dft_c2r_3d(Nx, Ny, Nz, reinterpret_cast<fftw_complex *>(&jpz[0]), &jz[0], FFTW_ESTIMATE);

    // Noise

    lx = std::vector<real_t>(Nsites, 0);
    ly = std::vector<real_t>(Nsites, 0);
    lz = std::vector<real_t>(Nsites, 0);
    
    lpx = std::vector<complex_t>(Np, complex_t{0, 0}); // The last dimension is cut in half + 1
    lpy = std::vector<complex_t>(Np, complex_t{0, 0}); // The last dimension is cut in half + 1
    lpz = std::vector<complex_t>(Np, complex_t{0, 0}); // The last dimension is cut in half + 1

    plan_lx_to_lpx = fftw_plan_dft_r2c_3d(Nx, Ny, Nz, &lx[0], reinterpret_cast<fftw_complex *>(&lpx[0]), FFTW_ESTIMATE);
    plan_ly_to_lpy = fftw_plan_dft_r2c_3d(Nx, Ny, Nz, &ly[0], reinterpret_cast<fftw_complex *>(&lpy[0]), FFTW_ESTIMATE);
    plan_lz_to_lpz = fftw_plan_dft_r2c_3d(Nx, Ny, Nz, &lz[0], reinterpret_cast<fftw_complex *>(&lpz[0]), FFTW_ESTIMATE);
    
    plan_lpx_to_lx = fftw_plan_dft_c2r_3d(Nx, Ny, Nz, reinterpret_cast<fftw_complex *>(&lpx[0]), &lx[0], FFTW_ESTIMATE);
    plan_lpy_to_ly = fftw_plan_dft_c2r_3d(Nx, Ny, Nz, reinterpret_cast<fftw_complex *>(&lpy[0]), &ly[0], FFTW_ESTIMATE);
    plan_lpz_to_lz = fftw_plan_dft_c2r_3d(Nx, Ny, Nz, reinterpret_cast<fftw_complex *>(&lpz[0]), &lz[0], FFTW_ESTIMATE);

    // Cold start

    for (int n = 0; n < Nsites; ++n) {
        jx[n] = 0;
        jy[n] = 0;
        jz[n] = 0;
    }

    // Initialize the 2D input array with some dummy data
    // Row-major order: index = row * Width + col
    // for (int r = 0; r < H; r++) {
    //     for (int c = 0; c < W; c++) {
    //         jx[r * W + c] = (double)(r + c); 
    //     }
    // }

    // Execute the 2D FFT

    std::ofstream jx_file("out/jx.dat");
    std::ofstream jy_file("out/jy.dat");
    std::ofstream jz_file("out/jz.dat");

    std::ofstream jpx_re_file("out/jpx_re.dat");
    std::ofstream jpy_re_file("out/jpy_re.dat");
    std::ofstream jpz_re_file("out/jpz_re.dat");

    std::ofstream jpx_im_file("out/jpx_im.dat");
    std::ofstream jpy_im_file("out/jpy_im.dat");
    std::ofstream jpz_im_file("out/jpz_im.dat");

    // Setup Runge-Kutta method for ideal step

    gsl_odeiv2_system sys {
        djdt_ideal,
        nullptr,
        static_cast<size_t>(3*Nsites),
        nullptr
    };

    gsl_odeiv2_step *id_step = gsl_odeiv2_step_alloc(gsl_odeiv2_step_rk4, 3*Nsites);

    auto jerr_ideal_step_storage = vector<real_t>(3*Nsites, 0);

    const int n_steps = 1000;

    double t = 0.0;

    for (int i=0; i<n_steps; ++i) {
        // DISSIPATIVE STEP

        fftw_execute(plan_jx_to_jpx);
        fftw_execute(plan_jy_to_jpy);
        fftw_execute(plan_jz_to_jpz);
        // JX IS DESTROYED

        // for (int n = 0; n < Nsites; ++n) {
        //     jpx[n] /= sqrt(Nsites);
        //     jpy[n] /= sqrt(Nsites);
        //     jpz[n] /= sqrt(Nsites);
        // }

        do_diss_step();

        perform_transverse_projection();

        // OUTPUT, THEN TRANSFORM TO REAL SPACE

        write_row(jpx_re_file, jpx_im_file, jpx);
        write_row(jpy_re_file, jpy_im_file, jpy);
        write_row(jpz_re_file, jpz_im_file, jpz);

        // HERE WE SHOULD READ OUT JX, JY, JZ

        fftw_execute(plan_jpx_to_jx);
        fftw_execute(plan_jpy_to_jy);
        fftw_execute(plan_jpz_to_jz);
        // JP IS DESTROYED

        for (int n = 0; n < Nsites; ++n) {
            jx[n] /= Nsites;
            jy[n] /= Nsites;
            jz[n] /= Nsites;
        }

        // OUTPUT, THEN IDEAL STEP

        write_row(jx_file, jx);
        write_row(jy_file, jy);
        write_row(jz_file, jz);

        // IDEAL STEP
        /*int status = gsl_odeiv2_step_apply(id_step, t, dt, &j_storage[0], &jerr_ideal_step_storage[0], nullptr, nullptr, &sys);
        if (status != GSL_SUCCESS) {
            fprintf(stderr, "Ideal step failed: %s\n", gsl_strerror(status));
            return 1;
        }*/

        t += dt;

        cout << "Finished step " << i << "/" << n_steps << endl;
    }

    gsl_odeiv2_step_free(id_step);

    // Print the complex output array (Size: H x W_complex)
    // printf("2D FFT Output (%d x %d complex grid):\n", H, W_complex);
    // for (int r = 0; r < H; r++) {
    //     for (int c = 0; c < W_complex; c++) {
    //         int idx = r * W_complex + c;
    //         printf("out[%d][%d] = %6.2f + %6.2fi\t", r, c, out[idx][0], out[idx][1]);
    //     }
    //     printf("\n");
    // }

    fftw_destroy_plan(plan_jx_to_jpx);
    fftw_destroy_plan(plan_jy_to_jpy);
    fftw_destroy_plan(plan_jz_to_jpz);

    fftw_destroy_plan(plan_jpx_to_jx);
    fftw_destroy_plan(plan_jpy_to_jy);
    fftw_destroy_plan(plan_jpz_to_jz);

    fftw_destroy_plan(plan_lx_to_lpx);
    fftw_destroy_plan(plan_ly_to_lpy);
    fftw_destroy_plan(plan_lz_to_lpz);

    fftw_destroy_plan(plan_lpx_to_lx);
    fftw_destroy_plan(plan_lpy_to_ly);
    fftw_destroy_plan(plan_lpz_to_lz);

    return 0;
}
