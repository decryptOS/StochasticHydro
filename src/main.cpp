#include <iostream>
#include <fstream>
#include <complex>
#include <random>
#include <vector>
#include <span>
#include <cassert>
#include <filesystem>
#include <string>

#include <gsl/gsl_pow_int.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>

#include <fftw3.h>

#include <boost/program_options.hpp>

namespace po = boost::program_options;
namespace fs = std::filesystem;

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

int Nx;
int Ny;
int Nz;

int Nsites;

double dt;

double eta = 1.0;
double temp = 1.0;
double mass_density = 1.0;

// Random number generator

mt19937 mt{random_device{}()};
normal_distribution<> normal_dist(0, 1);

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
                auto kx = 2*M_PI*(nx/static_cast<real_t>(Nx));
                auto ky = 2*M_PI*(ny/static_cast<real_t>(Ny));
                auto kz = 2*M_PI*(nz/static_cast<real_t>(Nz));

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
                // Real-space layout must match what fftw_plan_dft_r2c_3d(Nx,
                // Ny, Nz, ...) assumes: nx slowest, nz fastest.
                int idx = nx*Ny*Nz + ny*Nz + nz;

                int idx_xm = ((nx-1)&(Nx-1))*Ny*Nz + ny*Nz + nz;
                int idx_xp = ((nx+1)&(Nx-1))*Ny*Nz + ny*Nz + nz;

                int idx_ym = nx*Ny*Nz + ((ny-1)&(Ny-1))*Nz + nz;
                int idx_yp = nx*Ny*Nz + ((ny+1)&(Ny-1))*Nz + nz;

                int idx_zm = nx*Ny*Nz + ny*Nz + ( (nz-1)&(Nz-1) );
                int idx_zp = nx*Ny*Nz + ny*Nz + ( (nz+1)&(Nz-1) );

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

                // if(rand()%Nsites==0) {
                // cout<<(Dx_jx+Dy_jy+Dz_jz)<<endl;    
                // }

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
                auto kx = 2*M_PI*(nx/static_cast<real_t>(Nx));
                auto ky = 2*M_PI*(ny/static_cast<real_t>(Ny));
                auto kz = 2*M_PI*(nz/static_cast<real_t>(Nz));

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
    std::string output_folder;

    po::options_description desc("Options");
    desc.add_options()
        ("help", "print help message")
        ("nx", po::value<int>(&Nx)->required(), "Number of lattice sites in x")
        ("ny", po::value<int>(&Ny)->required(), "Number of lattice sites in y")
        ("nz", po::value<int>(&Nz)->required(), "Number of lattice sites in z")
        ("dt", po::value<double>(&dt)->required(), "Time step used in the simulation")
        ("eta", po::value<double>(&eta)->default_value(eta), "shear viscosity")
        ("output-folder", po::value<std::string>(&output_folder)->default_value("out"), "Folder to write output files to");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << desc << endl;
        return 0;
    }

    fs::create_directories(output_folder);

    Nsites = Nx*Ny*Nz;

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

    // Taylor-Green vortex

    // for (int nz = 0; nz < Nz; ++nz) {
    //     for (int ny = 0; ny < Ny; ++ny) {
    //         for (int nx = 0; nx < Nx; ++nx) {
    //             // Real-space layout must match what fftw_plan_dft_r2c_3d(Nx,
    //             // Ny, Nz, ...) assumes: nx slowest, nz fastest.
    //             int idx = nx*Ny*Nz + ny*Nz + nz;

    //             jx[idx] = sin(nx/static_cast<real_t>(Nx)*2*M_PI)*cos(ny/static_cast<real_t>(Ny)*2*M_PI) + 1.0;
    //             jy[idx] = -cos(nx/static_cast<real_t>(Nx)*2*M_PI)*sin(ny/static_cast<real_t>(Ny)*2*M_PI) + 1.0;
    //             jz[idx] = 0;
    //         }
    //     }
    // }

    std::ofstream jx_file(output_folder + "/jx.dat");
    std::ofstream jy_file(output_folder + "/jy.dat");
    std::ofstream jz_file(output_folder + "/jz.dat");

    std::ofstream jpx_re_file(output_folder + "/jpx_re.dat");
    std::ofstream jpy_re_file(output_folder + "/jpy_re.dat");
    std::ofstream jpz_re_file(output_folder + "/jpz_re.dat");

    std::ofstream jpx_im_file(output_folder + "/jpx_im.dat");
    std::ofstream jpy_im_file(output_folder + "/jpy_im.dat");
    std::ofstream jpz_im_file(output_folder + "/jpz_im.dat");

    // Setup Runge-Kutta method for ideal step

    gsl_odeiv2_system sys {
        djdt_ideal,
        nullptr,
        static_cast<size_t>(3*Nsites),
        nullptr
    };

    gsl_odeiv2_step *id_step = gsl_odeiv2_step_alloc(gsl_odeiv2_step_rkf45, 3*Nsites);

    auto jerr_ideal_step_storage = vector<real_t>(3*Nsites, 0);

    // Thermalization

    const int therm_steps = 10000;
    dt = 0.1;

    auto k_min = 2*M_PI/static_cast<real_t>(max({Nx, Ny, Nz}));
    auto eq_time_slow = 1/(eta*k_min*k_min/mass_density);

    auto therm_time = therm_steps*dt;

    cout << "Relaxation time of slowest mode= " << eq_time_slow << " (k_min=" << k_min << ")" << endl;
    cout << "  Numerical thermalization time= " << therm_time << endl;

    fftw_execute(plan_jx_to_jpx);
    fftw_execute(plan_jy_to_jpy);
    fftw_execute(plan_jz_to_jpz);

    for (int i=0; i<therm_steps; ++i) {
        do_diss_step();
    }

    fftw_execute(plan_jpx_to_jx);
    fftw_execute(plan_jpy_to_jy);
    fftw_execute(plan_jpz_to_jz);

    for (int n = 0; n < Nsites; ++n) {
        jx[n] /= Nsites;
        jy[n] /= Nsites;
        jz[n] /= Nsites;
    }

    // Actual simulation in equilibrium state

    const int n_steps = 10000;
    dt = 0.01;

    auto eq_time_fast = 1/(eta*4*M_PI*M_PI/mass_density);

    cout << "                          Nt*dt= " << n_steps*dt << endl;
    cout << "Relaxation time of fastest mode= " << eq_time_fast << endl;
    cout << "                             dt= " << dt << endl;
    
    const int write_every_n_steps = 10;

    double t = 0.0;

    for (int i=0; i<n_steps; ++i) {
        // DISSIPATIVE STEP

        fftw_execute(plan_jx_to_jpx);
        fftw_execute(plan_jy_to_jpy);
        fftw_execute(plan_jz_to_jpz);
        // JX IS DESTROYED

        perform_transverse_projection();

        do_diss_step();

        // OUTPUT, THEN TRANSFORM TO REAL SPACE

        if (i%write_every_n_steps == 0) {

            write_row(jpx_re_file, jpx_im_file, jpx);
            write_row(jpy_re_file, jpy_im_file, jpy);
            write_row(jpz_re_file, jpz_im_file, jpz);

        }

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

        if (i%write_every_n_steps == 0) {

            write_row(jx_file, jx);
            write_row(jy_file, jy);
            write_row(jz_file, jz);

        }

        // IDEAL STEP
        // int status = gsl_odeiv2_step_apply(id_step, t, dt, &j_storage[0], &jerr_ideal_step_storage[0], nullptr, nullptr, &sys);
        // if (status != GSL_SUCCESS) {
        //     fprintf(stderr, "Ideal step failed: %s\n", gsl_strerror(status));
        //     return 1;
        // }

        t += dt;

        // cout << "Finished step " << i << "/" << n_steps << endl;
    }

    gsl_odeiv2_step_free(id_step);

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
