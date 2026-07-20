#include <iostream>
#include <fstream>
#include <complex>
#include <random>
#include <vector>
#include <span>
#include <cassert>
#include <filesystem>
#include <string>
#include <sstream>
#include <charconv>
#include <chrono>

#include <gsl/gsl_pow_int.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>

#include <fftw3.h>

#include <omp.h>

#include <boost/program_options.hpp>
#include <boost/assert.hpp>

namespace po = boost::program_options;
namespace fs = std::filesystem;

using namespace std;

using real_t = double;
using complex_t = complex<real_t>;

constexpr size_t NsitesOMPThreshold = 256*256;  // minimum number of lattice sites to use OpenMP parallelization

// Momentum density

vector<real_t> j_storage;

span<real_t> jx;
span<real_t> jy;
span<real_t> jz;

vector<complex_t> jp_storage;

span<complex_t> jpx;
span<complex_t> jpy;
span<complex_t> jpz;

// All transforms are batched (howmany=3): one plan transforms the x, y and z
// components in a single threaded fftw_execute call.

fftw_plan plan_j_to_jp;
fftw_plan plan_jp_to_j;

// Time derivative of momentum density

vector<real_t> djdt_storage;

span<real_t> djxdt;
span<real_t> djydt;
span<real_t> djzdt;

vector<complex_t> djpdt_storage;

span<complex_t> djpxdt;
span<complex_t> djpydt;
span<complex_t> djpzdt;

fftw_plan plan_djdt_to_djpdt;
fftw_plan plan_djpdt_to_djdt;

// Langevin noise

vector<real_t> l_storage;

span<real_t> lx;
span<real_t> ly;
span<real_t> lz;

vector<complex_t> lp_storage;

span<complex_t> lpx;
span<complex_t> lpy;
span<complex_t> lpz;

fftw_plan plan_l_to_lp;

int Nx;
int Ny;
int Nz;

int Nsites;

double dt;
double sim_time;

double eta = 1.0;
double temp = 1.0;
double mass_density = 1.0;
double eta_uv_cutoff = 1.25;

// Random number generator

unsigned int seed = random_device{}();
mt19937 mt{seed};
normal_distribution<> normal_dist(0, 1);

/**
 * UV regulator for shear viscosity
 * 
 * Full shear viscosity:
 * eta(p) = (1+eta_reg_uv(p^2)) * eta
 */
inline double eta_reg_uv(double p2)
{
    double x = p2/(eta_uv_cutoff*eta_uv_cutoff);
    return x > 1 ? x-1 : 0;
    //return gsl_pow_2(p2/(eta_uv_cutoff*eta_uv_cutoff));
}

inline double eta_k2_dep(double p2)
{
    return (1+eta_reg_uv(p2))*eta;
}

// Rows are a few MB at large lattice sizes; per-value ostream formatting is
// too slow for that. Format with the locale-free to_chars into a reusable
// buffer (same text as ostream's default %.6g) and write each row in one go.
vector<char> row_buf;

inline void append_value(vector<char>& buf, real_t value)
{
    char tmp[32];
    auto [ptr, ec] = to_chars(tmp, tmp+sizeof(tmp), value, chars_format::general, 6);
    buf.insert(buf.end(), tmp, ptr);
    buf.push_back(' ');
}

template <typename T>
void write_row(ofstream& out, const span<T>& values)
{
    row_buf.clear();
    for (const auto& value : values) {
        append_value(row_buf, value);
    }
    row_buf.push_back('\n');
    out.write(row_buf.data(), row_buf.size());
}

void write_row(ofstream& out_re, ofstream& out_im, const span<complex_t>& values)
{
    row_buf.clear();
    for (const auto& value : values) {
        append_value(row_buf, value.real());
    }
    row_buf.push_back('\n');
    out_re.write(row_buf.data(), row_buf.size());

    row_buf.clear();
    for (const auto& value : values) {
        append_value(row_buf, value.imag());
    }
    row_buf.push_back('\n');
    out_im.write(row_buf.data(), row_buf.size());
}

void project_transverse(span<complex_t> jpx, span<complex_t> jpy, span<complex_t> jpz)
{
    int Nzh = Nz/2+1;

    #pragma omp parallel for collapse(2) if(Nsites>=NsitesOMPThreshold)
    for (int nx = 0; nx < Nx; ++nx) {
        for (int ny = 0; ny < Ny; ++ny) {
            for (int nz = 0; nz < Nzh; ++nz) {
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

void do_diss_step(double dt)
{
    for (int i=0; i<Nsites; ++i) {
        lx[i] = normal_dist(mt);
        ly[i] = normal_dist(mt);
        lz[i] = normal_dist(mt);
    }

    fftw_execute(plan_l_to_lp);

    int Nzh = Nz/2+1;

    #pragma omp parallel for collapse(2) if(Nsites>=NsitesOMPThreshold)
    for (int nx = 0; nx < Nx; ++nx) {
        for (int ny = 0; ny < Ny; ++ny) {
            for (int nz = 0; nz < Nzh; ++nz) {
                auto kx = 2*M_PI*(nx/static_cast<real_t>(Nx));
                auto ky = 2*M_PI*(ny/static_cast<real_t>(Ny));
                auto kz = 2*M_PI*(nz/static_cast<real_t>(Nz));

                // lattice spacing a=1
                auto ktx = 2*sin(kx/2);
                auto kty = 2*sin(ky/2);
                auto ktz = 2*sin(kz/2);

                // auto ktx = sin(kx);
                // auto kty = sin(ky);
                // auto ktz = sin(kz);

                auto kt2 = ktx*ktx+kty*kty+ktz*ktz;

                auto damp = eta_k2_dep(kt2)*kt2/mass_density;

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

inline int mod(int x, int m)
{
    int r = x % m;
    return r + (m & (r >> 31));  // adds m only when r is negative
}

int djdt_ideal(double t, const double y[], double dydt[], void *params)
{
    auto *jx = &y[0*Nsites];
    auto *jy = &y[1*Nsites];
    auto *jz = &y[2*Nsites];

    // printf("%d %d\n", -1%8, (-1)&(8-1));
    // exit(0);

    #pragma omp parallel for collapse(2) if(Nsites>=NsitesOMPThreshold)
    for (int nx = 0; nx < Nx; ++nx) {
        for (int ny = 0; ny < Ny; ++ny) {
            int row = nx*Ny*Nz + ny*Nz;

            int row_xm = mod(nx-1, Nx)*Ny*Nz + ny*Nz;
            int row_xp = mod(nx+1, Nx)*Ny*Nz + ny*Nz;

            int row_ym = nx*Ny*Nz + mod(ny-1, Ny)*Nz;
            int row_yp = nx*Ny*Nz + mod(ny+1, Ny)*Nz;

            for (int nz = 0; nz < Nz; ++nz) {
                // Real-space layout must match what fftw_plan_dft_r2c_3d(Nx,
                // Ny, Nz, ...) assumes: nx slowest, nz fastest.
                int idx = row + nz;

                int idx_xm = row_xm + nz;
                int idx_xp = row_xp + nz;

                int idx_ym = row_ym + nz;
                int idx_yp = row_yp + nz;

                int idx_zm = row + mod(nz-1, Nz);
                int idx_zp = row + mod(nz+1, Nz);

                // no static regulator:
                auto vx = jx[idx]/mass_density;
                auto vy = jy[idx]/mass_density;
                auto vz = jz[idx]/mass_density;

                // central differences
                //auto Dx_jx = (jx[idx_xp] - jx[idx_xm])/2.0;
                auto Dy_jx = (jx[idx_yp] - jx[idx_ym])/2.0;
                auto Dz_jx = (jx[idx_zp] - jx[idx_zm])/2.0;

                auto Dx_jy = (jy[idx_xp] - jy[idx_xm])/2.0;
                //auto Dy_jy = (jy[idx_yp] - jy[idx_ym])/2.0;
                auto Dz_jy = (jy[idx_zp] - jy[idx_zm])/2.0;

                auto Dx_jz = (jz[idx_xp] - jz[idx_xm])/2.0;
                auto Dy_jz = (jz[idx_yp] - jz[idx_ym])/2.0;
                //auto Dz_jz = (jz[idx_zp] - jz[idx_zm])/2.0;

                // if(rand()%Nsites==0) {
                // cout<<(Dx_jx+Dy_jy+Dz_jz)<<endl;    
                // }

                djxdt[idx] = -vy*( Dy_jx - Dx_jy ) -vz*( Dz_jx - Dx_jz );
                djydt[idx] = -vx*( Dx_jy - Dy_jx ) -vz*( Dz_jy - Dy_jz );
                djzdt[idx] = -vx*( Dx_jz - Dz_jx ) -vy*( Dy_jz - Dz_jy );
            }
        }
    }

    fftw_execute(plan_djdt_to_djpdt);

    project_transverse(djpxdt, djpydt, djpzdt);

    fftw_execute(plan_djpdt_to_djdt);   // destroys djp*, recomputed next call

    for (int n = 0; n < Nsites; ++n) {
        dydt[n+0*Nsites] = djxdt[n]/Nsites;
        dydt[n+1*Nsites] = djydt[n]/Nsites;
        dydt[n+2*Nsites] = djzdt[n]/Nsites;
    }

    return GSL_SUCCESS;
}

int main(int argc, const char *argv[])
{
    auto wall_clock_start = chrono::steady_clock::now();

    std::string output_folder;

    // Add a command line option for this
    int write_every_n_steps = 1;

    bool no_ideal_step = false;

    bool no_therm = false;

    po::options_description desc("Options");
    desc.add_options()
        ("help", "print help message")
        ("nx", po::value<int>(&Nx)->required(), "Number of lattice sites in x")
        ("ny", po::value<int>(&Ny)->required(), "Number of lattice sites in y")
        ("nz", po::value<int>(&Nz)->required(), "Number of lattice sites in z")
        ("dt", po::value<double>(&dt)->required(), "Time step used in the simulation")
        ("sim-time", po::value<double>(&sim_time)->required(), "Simulation time")
        ("output-every-n-steps", po::value<int>(&write_every_n_steps)->default_value(write_every_n_steps), "Write output only every n time steps (default: 1)")
        ("eta", po::value<double>(&eta)->default_value(eta), "Shear viscosity")
        ("eta-uv-cutoff", po::value<double>(&eta_uv_cutoff)->default_value(eta_uv_cutoff), "UV cutoff for shear viscosity")
        ("seed", po::value<unsigned int>(&seed), "Seed for the random number generator (default: random)")
        ("no-ideal-step", po::bool_switch(&no_ideal_step), "Disable the ideal step (only dissipative dynamics)")
        ("no-therm", po::bool_switch(&no_therm), "Disable the thermalization at the beginning")
        ("output-folder", po::value<std::string>(&output_folder), "Folder to write output files to (default: generated from simulation parameters)");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
        cout << desc << endl;
        return 0;
    }

    po::notify(vm);

    if (vm.count("seed")) {
        mt.seed(seed);
    }

    if (!vm.count("output-folder")) {
        std::ostringstream name;
        name << "sim-Nx" << Nx << "Ny" << Ny << "Nz" << Nz
             << "dt" << (dt*write_every_n_steps) << "eta" << eta << "Lam" << eta_uv_cutoff
             << "seed" << seed;
        if (no_ideal_step) {
            name << "x";
        }
        output_folder = name.str();
        cout << "Writing output to " << output_folder << endl;
    }

    fs::create_directories(output_folder);

    // Threaded FFTW: every batched plan below runs on all available cores.
    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());

    Nsites = Nx*Ny*Nz;
    if (Nsites>=NsitesOMPThreshold) {
        cout << "Using OpenMP in for loops" << endl;
    }

    cout << "OpenMP/FFTW threads: " << omp_get_max_threads()
        << " (of " << omp_get_num_procs() << " logical cores)" << endl;

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

    const int fft_dims[3] = {Nx, Ny, Nz};

    // NOTE: FFTW_MEASURE overwrites the arrays while planning, so all fields
    // must be initialized only after the plans below have been created.

    plan_j_to_jp = fftw_plan_many_dft_r2c(3, fft_dims, 3,
        j_storage.data(), nullptr, 1, Nsites,
        reinterpret_cast<fftw_complex *>(jp_storage.data()), nullptr, 1, Np,
        FFTW_MEASURE);

    plan_jp_to_j = fftw_plan_many_dft_c2r(3, fft_dims, 3,
        reinterpret_cast<fftw_complex *>(jp_storage.data()), nullptr, 1, Np,
        j_storage.data(), nullptr, 1, Nsites,
        FFTW_MEASURE);

    // Time derivative of momentum density

    djdt_storage = vector<real_t>(3*Nsites, 0);
    djxdt = span<real_t>(djdt_storage.data()+0*Nsites, Nsites);
    djydt = span<real_t>(djdt_storage.data()+1*Nsites, Nsites);
    djzdt = span<real_t>(djdt_storage.data()+2*Nsites, Nsites);

    djpdt_storage = vector<complex_t>(3*Np, complex_t{0, 0});
    djpxdt = span<complex_t>(djpdt_storage.data()+0*Np, Np);
    djpydt = span<complex_t>(djpdt_storage.data()+1*Np, Np);
    djpzdt = span<complex_t>(djpdt_storage.data()+2*Np, Np);

    plan_djdt_to_djpdt = fftw_plan_many_dft_r2c(3, fft_dims, 3,
        djdt_storage.data(), nullptr, 1, Nsites,
        reinterpret_cast<fftw_complex *>(djpdt_storage.data()), nullptr, 1, Np,
        FFTW_MEASURE);

    plan_djpdt_to_djdt = fftw_plan_many_dft_c2r(3, fft_dims, 3,
        reinterpret_cast<fftw_complex *>(djpdt_storage.data()), nullptr, 1, Np,
        djdt_storage.data(), nullptr, 1, Nsites,
        FFTW_MEASURE);

    // Noise

    l_storage = vector<real_t>(3*Nsites, 0);

    lx = span<real_t>(l_storage.data()+0*Nsites, Nsites);
    ly = span<real_t>(l_storage.data()+1*Nsites, Nsites);
    lz = span<real_t>(l_storage.data()+2*Nsites, Nsites);

    lp_storage = vector<complex_t>(3*Np, complex_t{0, 0}); // The last dimension is cut in half + 1

    lpx = span<complex_t>(lp_storage.data()+0*Np, Np);
    lpy = span<complex_t>(lp_storage.data()+1*Np, Np);
    lpz = span<complex_t>(lp_storage.data()+2*Np, Np);

    plan_l_to_lp = fftw_plan_many_dft_r2c(3, fft_dims, 3,
        l_storage.data(), nullptr, 1, Nsites,
        reinterpret_cast<fftw_complex *>(lp_storage.data()), nullptr, 1, Np,
        FFTW_MEASURE);

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

    //             jx[idx] = sin(nx/static_cast<real_t>(Nx)*2*M_PI)*cos(ny/static_cast<real_t>(Ny)*2*M_PI)+1.0;
    //             jy[idx] = -cos(nx/static_cast<real_t>(Nx)*2*M_PI)*sin(ny/static_cast<real_t>(Ny)*2*M_PI)+1.0;
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
    // gsl_odeiv2_step_rk2 doesn't work

    auto jerr_ideal_step_storage = vector<real_t>(3*Nsites, 0);

    // Thermalization

    auto k_min = 2*M_PI/static_cast<real_t>(max({Nx, Ny, Nz}));
    auto eq_time_slow = 1/(eta_k2_dep(k_min*k_min)*k_min*k_min/mass_density);

    auto therm_time = 5.0*eq_time_slow;

    const double therm_dt = therm_time/1000.0;

    const int therm_steps = static_cast<int>(floor(therm_time/therm_dt));

    auto k_max=M_PI;
    auto eq_time_fast = 1/(eta_k2_dep(k_max*k_max)*k_max*k_max/mass_density);

    cout << "Relaxation time of slowest mode= " << eq_time_slow << " (k_min=" << k_min << ")" << endl;
    cout << "  Numerical thermalization time= " << therm_time << endl;
    cout << "                Simulation time= " << sim_time << endl;
    cout << "Relaxation time of fastest mode= " << eq_time_fast << endl;
    cout << "                             dt= " << dt << endl;
    
    if (no_therm) {
        cout << "Thermalization disabled (--no-therm)" << endl;
    } else {
        auto therm_start = chrono::steady_clock::now();
        cout << "Thermalizing (" << therm_steps << " steps)..." << endl;

        fftw_execute(plan_j_to_jp);

        auto therm_t = 0.0;

        for (int i=0; i<therm_steps; ++i) {
            do_diss_step(therm_dt);

            // cout << "t=" << therm_t << "\r";
            // cout.flush();

            therm_t += therm_dt;
        }

        fftw_execute(plan_jp_to_j);

        for (int n = 0; n < Nsites; ++n) {
            jx[n] /= Nsites;
            jy[n] /= Nsites;
            jz[n] /= Nsites;
        }

        chrono::duration<double> therm_elapsed = chrono::steady_clock::now() - therm_start;
        cout << "Finished (took " << therm_elapsed.count() << " s)" << endl;
    }

    // Actual simulation in equilibrium state

    double t = 0.0;
    int i = 0;

    int n_steps = static_cast<int>(ceil(sim_time/dt));

    auto main_loop_start = chrono::steady_clock::now();

    while (t<sim_time) {
        auto step_start = chrono::steady_clock::now();

        // DISSIPATIVE STEP

        fftw_execute(plan_j_to_jp);
        // JX IS DESTROYED

        project_transverse(jpx, jpy, jpz);
        
        do_diss_step(dt);

        // TODO Optimize
        project_transverse(jpx, jpy, jpz);

        if (i%write_every_n_steps == 0) {

            write_row(jpx_re_file, jpx_im_file, jpx);
            write_row(jpy_re_file, jpy_im_file, jpy);
            write_row(jpz_re_file, jpz_im_file, jpz);

        }

        // OUTPUT, THEN TRANSFORM TO REAL SPACE

        // HERE WE SHOULD READ OUT JX, JY, JZ

        fftw_execute(plan_jp_to_j);
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
        if (!no_ideal_step) {
            int status = gsl_odeiv2_step_apply(id_step, t, dt, &j_storage[0], &jerr_ideal_step_storage[0], nullptr, nullptr, &sys);
            if (status != GSL_SUCCESS) {
                fprintf(stderr, "Ideal step failed: %s\n", gsl_strerror(status));
                return 1;
            }
        }

        t += dt;
        ++i;

        if (i%write_every_n_steps == 0) {

            chrono::duration<double> step_elapsed = chrono::steady_clock::now() - step_start;
            chrono::duration<double> main_loop_elapsed = chrono::steady_clock::now() - main_loop_start;
            cout << i << "/" << n_steps << " steps (runtime: " << main_loop_elapsed.count() << " s, last step took " << step_elapsed.count()*1e3 << " ms)" << endl;
            // cout.flush();

        }

        // cout << "Finished step " << i << "/" << n_steps << endl;
    }

    chrono::duration<double> main_loop_elapsed = chrono::steady_clock::now() - main_loop_start;
    cout << "                 Main loop time= " << main_loop_elapsed.count() << " s"
         << " (" << (i > 0 ? main_loop_elapsed.count()/i*1e3 : 0.0) << " ms/step, " << i << " steps)" << endl;

    gsl_odeiv2_step_free(id_step);

    fftw_destroy_plan(plan_j_to_jp);
    fftw_destroy_plan(plan_jp_to_j);

    fftw_destroy_plan(plan_djdt_to_djpdt);
    fftw_destroy_plan(plan_djpdt_to_djdt);

    fftw_destroy_plan(plan_l_to_lp);

    fftw_cleanup_threads();

    chrono::duration<double> wall_clock_elapsed = chrono::steady_clock::now() - wall_clock_start;
    cout << "                 Total run time= " << wall_clock_elapsed.count() << "s" << endl;

    return 0;
}
