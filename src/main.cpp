#include <iostream>
#include <complex>

#include <gsl/gsl_pow_int.h>
#include <fftw3.h>

using namespace std;

using real_t = double;
using complex_t = std::complex<real_t>;

std::array<std::vector<real_t>, 3> jx;
std::array<std::vector<complex_t>, 3> jp;

fftw_plan plan;
fftw_plan plan_inverse;

int Nx = 8;
int Ny = 8;
int Nz = 1;

double dt = 0.1;
double eta = 1.0;
// mass density = 1

void print_jx(int i)
{
    for (int nz = 0; nz < Nz/2+1; ++nz) {
        for (int ny = 0; ny < Ny; ++ny) {
            for (int nx = 0; nx < Nx; ++nx) {
                int idx = nz*Nx*Ny + ny*Nx + nx;

                cout << jx[i][idx] << " ";
            }
            cout << endl;
        }
    }
}

void print_jp(int i)
{
    for (int nz = 0; nz < Nz/2+1; ++nz) {
        for (int ny = 0; ny < Ny; ++ny) {
            for (int nx = 0; nx < Nx; ++nx) {
                int idx = nz*Nx*Ny + ny*Nx + nx;

                cout << jx[i][idx] << " ";
            }
            cout << endl;
        }
    }
}

int main(int argc, const char *argv[])
{
    cout << "Program started!" << endl;

    //WARNING:
//The fftw_plan_dft_c2r function destroys (overwrites) its input array (out) during execution to save memory.

    for (int i = 0; i < 1; ++i) {
        jx[i] = std::vector<real_t>(Nx*Ny*Nz, 0);
        jp[i] = std::vector<complex_t>(Nx*Ny * (Nz/2+1), complex_t{0, 0}); // The last dimension is cut in half + 1

        for (int n = 0; n < jx[i].size(); ++n) {
            jx[i][n] = sin(n/Nx);
        }

        print_jx(i);
        cout << endl;

        plan = fftw_plan_dft_r2c_3d(Nx, Ny, Nz, &jx[i][0], reinterpret_cast<fftw_complex *>(&jp[i][0]), FFTW_ESTIMATE);
        plan_inverse = fftw_plan_dft_c2r_3d(Nx, Ny, Nz, reinterpret_cast<fftw_complex *>(&jp[i][0]), &jx[i][0], FFTW_ESTIMATE);

        // Initialize the 2D input array with some dummy data
        // Row-major order: index = row * Width + col
        // for (int r = 0; r < H; r++) {
        //     for (int c = 0; c < W; c++) {
        //         jx[r * W + c] = (double)(r + c); 
        //     }
        // }

        // Execute the 2D FFT

        fftw_execute(plan);
        // JX IS DESTROYED

        print_jp(i);
        cout << endl;

        // Time evolution
        for (int nz = 0; nz < Nz/2+1; ++nz) {
            for (int ny = 0; ny < Ny; ++ny) {
            for (int nx = 0; nx < Nx; ++nx) {
                int idx = nz*Nx*Ny + ny*Nx + nx;

                real_t kx = 2*M_PI*nx/Nx;
                real_t ky = 2*M_PI*ny/Ny;
                real_t kz = 2*M_PI*nz/Nz;

                real_t damp = eta*(kx*kx+ky*ky+kz*kz);

                jp[i][idx] *= exp(-damp*dt);
            }
        }
        }

        print_jp(i);
        cout << endl;

        fftw_execute(plan_inverse);
        for (int n = 0; n < jx[i].size(); ++n) {
            jx[i][n] /= Nx*Ny*Nz;
        }
        // JP IS DESTROYED

        print_jx(i);

        // Print the complex output array (Size: H x W_complex)
        // printf("2D FFT Output (%d x %d complex grid):\n", H, W_complex);
        // for (int r = 0; r < H; r++) {
        //     for (int c = 0; c < W_complex; c++) {
        //         int idx = r * W_complex + c;
        //         printf("out[%d][%d] = %6.2f + %6.2fi\t", r, c, out[idx][0], out[idx][1]);
        //     }
        //     printf("\n");
        // }
        
        // Clean up
        fftw_destroy_plan(plan);
    }

    return 0;
}
