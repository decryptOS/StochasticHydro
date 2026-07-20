cmake_minimum_required( VERSION 3.20 )
project( 
    sim 
    LANGUAGES CXX 
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(GSL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3 REQUIRED IMPORTED_TARGET fftw3)
find_library(FFTW3_OMP_LIBRARY fftw3_omp REQUIRED HINTS ${FFTW3_LIBRARY_DIRS})
find_package(Boost REQUIRED CONFIG COMPONENTS program_options)

# Apple clang ships without an OpenMP runtime; point FindOpenMP at
# Homebrew's keg-only libomp.
if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY
    )
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
    set(OpenMP_CXX_LIB_NAMES omp)
    set(OpenMP_omp_LIBRARY ${LIBOMP_PREFIX}/lib/libomp.dylib)
endif()

find_package(OpenMP REQUIRED)

add_executable( sim src/main.cpp )

target_compile_options( sim PRIVATE -O3 -Wall -march=native )
target_compile_definitions( sim PRIVATE HAVE_INLINE )

target_link_libraries( sim
    GSL::gsl
    GSL::gslcblas
    Boost::boost
    Boost::program_options
    ${FFTW3_OMP_LIBRARY}
    PkgConfig::FFTW3
    OpenMP::OpenMP_CXX
)
