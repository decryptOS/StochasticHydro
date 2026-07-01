cmake_minimum_required( VERSION 3.20 )
project( 
    st-hydro 
    LANGUAGES CXX 
)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "-O3 -Wall -march=native -DHAVE_INLINE") # -Wall -Wextra

find_package(GSL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3 REQUIRED IMPORTED_TARGET fftw3)
find_package(Boost REQUIRED CONFIG)

add_executable( st-hydro src/main.cpp )

target_compile_options( st-hydro PRIVATE -O3 -Wall -march=native )
target_compile_definitions( st-hydro PRIVATE HAVE_INLINE )

target_link_libraries( st-hydro
    GSL::gsl
    GSL::gslcblas
    Boost::boost
    PkgConfig::FFTW3
)
