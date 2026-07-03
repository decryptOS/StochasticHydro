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
find_package(Boost REQUIRED CONFIG COMPONENTS program_options)

add_executable( sim src/main.cpp )

target_compile_options( sim PRIVATE -O3 -Wall -Wconversion -march=native )
target_compile_definitions( sim PRIVATE HAVE_INLINE )

target_link_libraries( sim
    GSL::gsl
    GSL::gslcblas
    Boost::boost
    Boost::program_options
    PkgConfig::FFTW3
)
