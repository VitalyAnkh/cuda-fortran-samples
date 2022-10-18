cmake_minimum_required(VERSION 3.23)
project(cuda_fortran_samples CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(cuda_fortran_samples src/main.cu)

set_target_properties(cuda_fortran_samples PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
