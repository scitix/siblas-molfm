cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_ARCHITECTURES 100a)
project(bench_backward CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

# Prefer real shared libraries over conda stubs
if(DEFINED ENV{CONDA_PREFIX})
    set(_CONDA_LIB "$ENV{CONDA_PREFIX}/lib")
    foreach(_lib cublas cublasLt curand)
        set(_cache_var "CUDA_${_lib}_LIBRARY")
        if(${_cache_var} MATCHES "/stubs/")
            set(_real "${_CONDA_LIB}/lib${_lib}.so")
            if(EXISTS "${_real}")
                set(${_cache_var} "${_real}" CACHE FILEPATH "Override stubs" FORCE)
                message(STATUS "Overriding ${_cache_var} -> ${_real}")
            endif()
        endif()
    endforeach()
endif()

# CUTLASS (header-only)
if(NOT CUTLASS_DIR)
    set(CUTLASS_DIR "/volume/code/jjcheng/cutlass")
endif()
set(CUTLASS_INCLUDE "${CUTLASS_DIR}/include")
set(CUTLASS_UTIL_INCLUDE "${CUTLASS_DIR}/tools/util/include")

if(NOT EXISTS "${CUTLASS_INCLUDE}/cutlass/cutlass.h")
    message(FATAL_ERROR "CUTLASS not found at ${CUTLASS_INCLUDE}. Pass -DCUTLASS_DIR=/path/to/cutlass")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -DNDEBUG --expt-relaxed-constexpr \
    -DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1 \
    -DCUTLASS_ENABLE_GDC_FOR_SM100=1 \
    -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1")

add_executable(bench_backward bench_backward.cu)

target_include_directories(bench_backward PRIVATE
    ${CUTLASS_INCLUDE}
    ${CUTLASS_UTIL_INCLUDE}
    $ENV{CONDA_PREFIX}/include
)

target_link_libraries(bench_backward PRIVATE
    CUDA::cublas
    CUDA::cublasLt
    CUDA::cudart
    CUDA::curand
)
