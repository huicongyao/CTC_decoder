cmake_minimum_required(VERSION 3.22)
project(CTC_decoder)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
    message(STATUS "CMAKE_BUILD_TYPE = Debug")
else ()
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O2 ")
    message(STATUS "CMAKE_BUILD_TYPE = Release")
endif ()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.4/bin/nvcc)
set(CMAKE_PREFIX_PATH /home/yaohc/miniconda3/envs/Torch2.0/lib/python3.9/site-packages/torch/share/cmake)
find_package(Torch REQUIRED)
find_package(spdlog REQUIRED)
find_package(ZLIB REQUIRED)
find_package(OpenMP REQUIRED)

#add_library(ctc_decode SHARED ctc_decode.cpp
#        ctc_decoder.h
#        ctc_utils.h)
#
#target_link_libraries(
#        ctc_decode
#        ${TORCH_LIBRARIES}
#)

add_executable(
        ${PROJECT_NAME}
        main.cpp
        ctc_decode_C.h
        ctc_decode_C.cpp
        ctc_decoder.h
        ctc_decoder.cpp
        3rdParty/cnpy.h
        3rdParty/cnpy.cpp
)

target_link_libraries (
        ${PROJECT_NAME}
        PUBLIC
        ${TORCH_LIBRARIES}
        ${ZLIB_LIBRARIES}
        OpenMP::OpenMP_CXX
)





#pybind11_add_module(hadamard_module pybind_test.cpp)