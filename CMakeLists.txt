cmake_minimum_required(VERSION 3.22)
project(CTC_decoder)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
    message(STATUS "CMAKE_BUILD_TYPE = Debug")
else ()
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
    message(STATUS "CMAKE_BUILD_TYPE = Release")
endif ()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH /home/yaohc/miniconda3/envs/Torch2.0/lib/python3.9/site-packages/torch/share/cmake)
find_package(Torch REQUIRED)
#find_package(ZLIB REQUIRED)
find_package(spdlog REQUIRED)


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
)

target_link_libraries (
        ${PROJECT_NAME}
        ${TORCH_LIBRARIES}
)





#pybind11_add_module(hadamard_module pybind_test.cpp)