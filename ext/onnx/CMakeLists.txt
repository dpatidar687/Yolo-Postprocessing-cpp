cmake_minimum_required(VERSION 3.15)
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
project(onnx_handy_lib C CXX)
set(CMAKE_CXX_STANDARD 17)

set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR})

#FIND_PACKAGE(OpenMP)




set(SRC_DIR ${PROJECT_SOURCE_DIR}/src/)
file(GLOB SRC ${SRC_DIR}*.cpp)
message("globbed source files : ${SRC}")
add_library(onnx_handy_lib STATIC ${SRC})

target_compile_options(${PROJECT_NAME} PUBLIC "-DUSE_SSE4_INSTRUCTIONS=ON"
    PUBLIC "-DUSE_AVX_INSTRUCTIONS=ON"
    #PUBLIC -Wall -pedantic -Wextra
    PUBLIC -fPIC
    )

find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS /usr/local/include/onnxruntime/core/session/)
find_path(ONNX_RUNTIME_PROVIDER_OPTIONS_INCLUDE_DIRS core/framework/provider_options.h HINTS /usr/local/include/onnxruntime/)

find_path(ONNX_RUNTIME_OPENVINO_INCLUDE_DIRS openvino_provider_factory.h HINTS /usr/local/include/onnxruntime/core/providers/openvino/)
find_path(ONNX_RUNTIME_TRT_INCLUDE_DIRS tensorrt_provider_factory.h HINTS /usr/local/include/onnxruntime/core/providers/tensorrt/)
find_path(ONNX_RUNTIME_CUDA_PROVIDER_INCLUDE_DIRS cuda_provider_factory.h HINTS /usr/local/include/onnxruntime/core/providers/cuda/)

find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /usr/local/lib)


target_include_directories(onnx_handy_lib PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${ONNX_RUNTIME_CUDA_PROVIDER_INCLUDE_DIRS} ${ONNX_RUNTIME_OPENVINO_INCLUDE_DIRS} ${ONNX_RUNTIME_TRT_INCLUDE_DIRS} ${ONNX_RUNTIME_PROVIDER_OPTIONS_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/includes)

target_link_libraries(onnx_handy_lib ${ONNX_RUNTIME_LIB})



