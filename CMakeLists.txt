cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 17)
project(your_app)

set(ONNXRUNTIME_ROOTDIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-x64-1.12.0")

#set(ONNXRUNTIME_ROOTDIR "${CMAKE_CURRENT_SOURCE_DIR}/libonnxruntime.so")
# include_directories("${ONNXRUNTIME_ROOTDIR}/include"                           # Pre-built package
#                     "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime"               # Linux local install to /usr/local
#                     "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session") # Windows local install

# link_directories("${ONNXRUNTIME_ROOTDIR}/lib")


# include_directories("${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-x64-1.12.0/include")
find_package(PythonInterp 3.8.10 EXACT REQUIRED COMPONENTS Development)
find_package(PythonLibs 3.8.10 EXACT REQUIRED COMPONENTS Development)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenCV REQUIRED)
# set(ONNXRUNTIME_ROOT "/home/manish/onnxruntime")
# find_package(ONNXRuntime REQUIRED)


set(Python_INCLUDE_DIRS "/usr/include/python3.8")
include_directories(${OpenCV_INCLUDE_DIRS})
# include_directories(include)
include_directories(${Python_INCLUDE_DIRS})
#include_directories(/usr/local/include/onnxruntime)

# pybind11_add_module(your_app src/main.cpp)
# target_link_libraries(your_app PUBLIC /home/manish/onnxruntime-linux-x64-1.12.0/lib/libonnxruntime.so ${OpenCV_LIBS} )


# add_executable(your_app1 src/main2.cpp)
# target_link_libraries(your_app1 PUBLIC /home/manish/onnxruntime-linux-x64-1.17.1/lib/libonnxruntime.so ${OpenCV_LIBS} )
file(GLOB SOURCE_FILES "src/*.cpp")
file(GLOB HEADER_FILES "src/*.h")

pybind11_add_module(run_yolo_onnx ${SOURCE_FILES} ${HEADER_FILES})
# add_executable(run_yolo_onnx src/main3.cpp)
target_include_directories(run_yolo_onnx PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)



find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS /usr/local/include/onnxruntime/core/session/)
find_path(ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS cuda_provider_factory.h HINTS /usr/local/include/onnxruntime/core/providers/cuda/)
find_path(ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS cpu_provider_factory.h HINTS /usr/local/include/onnxruntime/core/providers/cpu/)

find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /usr/local/lib)

message("lib : ${ONNX_RUNTIME_LIB}")

include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS})
include_directories(${PROJECT_NAME} /usr/local/include/onnxruntime)


target_link_libraries(run_yolo_onnx PRIVATE ${ONNX_RUNTIME_LIB} ${OpenCV_LIBS} )

# target_compile_definitions(${PROJECT_NAME} PRIVATE -DENABLE_ORT)
# target_link_libraries(${PROJECT_NAME} PUBLIC ${ONNX_RUNTIME_LIB})
