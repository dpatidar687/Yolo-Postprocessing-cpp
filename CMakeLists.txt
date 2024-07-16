cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 17)
project(your_app)


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


file(GLOB SOURCE_FILES "src/*.cpp" "models/*.cpp")
file(GLOB HEADER_FILES "src/*.h" "models/*.h")


pybind11_add_module(run_yolo_onnx ${SOURCE_FILES} ${HEADER_FILES})
target_include_directories(run_yolo_onnx PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/models)



find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS /usr/local/include/onnxruntime/core/session/)
find_path(ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS cuda_provider_factory.h HINTS /usr/local/include/onnxruntime/core/providers/cuda/)
find_path(ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS cpu_provider_factory.h HINTS /usr/local/include/onnxruntime/core/providers/cpu/)

find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /usr/local/lib)

message("lib : ${ONNX_RUNTIME_LIB}")

include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS})
include_directories(${PROJECT_NAME} /usr/local/include/onnxruntime)


target_link_libraries(run_yolo_onnx PRIVATE ${ONNX_RUNTIME_LIB} ${OpenCV_LIBS}  )





