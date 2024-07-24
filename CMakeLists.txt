cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 17)
project(run_yolo_onnx C CXX)



find_package(PythonInterp 3.8.10 EXACT REQUIRED COMPONENTS Development)
find_package(PythonLibs 3.8.10 EXACT REQUIRED COMPONENTS Development)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenCV REQUIRED)


set(Python_INCLUDE_DIRS "/usr/include/python3.8")
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Python_INCLUDE_DIRS})




option(BUILD_WITH_CUDA "Build lib with cuda" ON)
option(BUILD_WITH_DEEPSTREAM "Build with DEEPSTREAM library" ON)
option(BUILD_WITH_ONNXRUNTIME "Build with ONNX Runtime library" ON)
option(BUILD_WITH_OPENVINO "Build with OpenVINO" ON)
option(BUILD_WITH_MILVUS "Build with Milvus" ON)
option(BUILD_WITH_SPDLOG "Build with spdlog library" ON)
option(BUILD_WITH_TENSORRT "Build with TensorRT" ON)




file(GLOB SOURCE_FILES "src/*.cpp" 
"models/*.cpp" 
"accelerators/OpenVINO/*.cpp"
  "accelerators/TensorRT/*.cpp"
  "accelerators/ORT/*.cpp"
  )
file(GLOB HEADER_FILES "src/*.h" 
"models/*.h" 
"accelerators/OpenVINO/*.h"
  "accelerators/TensorRT/*.h"
  "accelerators/ORT/*.h"
)


pybind11_add_module(run_yolo_onnx ${SOURCE_FILES} ${HEADER_FILES})

target_include_directories(run_yolo_onnx PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src 
    ${CMAKE_CURRENT_SOURCE_DIR}/models
    ${CMAKE_CURRENT_SOURCE_DIR}/accelerators/OpenVINO
    ${CMAKE_CURRENT_SOURCE_DIR}/accelerators/TensorRT
    ${CMAKE_CURRENT_SOURCE_DIR}/accelerators/ORT
    )



find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS /usr/local/include/onnxruntime/core/session/)
find_path(ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS cuda_provider_factory.h HINTS /usr/local/include/onnxruntime/core/providers/cuda/)
find_path(ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS cpu_provider_factory.h HINTS /usr/local/include/onnxruntime/core/providers/cpu/)

find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /usr/local/lib)

message("lib : ${ONNX_RUNTIME_LIB}")

include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS})
include_directories(${PROJECT_NAME} /usr/local/include/onnxruntime)
target_link_libraries(run_yolo_onnx PRIVATE ${ONNX_RUNTIME_LIB} ${OpenCV_LIBS} )  



# find_package(OpenVINO REQUIRED)
# include_directories(/opt/intel/openvino/runtime/include)
# target_link_libraries(${PROJECT_NAME} PRIVATE openvino::runtime)
# target_compile_definitions(${PROJECT_NAME} PRIVATE -DENABLE_VINO)

set(CMAKE_PREFIX_PATH "/opt/intel/openvino/runtime/cmake" ${CMAKE_PREFIX_PATH})
if(BUILD_WITH_SPDLOG)
	include(cmake/spdlog.cmake)
endif(BUILD_WITH_SPDLOG)

if(BUILD_WITH_OPENVINO)
  include(cmake/openvino.cmake)
endif(BUILD_WITH_OPENVINO)

if(BUILD_WITH_TENSORRT)
  include(cmake/tensorrt.cmake)
endif(BUILD_WITH_TENSORRT)


if(BUILD_WITH_CUDA)
    target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC "/usr/local/cuda/include")
    target_compile_definitions(${PROJECT_NAME} PRIVATE -DENABLE_CUDA)
    target_link_libraries(${PROJECT_NAME} PUBLIC -L/usr/local/cuda/lib64)
endif(BUILD_WITH_CUDA)
