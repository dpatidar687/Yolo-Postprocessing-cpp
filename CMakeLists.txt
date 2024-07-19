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


# if(BUILD_WITH_SPDLOG)
# 	include(cmake/spdlog.cmake)
# endif(BUILD_WITH_SPDLOG)

# if(BUILD_WITH_CUDA)
#     target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC "/usr/local/cuda/include")
#     target_compile_definitions(${PROJECT_NAME} PRIVATE -DENABLE_CUDA)
#     target_link_libraries(${PROJECT_NAME} PUBLIC -L/usr/local/cuda/lib64)
# endif(BUILD_WITH_CUDA)

# if(BUILD_WITH_DEEPSTREAM)
#   include(cmake/deepstream.cmake)
# endif(BUILD_WITH_DEEPSTREAM)

# if(BUILD_WITH_ONNXRUNTIME)
#   include(cmake/onnxruntime.cmake)
# endif(BUILD_WITH_ONNXRUNTIME)

# if(BUILD_WITH_OPENVINO)
#   include(cmake/openvino.cmake)
# endif(BUILD_WITH_OPENVINO)

# if(BUILD_WITH_TENSORRT)
#   include_directories("/usr/local/cuda-11.8/targets/x86_64-linux/include/")
#   include(cmake/tensorrt.cmake)
# endif(BUILD_WITH_TENSORRT)









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

find_package(OpenVINO REQUIRED)
include_directories(/opt/intel/openvino/runtime/include)
target_link_libraries(${PROJECT_NAME} PRIVATE openvino::runtime)
target_compile_definitions(${PROJECT_NAME} PRIVATE -DENABLE_VINO)




# message("lib : ${OpenVINO}")

# target_link_directories(run_yolo_onnx PRIVATE ${OpenVINO_INCLUDE_DIRS})


#################################################################################################################~

# cmake_minimum_required(VERSION 3.15)
# project(perceptor C CXX)
# set (PROJECT_VERSION 1.1)
# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_VERBOSE_MAKEFILE OFF)
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/Release)


# #------ OPTIONS -------
# option(BUILD_WITH_CURL "Build lib with curl" ON)
# option(BUILD_WITH_CUDA "Build lib with cuda" ON)
# option(BUILD_WITH_DEEPSTREAM "Build with DEEPSTREAM library" ON)
# option(BUILD_WITH_ONNXRUNTIME "Build with ONNX Runtime library" ON)
# option(BUILD_WITH_OPENVINO "Build with OpenVINO" ON)
# option(BUILD_WITH_MILVUS "Build with Milvus" ON)
# option(BUILD_WITH_SPDLOG "Build with spdlog library" ON)
# option(BUILD_WITH_TENSORRT "Build with TensorRT" ON)
# option(BUILD_WITH_EIGEN "Build with Eigen" ON)
# option(DEBUG           "ENABLE DEBUG MODE" OFF)
# option(DEV_MODE "Enable Development mode" OFF)
# option(VIDEO_LOOPING "Enable Video Looping" ON)



# find_package(OpenCV REQUIRED)


# #------ MTX CORE ----
# file(GLOB SOURCE_FILES 
#   "src/gstreamer/*.cpp"
#   "src/core/*.cpp" 
#   "src/event/*.cpp" 
#   "src/models/*.cpp" 
#   "src/modules/*.cpp" 
#   "ext/utils/*.cpp"
#   "src/lambdas/create_vectors/*.cpp"
#   "src/accelerators/OpenVINO/*.cpp"
#   "src/accelerators/TensorRT/*.cpp"
#   "src/accelerators/ORT/*.cpp"
#   )


# add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} )
# target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/src )
# target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/src/gstreamer )
# target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR})
# target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/src/event)
# target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/src/core)
# target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/src/models)
# target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/src/accelerators/OpenVINO)
# target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/src/accelerators/TensorRT)
# target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/src/accelerators/ORT)


# if(BUILD_WITH_CUDA)
#     target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC "/usr/local/cuda/include")
#     target_compile_definitions(${PROJECT_NAME} PRIVATE -DENABLE_CUDA)
#     target_link_libraries(${PROJECT_NAME} PUBLIC -L/usr/local/cuda/lib64)
# endif(BUILD_WITH_CUDA)

# target_compile_definitions(${PROJECT_NAME}
#     PUBLIC "USE_SSE4_INSTRUCTIONS=ON"
#     PUBLIC "USE_AVX_INSTRUCTIONS=ON"
# )

# target_compile_options(${PROJECT_NAME}
#     #PUBLIC -fPIC
#     PUBLIC -g
# )

# #-------MILVUS INTERFACES-------------------
# if(BUILD_WITH_MILVUS)
#     target_link_libraries(${PROJECT_NAME} PRIVATE milvus_sdk)
# endif(BUILD_WITH_MILVUS)

# if(BUILD_WITH_SPDLOG)
# 	include(cmake/spdlog.cmake)
# endif(BUILD_WITH_SPDLOG)

# if(BUILD_WITH_ONNXRUNTIME)
#     target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/ext/onnx/includes)
#     target_link_libraries(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/libonnx_handy_lib.a)
# endif(BUILD_WITH_ONNXRUNTIME)

# if(BUILD_WITH_CURL)
#   include(cmake/curl.cmake)
# endif(BUILD_WITH_CURL)

# include(cmake/gst.cmake)

# if(BUILD_WITH_DEEPSTREAM)
#   include(cmake/deepstream.cmake)
# endif(BUILD_WITH_DEEPSTREAM)

# if(BUILD_WITH_ONNXRUNTIME)
#   include(cmake/onnxruntime.cmake)
# endif(BUILD_WITH_ONNXRUNTIME)

# if(BUILD_WITH_OPENVINO)
#   include(cmake/openvino.cmake)
# endif(BUILD_WITH_OPENVINO)

# if(BUILD_WITH_TENSORRT)
#   include(cmake/tensorrt.cmake)
# endif(BUILD_WITH_TENSORRT)

# if(BUILD_WITH_EIGEN)
#   include(cmake/eigen.cmake)
# endif(BUILD_WITH_EIGEN)

# if(DEV_MODE)
#   add_definitions(-DMODEL_PATH="/docker/apps/models/")
#   target_compile_definitions(${PROJECT_NAME} PRIVATE -DDEV_MODE)
# else()
#   add_definitions(-DMODEL_PATH="/docker/models/")
# endif(DEV_MODE)

# if(VIDEO_LOOPING)
#   target_compile_definitions(${PROJECT_NAME} PRIVATE -DVIDEO_LOOPING)
# endif(VIDEO_LOOPING)

# add_definitions(-w)

# #---------------------------

# #------ EVAL PACKAGES ------
# # NOTE: If package is optional, package_FOUND is to be handled
# include_directories( ${OpenCV_INCLUDE_DIRS} )
# target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS} )
# target_link_libraries(${PROJECT_NAME} PUBLIC -lrdkafka -lcppkafka )
# #target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} -lboost_system -lboost_thread)
# #---------------------------


