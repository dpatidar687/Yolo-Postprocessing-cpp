cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 17)
project(your_app)


set(ONNXRUNTIME_ROOTDIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-x64-1.12.0")
# include_directories("${ONNXRUNTIME_ROOTDIR}/include"                           # Pre-built package
#                     "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime"               # Linux local install to /usr/local
#                     "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session") # Windows local install

# link_directories("${ONNXRUNTIME_ROOTDIR}/lib")


# include_directories("${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-x64-1.12.0/include")
find_package(PythonInterp 3.8 EXACT REQUIRED COMPONENTS Development)
find_package(PythonLibs 3.8 EXACT REQUIRED COMPONENTS Development)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenCV REQUIRED)
# set(ONNXRUNTIME_ROOT "/home/manish/onnxruntime")
# find_package(ONNXRuntime REQUIRED)


set(Python_INCLUDE_DIRS "/usr/include/python3.8")
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)
include_directories(${Python_INCLUDE_DIRS})
include_directories("${ONNXRUNTIME_ROOTDIR}/include")

# pybind11_add_module(your_app src/main.cpp)
# target_link_libraries(your_app PUBLIC /home/manish/onnxruntime-linux-x64-1.12.0/lib/libonnxruntime.so ${OpenCV_LIBS} )


# add_executable(your_app1 src/main2.cpp)
# target_link_libraries(your_app1 PUBLIC /home/manish/onnxruntime-linux-x64-1.17.1/lib/libonnxruntime.so ${OpenCV_LIBS} )

pybind11_add_module(run_yolo_onnx src/run.cpp)
# add_executable(run_yolo_onnx src/main3.cpp)
target_link_libraries(run_yolo_onnx PRIVATE "${ONNXRUNTIME_ROOTDIR}/lib/libonnxruntime.so" ${OpenCV_LIBS} )

# python_add_library(your_app src/main.cpp)
# add_executable(your_app src/main.cpp)
# target_include_directories(your_app PUBLIC include /usr/include/python3.8 /usr/local/lib/python3.8/dist-packages/pybind11/include)
#     "${ONNXRUNTIME_ROOTDIR}/include"                           # Pre-built package
#                     "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime"               # Linux local install to /usr/local
#                     "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session")
# target_link_libraries(your_app PUBLIC "${ONNXRUNTIME_ROOTDIR}/lib/libonnxruntime.so.1.17.1" ${OpenCV_LIBS}  /usr/lib/x86_64-linux-gnu/libpython3.8.so)

#  message("!!! ${Python_INCLUDE_DIRS}")

# target_include_directories(your_app PRIVATE  {Python_INCLUDE_DIRS} {Pybind11_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
# set_target_properties(your_app PROPERTIES
#                                 INTERPROCEDURAL_OPTIMIZATION ON
#                                 VISIBILITY_INLINES_HIDDEN ON)
