include(cmake/FindTensorRT.cmake)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.8)

target_link_libraries(${PROJECT_NAME} PUBLIC ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES})
target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
target_compile_definitions(${PROJECT_NAME} PRIVATE -DENABLE_TRT)
