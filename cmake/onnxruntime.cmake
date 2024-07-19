
find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS /usr/local/include/onnxruntime/core/session/)
find_path(ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS cuda_provider_factory.h HINTS /usr/local/include/onnxruntime/core/providers/cuda/)
find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /usr/local/lib)

message("lib : ${ONNX_RUNTIME_LIB}")

include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS})
include_directories(${PROJECT_NAME} /usr/local/include/onnxruntime)

target_compile_definitions(${PROJECT_NAME} PRIVATE -DENABLE_ORT)
target_link_libraries(${PROJECT_NAME} PUBLIC ${ONNX_RUNTIME_LIB})
