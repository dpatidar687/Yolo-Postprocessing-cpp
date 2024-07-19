find_package(OpenVINO REQUIRED)
include_directories(/opt/intel/openvino/runtime/include)
target_link_libraries(${PROJECT_NAME} PRIVATE openvino::runtime)
target_compile_definitions(${PROJECT_NAME} PRIVATE -DENABLE_VINO)

