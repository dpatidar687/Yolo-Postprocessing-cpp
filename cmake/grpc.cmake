file(GLOB grpc_path "/usr/local/lib/grpc/*.so")
target_include_directories(${PROJECT_NAME} PUBLIC "/usr/local/include/grpc/")
target_compile_definitions(${PROJECT_NAME} PRIVATE -DENABLE_GRPC)
link_libraries(${PROJECT_NAME} ${grpc_path})
