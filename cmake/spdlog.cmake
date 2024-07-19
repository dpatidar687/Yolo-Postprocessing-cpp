add_subdirectory(${CMAKE_SOURCE_DIR}/ext/spdlog)
#include_directories(${CMAKE_SOURCE_DIR}/../../ext/spdlog)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/ext/spdlog/include/)
target_link_libraries(${PROJECT_NAME} PUBLIC spdlog)
