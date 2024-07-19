include_directories(/usr/include/eigen3)
# target_link_libraries(${PROJECT} /usr/include/eigen3)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fno-math-errno -mfma -DNDEBUG -mavx")
message(STATUS "Eigen3 found")
