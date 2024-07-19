find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
message(STATUS "VERBOSE IS ON")
message(${GSTREAMER_LIBRARIES})

if(GSTREAMER_INCLUDE_DIRS)
else(GSTREAMER_INCLUDE_DIRS)
        message(STATUS "GStreamer Includes Not Found")
endif(GSTREAMER_INCLUDE_DIRS)

if(GSTREAMER_LIBRARIES)
        message(${GSTREAMER_LIBRARIES})
else(GSTREAMER_LIBRARIES)
        message(STATUS "GStreamer Libraries Not Found")
endif(GSTREAMER_LIBRARIES)

include_directories(${GSTREAMER_INCLUDE_DIRS} ${GSTREAMER-BASE_INCLUDE_DIRS})

include_directories(/usr/include/gstreamer-1.0 /usr/lib/x86_64-linux-gnu/gstreamer-1.0/include /usr/include/glib-2.0 /usr/include/libxml2 /usr/lib/x86_64-linux-gnu/glib-2.0/include/)

include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${GSTREAMER_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${GSTREAMER_LIBRARIES} )
