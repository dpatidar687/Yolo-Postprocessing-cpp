set (LIB_INSTALL_DIR "/opt/nvidia/deepstream/deepstream/lib/")
#file(GLOB NVSHARED ${LIB_INSTALL_DIR}*.so)
#message(${NVSHARED})
target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE "/opt/nvidia/deepstream/deepstream/sources/includes")
target_compile_definitions(${PROJECT_NAME} PRIVATE -DENABLE_DS)
#target_link_libraries(${PROJECT_NAME} PRIVATE ${NVSHARED} nvds_meta nvdsgst_meta )
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIB_INSTALL_DIR}/libnvdsgst_meta.so
    ${LIB_INSTALL_DIR}/libnvbufsurface.so
    ${LIB_INSTALL_DIR}/libnvbuf_fdmap.so
    ${LIB_INSTALL_DIR}/libnvds_meta.so )
    #nvds_meta nvdsgst_meta )
