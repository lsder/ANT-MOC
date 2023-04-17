add_subdirectory(moc)
add_subdirectory(gpu)
if (USE_ROCM)
    # add the executable
    message("${MOC_LIBS}")
    set_source_files_properties(newmoc.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
    hip_add_executable(newmoc newmoc.cpp)
    set_property(TARGET newmoc PROPERTY HIP_ARCHITECTURES gfx906)
    target_link_libraries(newmoc PUBLIC ${MOC_LIBS}   )
    target_include_directories(newmoc PUBLIC ${MOC_INCLUDE_DIR} )

elseif (USE_CUDA)
    # add the executable
    add_executable(newmoc newmoc.cpp)
    target_link_libraries(newmoc PUBLIC ${MOC_LIBS} )
    target_include_directories(newmoc PUBLIC ${PROJECT_BINARY_DIR} ${MOC_INCLUDE_DIR} )
else()
    add_executable(newmoc newmoc.cpp )
    target_link_libraries(newmoc PUBLIC ${MOC_LIBS} )
    target_include_directories(newmoc PUBLIC ${PROJECT_BINARY_DIR} ${MOC_INCLUDE_DIR} ${DIR_CSG_INCS})
endif()
 
