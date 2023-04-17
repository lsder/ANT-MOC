list(APPEND EXTRA_LIBS ${MOC_LIBS}     )
list(APPEND EXTRA_INCLUDES ${MOC_INCLUDE_DIR}  )



if (USE_ROCM)
set_source_files_properties(gpu_opt_test.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(gpu_opt_test gpu_opt_test.cpp)
target_link_libraries(gpu_opt_test PUBLIC ${EXTRA_LIBS} )
target_include_directories(gpu_opt_test PUBLIC ${MOC_INCLUDE_DIR} )

set_source_files_properties(track_geo_gen.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(track_geo_gen track_geo_gen.cpp)
target_link_libraries(track_geo_gen PUBLIC ${EXTRA_LIBS} )
target_include_directories(track_geo_gen PUBLIC ${MOC_INCLUDE_DIR} )
endif()
if (USE_CUDA)
include_directories(include ${Boost_INCLUDE_DIRS}/include/csg)
add_executable(Point_test Point_test.cpp)
target_link_libraries(Point_test PUBLIC ${EXTRA_LIBS}  )
target_link_libraries(Point_test PUBLIC  "${PROJECT_BINARY_DIR}" ${EXTRA_INCLUDES})

add_executable(log_test log_test.cpp)
target_link_libraries(log_test PUBLIC ${EXTRA_LIBS} OpenMP::OpenMP_CXX )
target_link_libraries(log_test PUBLIC  "${PROJECT_BINARY_DIR}" ${EXTRA_INCLUDES})

add_executable(timer_test timer_test.cpp)
target_link_libraries(timer_test PUBLIC ${EXTRA_LIBS} OpenMP::OpenMP_CXX )
target_link_libraries(timer_test PUBLIC  "${PROJECT_BINARY_DIR}" ${EXTRA_INCLUDES})

endif()