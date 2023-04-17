
if (USE_CUDA)
    aux_source_directory(. DIR_GPU_SRCS)
    list(REMOVE_ITEM DIR_GPU_SRCS "./OTFonGPU.cu")
    set(DIR_CSG_INCS "include/gpu/GPUQuery.h" "include/gpu/GPUSolver.h" "include/gpu/clone.h"  )

    remove_definitions(-fopenmp)
    # 生成链接库
    add_library (${moc_lib_acc} ${DIR_GPU_SRCS})
    set_target_properties(${moc_lib_acc} PROPERTIES CUDA_ARCHITECTURES "35;50;72")
    target_link_libraries(${moc_lib_acc} PUBLIC ${MOC_LIBS})
    target_include_directories(${moc_lib_acc} PUBLIC ${MOC_INCLUDE_DIR})
endif()

if (USE_ROCM)
    set(DIR_GPU_SRCS "src/gpu/GPUQuery.cu" "src/gpu/GPUSolver.cu" "src/gpu/clone.cu"  )

    set(DIR_CSG_INCS "include/gpu/GPUQuery.h" "include/gpu/GPUSolver.h" "include/gpu/clone.h"  )

    hipify_source_files(DIR_DCU_SRCS  ${DIR_GPU_SRCS})
    hipify_header_files(DIR_DCU_INCS  ${DIR_CSG_INCS})
     
    add_custom_target(include_dep DEPENDS ${DIR_DCU_INCS})

    set_source_files_properties(${DIR_DCU_SRCS} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
    hip_add_library(${moc_lib_acc} STATIC ${DIR_DCU_SRCS} ${DIR_DCU_INCS}  )
    add_dependencies(${moc_lib_acc} include_dep)
    target_include_directories(${moc_lib_acc} PUBLIC  ${MOC_INCLUDE_DIR}  )
    target_link_libraries(${moc_lib_acc} PUBLIC ${MOC_LIBS} )
endif()

set( MOC_LIBS  ${MOC_LIBS} ${moc_lib_acc} PARENT_SCOPE)