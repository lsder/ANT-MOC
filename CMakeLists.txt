cmake_minimum_required(VERSION 3.0)
project(NewMOC LANGUAGES CXX )
set(CMAKE_CXX_STANDARD 14)

option(USE_MPI                 "Enable MPI routines in antmoc"     ON)
option(USE_OPENMP              "Enable MPI routines in antmoc"     ON)
option(USE_ROCM                "Enable ROCM solver"                ON)
option(USE_CUDA                "Enable CUDA solver"                OFF)

set(CMAKE_BUILD_TYPE "Debug")
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/run )
set( CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

include(yaml-cpp)

set(MOC_LIBS ${MOC_LIBS} ${yaml-cpp_LIB})
set(MOC_INCLUDE_DIR ${MOC_INCLUDE_DIR}  ${yaml-cpp_INCLUDE_DIR}  )
# ---[   find_package  MPI
if (USE_MPI)
    find_package(MPI)
    add_definitions(-DMPIx)
    set( MOC_INCLUDE_DIR ${MOC_INCLUDE_DIR} ${MPI_INCLUDE_PATH} )
    set(MOC_LIBS ${MOC_LIBS} mpi_cxx mpi ${MPI_C_LIBRARIES})
endif()

if (USE_OPENMP)
    find_package(OpenMP)
    add_definitions( -DOPENMP  -fopenmp)
    set(MOC_LIBS ${MOC_LIBS} OpenMP::OpenMP_CXX)
endif()

# ---[   find_package  ROCM
if (USE_ROCM)
    include(HipBuildSystem)
    set(CMAKE_MODULE_PATH "$ENV{ROCM_PATH}/hip/cmake" ${CMAKE_MODULE_PATH})
    find_package(HIP   REQUIRED)
    add_definitions(-DROCM )
    set( MOC_INCLUDE_DIR ${MOC_INCLUDE_DIR} ${PROJECT_BINARY_DIR}/include/gpu ${PROJECT_SOURCE_DIR}/include/gpu )
    message(STATUS "Found HIP: " ${HIP_VERSION})
    find_package(ROCM REQUIRED)
    if(NOT ROCM_PATH OR ROCM_PATH STREQUAL "")
        if(DEFINED ENV{ROCM_PATH} AND NOT $ENV{ROCM_PATH} STREQUAL "")
            set(ROCM_PATH $ENV{ROCM_HOME})
            
        else()
            set(ROCM_PATH ${HIP_ROOT_DIR})
        endif()
    endif()
endif()

# ---[   find_package  CUDA
if (USE_CUDA)
    find_package(CUDA REQUIRED)
    add_definitions(-DNVCC )
    enable_language(CUDA)
    set( MOC_INCLUDE_DIR ${MOC_INCLUDE_DIR}  ${PROJECT_SOURCE_DIR}/include/gpu )
endif()

# ---[   find_package  test
# if (ENABLE_TEST)
#     include(GoogleTest)
#     enable_testing()
#     add_subdirectory("${PROJECT_SOURCE_DIR}/tests")
# endif()

add_definitions( -DFP_PRECISION=double -DCMFD_PRECISION=float -DLINALG_TOL=1E-7 -DVEC_ALIGNMENT=64 -DVEC_LENGTH=4 )

# ---[ Includes
set( MOC_INCLUDE_DIR ${MOC_INCLUDE_DIR} ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/include/moc  ${PROJECT_SOURCE_DIR}/include/gpu)
                    
# ---[ lib
set(moc_lib "libmoc")
set(moc_lib_acc "libmoc_acc")#加速卡
set(MOC_LIBS ${MOC_LIBS} )

add_subdirectory(include/gpu)# 就是为了建个文件夹
add_subdirectory(src)
# add_subdirectory(test)

