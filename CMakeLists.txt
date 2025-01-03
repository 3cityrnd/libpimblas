cmake_minimum_required(VERSION 3.13)
project(pimblas) 
set(PIM_BLAS_VERSION 0.1)

#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(BUILD_TESTS "Build tests" ON)
option(ADD_GTEST_LIB  "ADD GOOGLE TEST" OFF)

option(USE_SDK_HOST_CXX "USE UPMEME SDK COMPILERS" ON)
option(LOGGING "Enabled logging spdlog" ON)


if(NOT DEFINED NR_TASKLETS)
  set(NR_TASKLETS "16")  # optimal usage
endif()



set(CND_HOME $ENV{CONDA_PREFIX})
if(NOT CND_HOME)
message(FATAL_ERROR "Conda is not installed")
endif()

find_library(FOUND_PYT36 NAMES python3.6m PATHS "${CND_HOME}/lib" )
if(NOT FOUND_PYT36)
    message(FATAL_ERROR "Can't find python3.6m ${CND_HOME}/lib")
 endif()


 set(GEHD_SCRIPT "${CMAKE_SOURCE_DIR}/bin/genhd.sh")
 include(FetchContent)
 

set(PIMBLAS_DEFAULT_KERENEL_DIR ${CMAKE_BINARY_DIR}/kernels)

message(STATUS  "Set Default Pimblas kernel dir ${PIMBLAS_DEFAULT_KERENEL_DIR}")

set(UPH $ENV{UPMEM_HOME})
if(NOT UPH)
   message(FATAL_ERROR "UPMEM_HOME is not detected https://sdk.upmem.com/")
endif()

message(STATUS "UPMEM sdk detected in [${UPH}]")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON) 
set(CMAKE_CXX_EXTENSIONS OFF)


set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)


if(LOGGING)
include("cmake/spdlog.cmake")
add_definitions(-DLOGGING)
endif()


if(USE_SDK_HOST_CXX)
#  set(CMAKE_C_COMPILER "${UPH}/bin/clang" CACHE STRING "Custom C compiler" FORCE)
#  set(CMAKE_CXX_COMPILER "${UPH}/bin/clang++" CACHE STRING "Custom C++ compiler" FORCE )
  set(CMAKE_C_COMPILER "${UPH}/bin/clang" )
  set(CMAKE_CXX_COMPILER "${UPH}/bin/clang++")

 endif()

if(ADD_GTEST_LIB)
   include("cmake/gtest.cmake")
   add_definitions(-DADD_GTEST_LIB)
endif()


message(STATUS "CMAKE_C_COMPILER  [${CMAKE_C_COMPILER}]")
message(STATUS "CMAKE_CXX_COMPILER [${CMAKE_CXX_COMPILER}]")

add_subdirectory(src)

if(BUILD_TESTS)
   message(STATUS "Build tests !")
   enable_testing()
   add_subdirectory(tests)
endif()

