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

option(BUILD_TORCH_CPU_CATCH_ALLOCATOR "Build torch_cpu_catch" ON)
option(BUILD_TORCH_CPU_BLAS_CATCH "Build torch_blas_catch" ON)


if(NOT DEFINED NR_TASKLETS)
  set(NR_TASKLETS "16")  # optimal usage
endif()



# set(CND_HOME $ENV{CONDA_PREFIX})
# if(NOT CND_HOME)
# message(FATAL_ERROR "Conda is not installed")
# endif()

# find_library(FOUND_PYT36 NAMES python3.6m PATHS "${CND_HOME}/lib" )
# if(NOT FOUND_PYT36)
#     message(FATAL_ERROR "Can't find python3.6m ${CND_HOME}/lib")
#  endif()


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


# Override UPMEM std
set(SEARCH_LIBSTDCXX_PATHS
    /lib/x86_64-linux-gnu/
    /lib64/
)

find_library(LIBSTDCXX_PATH
    NAMES libstdc++.so.6
    PATHS ${SEARCH_LIBSTDCXX_PATHS}
    NO_DEFAULT_PATH
)

if(LIBSTDCXX_PATH)
    get_filename_component(LIBSTDCXX_DIR ${LIBSTDCXX_PATH} DIRECTORY)
    message(STATUS "Found libstdc++ in: ${LIBSTDCXX_DIR}")
else()
    message(FATAL_ERROR "libstdc++ not found in provided paths.")
endif()

if(LOGGING)
message(STATUS "Downloading spdlog ... ")
include("cmake/spdlog.cmake")
add_definitions(-DLOGGING)
endif()


if(BUILD_TORCH_CPU_CATCH_ALLOCATOR)
add_definitions(-DTORCH_CPU_CATCH_ALLOCATOR)
endif()

if(BUILD_TORCH_CPU_BLAS_CATCH)
add_definitions(-DTORCH_CPU_BLAS_CATCH)
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

set(CLANG_FORMAT "${CMAKE_CURRENT_LIST_DIR}/bin/clang-format")


file(GLOB_RECURSE CLANG_FILES_UPDATE
    "${CMAKE_CURRENT_LIST_DIR}/src/*.c"
    "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/*.h"
    "${CMAKE_CURRENT_LIST_DIR}/src/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/*.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/tests/*.c"
    "${CMAKE_CURRENT_LIST_DIR}/tests/*.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/tests/*.h"
    "${CMAKE_CURRENT_LIST_DIR}/tests/*.hpp"
    
)



add_custom_target(format
    COMMAND ${CLANG_FORMAT} -i ${CLANG_FILES_UPDATE}
    COMMENT "Running clang-format on source files"
)