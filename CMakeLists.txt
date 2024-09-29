cmake_minimum_required(VERSION 3.21)
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()
set (CMAKE_CXX_STANDARD 17)
project(RTATBLAS)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_library(rtatblas INTERFACE)

# GPU libraries
find_package(CUDAToolkit)

if (DEFINED CMAKE_HIP_PLATFORM OR NOT CUDAToolkit_FOUND)
  find_package(HIP REQUIRED)
  find_package(hipBLAS REQUIRED)
  find_package(hiprand REQUIRED)
  find_package(rocBLAS REQUIRED)
  set(GPU_LIBRARIES hip::host roc::rocblas roc::hipblas hip::hiprand)
  set(_RTAT_HIP 1)
else()
  set(GPU_LIBRARIES CUDA::cudart CUDA::cublas CUDA::curand)
  set(_RTAT_CUDA 1)
endif()
message("GPU LIBS " ${GPU_LIBRARIES})

# JSON
find_package(nlohmann_json 3.11.3 QUIET)
if  (NOT nlohmann_json_FOUND)
  include(FetchContent)

  FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
  FetchContent_MakeAvailable(json)
endif()

# Check if the project is being added as a submodule
if (NOT ${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
  #inherit the cmake cxx flags from the parent project
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ")
else()
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g -Wall -Wextra -Wpedantic -Werror")
endif()

add_subdirectory(src)

if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

install(TARGETS rtatblas RUNTIME DESTINATION lib)
#install(TARGETS autotune RUNTIME DESTINATION bin)
