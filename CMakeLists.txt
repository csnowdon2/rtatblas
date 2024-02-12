cmake_minimum_required(VERSION 3.21)
set (CMAKE_CXX_STANDARD 17)
project(RTATBLAS)

# GPU libraries
find_package(CUDA)
if (NOT CUDA_FOUND)
  find_package(HIP REQUIRED)
  find_package(hipBLAS REQUIRED)
  find_package(rocBLAS REQUIRED)
  set(GPU_LIBRARIES ${HIP_LIBRARIES})
  set(GPU_LIBRARIES ${HIPBLAS_LIBRARIES})
  set(GPU_INCLUDE_DIRS ${HIP_INCLUDE_DIRS})
  add_definitions(-DHIP)
else()
  set(GPU_LIBRARIES ${CUDA_LIBRARIES} -lcublas)
  set(GPU_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
  add_definitions(-DCUDA)
endif()
message("GPU LIBS " ${GPU_LIBRARIES})

# JSON
include(FetchContent)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz)
FetchContent_MakeAvailable(json)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror")

add_subdirectory(src)
enable_testing()
add_subdirectory(tests)
