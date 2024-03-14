cmake_minimum_required(VERSION 3.21)
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()
set (CMAKE_CXX_STANDARD 17)
project(RTATBLAS)

# GPU libraries
find_package(CUDAToolkit)

if (NOT CUDAToolkit_FOUND)
  find_package(HIP REQUIRED)
  find_package(hipBLAS REQUIRED)
  find_package(rocBLAS REQUIRED)
  set(GPU_LIBRARIES hip::host roc::rocblas roc::hipblas)
  add_definitions(-DHIP)
else()
  set(GPU_LIBRARIES CUDA::cudart CUDA::cublas)
  add_definitions(-DCUDA)
endif()
message("GPU LIBS " ${GPU_LIBRARIES})

# JSON
find_package(nlohmann_json 3.11.3)
if  (NOT nlohmann_json_FOUND)
  include(FetchContent)

  FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
  FetchContent_MakeAvailable(json)
endif()

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror")

add_subdirectory(src)
enable_testing()
add_subdirectory(tests)

install(TARGETS rtatblas RUNTIME DESTINATION lib)
install(TARGETS autotune RUNTIME DESTINATION bin)
