cmake_minimum_required(VERSION 3.23)
project(RTATBLAS)


#enable_language(CUDA)
find_package(CUDA REQUIRED)
message(${CUDA_LIBRARIES})

add_subdirectory(src)
enable_testing()
add_subdirectory(tests)
