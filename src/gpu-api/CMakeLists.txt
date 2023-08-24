add_library(gpu_api gpu-api.cpp)
target_include_directories(gpu_api PUBLIC .)
target_link_libraries(gpu_api PUBLIC ${CUDA_LIBRARIES} -lcublas)
target_include_directories(gpu_api PUBLIC ${CUDA_INCLUDE_DIRS})
