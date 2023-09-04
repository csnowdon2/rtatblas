add_library(gpu_api gpu-api.cpp)
target_include_directories(gpu_api PUBLIC .)
target_link_libraries(gpu_api PUBLIC ${GPU_LIBRARIES})
target_include_directories(gpu_api PUBLIC ${GPU_INCLUDE_DIRS})
