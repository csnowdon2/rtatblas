add_library(gpu-api gpu-api.cpp)
target_link_libraries(gpu-api PUBLIC ${GPU_LIBRARIES})
target_include_directories(gpu-api PUBLIC ${CMAKE_CURRENT_LIST_DIR} ${GPU_INCLUDE_DIRS})
