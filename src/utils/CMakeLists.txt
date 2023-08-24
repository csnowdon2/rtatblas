add_library(utils rolling_average.cpp event_timer_buffer.cpp performance_record.cpp)
target_link_libraries(utils PUBLIC gpu_api)
target_include_directories(utils PUBLIC .)
target_include_directories(utils PUBLIC ${CUDA_INCLUDE_DIRS})
