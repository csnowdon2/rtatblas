add_library(utils rolling_average.cpp event_timer_buffer.cpp)
target_link_libraries(utils PUBLIC gpu_api)
target_include_directories(utils PUBLIC .)
target_include_directories(utils PUBLIC ${GPU_INCLUDE_DIRS})
