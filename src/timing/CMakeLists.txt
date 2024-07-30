add_library(timing timer_bank.cpp device_timer.cpp)
target_link_libraries(timing PUBLIC gpu-api)
target_include_directories(timing PUBLIC ${CMAKE_CURRENT_LIST_DIR})
