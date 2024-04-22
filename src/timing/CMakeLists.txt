add_library(timing timer_bank.cpp device_timer.cpp)
target_link_libraries(timing PUBLIC utils)
target_include_directories(timing PUBLIC . ..)
