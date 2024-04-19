add_library(timing timer_bank.cpp)
target_link_libraries(timing PUBLIC utils)
target_include_directories(timing PUBLIC . ..)
