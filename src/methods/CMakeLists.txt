add_library(methods gemm.cpp)
target_link_libraries(methods PUBLIC gpu-api timing matrix_ops)
target_include_directories(methods PUBLIC ${CMAKE_CURRENT_LIST_DIR})
