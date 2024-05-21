add_library(methods gemm.cpp trsm.cpp)
target_link_libraries(methods PUBLIC gpu-api timing matrix_ops nlohmann_json::nlohmann_json)
target_include_directories(methods PUBLIC ${CMAKE_CURRENT_LIST_DIR})
