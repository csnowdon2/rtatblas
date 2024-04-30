add_library(matrix_ops INTERFACE)
target_include_directories(matrix_ops INTERFACE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(matrix_ops INTERFACE gpu-api)

