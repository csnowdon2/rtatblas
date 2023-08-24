add_library(matrix_ops INTERFACE)
target_include_directories(matrix_ops INTERFACE .)
target_link_libraries(matrix_ops INTERFACE ${CUDA_LIBRARIES})
target_include_directories(matrix_ops INTERFACE ${CUDA_INCLUDE_DIRS})

