add_subdirectory(utils)
add_subdirectory(matrix_ops)
add_subdirectory(gpu-api)

add_library(rtatblas INTERFACE)
target_link_libraries(rtatblas INTERFACE utils matrix_ops)
target_include_directories(rtatblas INTERFACE . utils matrix_ops)

add_subdirectory(app)
