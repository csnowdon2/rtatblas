add_subdirectory(utils)
add_subdirectory(matrix_ops)
add_subdirectory(gpu-api)

add_library(rtatblas INTERFACE)
target_link_libraries(rtatblas INTERFACE utils matrix_ops nlohmann_json::nlohmann_json)
target_include_directories(rtatblas INTERFACE . utils matrix_ops)

add_subdirectory(app)
