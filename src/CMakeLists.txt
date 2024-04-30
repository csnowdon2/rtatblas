add_subdirectory(gpu-api)
add_subdirectory(timing)
add_subdirectory(matrix_ops)

add_library(rtatblas INTERFACE)
target_link_libraries(rtatblas INTERFACE timing matrix_ops nlohmann_json::nlohmann_json)
target_include_directories(rtatblas INTERFACE ${CMAKE_CURRENT_LIST_DIR} timing matrix_ops)

#add_subdirectory(app)
