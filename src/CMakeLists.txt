add_subdirectory(gpu-api)
add_subdirectory(timing)
add_subdirectory(matrix_ops)
add_subdirectory(methods)
add_subdirectory(planning)

target_include_directories(rtatblas INTERFACE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(rtatblas INTERFACE gpu-api timing matrix_ops methods planning nlohmann_json::nlohmann_json)

if (NOT DEFINED RTATBLAS_BUILD_APP)
    set(RTATBLAS_BUILD_APP ON)
endif()

if (RTATBLAS_BUILD_APP)
    add_subdirectory(app)
endif()
