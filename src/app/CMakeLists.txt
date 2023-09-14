add_executable(measure_performance measure_performance.cpp)
target_link_libraries(measure_performance PUBLIC rtatblas)

add_executable(run_tests run_tests.cpp)
target_link_libraries(run_tests PUBLIC rtatblas)

add_executable(generate_input generate_input.cpp)
target_link_libraries(generate_input PUBLIC rtatblas)

add_executable(autotune autotune.cpp)
target_link_libraries(autotune PUBLIC rtatblas)
