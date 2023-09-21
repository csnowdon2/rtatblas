add_library(app_common INTERFACE runner.h problemset.h json_planner.h)
target_link_libraries(app_common INTERFACE rtatblas)

add_executable(measure_performance measure_performance.cpp)
target_link_libraries(measure_performance PUBLIC app_common)

add_executable(run_tests run_tests.cpp)
target_link_libraries(run_tests PUBLIC app_common)

add_executable(run_tests_exhaustive run_tests_exhaustive.cpp)
target_link_libraries(run_tests_exhaustive PUBLIC app_common)

add_executable(generate_input generate_input.cpp)
target_link_libraries(generate_input PUBLIC app_common)

add_executable(autotune autotune.cpp)
target_link_libraries(autotune PUBLIC app_common)
