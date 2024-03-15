set(CMAKE_CXX_STANDARD 17)

find_package(GTest)
if (NOT GTest_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    GTest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  )
  # For Windows: Prevent overriding the parent project's compiler/linker settings
  FetchContent_MakeAvailable(googletest)
endif()
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

include(GoogleTest)

add_executable(api_test api_test.cpp)
target_link_libraries(api_test gpu_api GTest::gtest_main)

add_executable(utils_test utils_test.cpp)
target_link_libraries(utils_test utils GTest::gtest_main)

add_executable(plan_test plan_test.cpp)
target_link_libraries(plan_test gpu_api utils GTest::gtest_main)

add_executable(matrix_ops_test matrixop_test.cpp)
target_link_libraries(matrix_ops_test gpu_api matrix_ops GTest::gtest_main)
target_compile_options(matrix_ops_test PUBLIC -g)

add_executable(planning_test planning_test.cpp)
target_link_libraries(planning_test utils gpu_api matrix_ops GTest::gtest_main)
gtest_discover_tests(api_test)
gtest_discover_tests(utils_test)
gtest_discover_tests(plan_test)
gtest_discover_tests(matrix_ops_test)
gtest_discover_tests(planning_test)

