set(CMAKE_CXX_STANDARD 17)

find_package(GTest QUIET)
if (NOT GTest_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  )
  # For Windows: Prevent overriding the parent project's compiler/linker settings
  FetchContent_MakeAvailable(googletest)
endif()
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

include(GoogleTest)

add_executable(api_test api_test.cpp)
target_link_libraries(api_test gpu-api GTest::gtest_main)

add_executable(timing_test timing_test.cpp)
target_link_libraries(timing_test timing GTest::gtest_main)

add_executable(plan_test plan_test.cpp)
target_link_libraries(plan_test gpu-api timing GTest::gtest_main)

add_executable(matrix_ops_test matrixop_test.cpp)
target_link_libraries(matrix_ops_test gpu-api matrix_ops GTest::gtest_main)

add_executable(executor_test executor_test.cpp)
target_link_libraries(executor_test methods GTest::gtest_main)

add_executable(planning_test planning_test.cpp)
target_link_libraries(planning_test rtatblas GTest::gtest_main)
gtest_discover_tests(api_test)
gtest_discover_tests(timing_test)
gtest_discover_tests(plan_test)
gtest_discover_tests(matrix_ops_test)
gtest_discover_tests(planning_test)
gtest_discover_tests(executor_test)

