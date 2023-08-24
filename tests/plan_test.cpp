#include <gtest/gtest.h>
#include <iostream>
#include "../src/plan.h"
#include "../src/planning_system.h"
#include <string>


using Test_Op1 = Option<int, 16, 32, 64>;
using Test_Op2 = Option<bool, true, false>;
using Test_Op3 = Option<char, 'a', 'b', 'c'>;

using Test_Ops = Options<Test_Op1, Test_Op2, Test_Op3>;

TEST(Plan_Test, Enumerate) {
  auto op_enumeration = Test_Ops::enumerate();
  for (auto &op : op_enumeration) {
    std::cout << std::get<0>(op) << " ";
    std::cout << std::get<1>(op) << " ";
    std::cout << std::get<2>(op) << " ";
    std::cout << std::endl;
  }
  FAIL();
}
