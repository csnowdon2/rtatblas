#include <algorithm>
#include <gtest/gtest.h>
#include <set>
#include <iostream>
#include "../src/plan.h"
#include "../src/planning_system.h"
#include <string>


using Test_Op1 = Option<int, 16, 32, 64>;
using Test_Op2 = Option<bool, true, false>;
using Test_Op3 = Option<char, 'a', 'b', 'c'>;

using Test_Ops = Options<Test_Op1, Test_Op2, Test_Op3>;

TEST(Plan_Test, Enumerate) {
  std::set<Test_Ops> enum_ops;
  for (auto &op : Test_Ops::enumerate())
    enum_ops.insert(op);

  std::set<Test_Ops> expected_ops;
  expected_ops.emplace(16, true , 'a');
  expected_ops.emplace(16, true , 'b');
  expected_ops.emplace(16, true , 'c');
  expected_ops.emplace(16, false, 'a');
  expected_ops.emplace(16, false, 'b');
  expected_ops.emplace(16, false, 'c');
  expected_ops.emplace(32, true , 'a');
  expected_ops.emplace(32, true , 'b');
  expected_ops.emplace(32, true , 'c');
  expected_ops.emplace(32, false, 'a');
  expected_ops.emplace(32, false, 'b');
  expected_ops.emplace(32, false, 'c');
  expected_ops.emplace(64, true , 'a');
  expected_ops.emplace(64, true , 'b');
  expected_ops.emplace(64, true , 'c');
  expected_ops.emplace(64, false, 'a');
  expected_ops.emplace(64, false, 'b');
  expected_ops.emplace(64, false, 'c');
  expected_ops.emplace(65, false, 'c');

  std::set<Test_Ops> intersection;
  std::set_intersection(expected_ops.begin(), expected_ops.end(), 
                        enum_ops.begin(),     enum_ops.end(), 
                        std::inserter(intersection, intersection.begin()));

  ASSERT_EQ(enum_ops.size(), intersection.size());
}
