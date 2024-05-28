#include "test_harness.h"
#include <iostream>
#include <nlohmann/json.hpp>
#include <planning_system.h>
#include <fstream>
#include <gemm.h>
#include <trsm.h>
#include <syrk.h>

using namespace rtat;

template<typename Executor>
nlohmann::json dispatch_tests(Input_File &file) {
  Test_Harness<Planning_System<Executor>> harness(file.problem_json);

  switch (file.run_type.val) {
    case Run_Type::EXHAUSTIVE:
      return harness.run_exhaustive(file.repetitions);

    case Run_Type::AUTOTUNE:
      return harness.run_autotune(file.repetitions);
  }
}

template<template <typename> typename Executor>
nlohmann::json dispatch_tests(Input_File& file) {
  switch (file.data_type.val) {
    case Data_Type::FLOAT:
      return dispatch_tests<Executor<float>>(file);

    case Data_Type::DOUBLE:
      return dispatch_tests<Executor<double>>(file);
  }
}

nlohmann::json dispatch_tests(Input_File& file) {
  switch (file.method.val) {
    case Method::GEMM:
      return dispatch_tests<GEMM_Executor>(file);

    case Method::GEMM_PAD:
      return dispatch_tests<GEMM_Executor_Pad>(file);

    case Method::SYRK:
      return dispatch_tests<SYRK_Executor>(file);

    case Method::TRSM:
      return dispatch_tests<TRSM_Executor>(file);
  }
}

int main(int argc, char *argv[]) {
  if (argc != 2) { 
    std::cout << "Expected command line args: filename" << std::endl;
    return 1;
  }

  std::string filename(argv[1]);

  nlohmann::json input_json 
    = nlohmann::json::parse(std::ifstream(filename));

  Input_File input_file(input_json);
  nlohmann::json output_json;
  output_json["problems"] = dispatch_tests(input_file);
  output_json["keywords"] = input_json["keywords"];
  std::cout << output_json.dump(2) << std::endl;

  return 0;
}
