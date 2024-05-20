#include "problemset.h"
#include "runner.h"
#include <iostream>

using namespace rtat;

template<typename T> 
void autotune(Problem &problem, int reps) {
  SmartRunner<T> runner;
  Problem_Set problems;

  problems.add_problem(problem);
  runner.run_problems(problems, reps);
  runner.sync();
  runner.json_output(std::cout);
}

int main(int argc, char *argv[]) {
  if (argc != 8) { 
    std::cout << "Expected command line args: precision m k n opA opB reps" << std::endl;
    return 1;
  }

  std::string precision(argv[1]);
  int m = atoi(argv[2]);
  int k = atoi(argv[3]);
  int n = atoi(argv[4]);
  cublasOperation_t opA = argv[5][0] == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t opB = argv[6][0] == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T;
  int reps = atoi(argv[7]);

  Problem problem(m,k,n,opA,opB);

  if (precision == "double") {
    autotune<double>(problem, reps);
  } else if (precision == "single") {
    autotune<float>(problem, reps);
  } else {
    throw("precision must be 'single' or 'double'");
  }
}
