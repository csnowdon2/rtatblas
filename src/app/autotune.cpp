#include "problemset.h"
#include "runner.h"
#include <iostream>

int main(int argc, char *argv[]) {
  if (argc != 7) { 
    std::cout << "Expected command line args: m k n opA opB reps" << std::endl;
    return 1;
  }

  int m = atoi(argv[1]);
  int k = atoi(argv[2]);
  int n = atoi(argv[3]);
  cublasOperation_t opA = argv[4][0] == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t opB = argv[5][0] == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T;
  Problem p(m,k,n,opA,opB);

  int reps = atoi(argv[6]);

  Runner runner(true);

  Problem_Set problems;
  problems.add_problem(p);
  // TODO check for duplicate dimensions when using smart measurement
  runner.run_problems(problems, reps);
  runner.sync();
  runner.print_analytics();
}
