#include "problemset.h"
#include "runner.h"
#include <iostream>

using namespace rtat;

template<typename T> 
void exhaustive(Problem_Set &problems, int reps) {
  RoundRobinRunner<T> runner;
  runner.run_problems(problems, reps*GEMM_Options::enumerate().size());
  runner.sync();
  runner.json_output(std::cout);
}

int main(int argc, char *argv[]) {
  if (argc < 3 || argc > 4) { 
    std::cout << "Expected command line args: filename precision [reps]" << std::endl;
    return 1;
  }

  std::string filename(argv[1]);
  std::string precision(argv[2]);
  int reps = 10;
  if (argc >= 3)
    reps = atoi(argv[3]);

  std::cout << "Exhaustively running file " << filename << " in "
            <<  precision << " precision with " 
            << reps << " reps" << std::endl;

  Problem_Set problems(filename);
  if (precision == "double") {
    exhaustive<double>(problems, reps);
  } else if (precision == "single") {
    exhaustive<float>(problems, reps);
  } else {
    throw("precision must be 'single' or 'double'");
  }

  return 0;
}

