#include "problemset.h"
#include "runner.h"
#include <sstream>
#include <iostream>

int main(int argc, char *argv[]) {
  if (argc < 2 || argc > 3) { 
    std::cout << "Expected command line args: filename [reps]" << std::endl;
    return 1;
  }

  std::string filename(argv[1]);
  int reps = 10;
  if (argc >= 3)
    reps = atoi(argv[2]);

  std::cout << "Running file " << filename << " with " << reps << " reps" << std::endl;
  SmartRunner runner;
  Problem_Set problems(filename);
  // TODO check for duplicate dimensions when using smart measurement
  runner.run_problems(problems, reps);
  runner.json_output(filename + ".o");
}
