#include "problemset.h"
#include <planning_system.h>
#include <iostream>
#include <random>

using namespace rtat;

class Random_Problem_Generator {
  size_t footprint_lb;
  size_t footprint_ub;
  double aspect_max = 8.0;

  std::mt19937 rng;

  struct Aspect_Dims {
    double aspect1, aspect2;

    std::tuple<size_t, size_t, size_t> get_dims(size_t N) {
      size_t m = N;
      size_t k = (size_t)(m*aspect1);
      size_t n = (size_t)(k*aspect2);
      return std::make_tuple(m, k, n);
    }

    double calculate_flops(size_t N) {
      auto [m,k,n] = get_dims(N);
      return ((size_t)2)*m*k*n;
    }

    size_t calculate_footprint(size_t N) {
      auto [m,k,n] = get_dims(N);
      return ((size_t)8)*(m*k+k*n+m*n);
    }

    size_t N_from_flops(double flopcount) {
      return std::cbrt(flopcount/(2*aspect1*aspect1*aspect2));
    }

    size_t N_from_footprint(size_t footprint) {
      // footprint = 8*(N*N*a1 + N*a1*N*a1*a2 + N*N*a1*a2)
      //           = 8*a1*N^2*(1+a1*a2+a2)
      return (size_t)std::sqrt((footprint/(8*aspect1*(1 + aspect1*aspect2 + aspect2))));
    }


    double generate_aspect(double max, std::mt19937 &gen) {
      std::uniform_real_distribution<double> real_dist(1.0, max);
      std::uniform_int_distribution<> bool_dist(0,1);
      double aspect = real_dist(gen);
      if (bool_dist(gen)) 
        aspect = 1/aspect;
      return aspect;
    }

    Aspect_Dims(double max, std::mt19937 &gen) : aspect1(generate_aspect(max, gen)),
                                                 aspect2(generate_aspect(max, gen)) {}

  };



public:
  Random_Problem_Generator(int seed) : rng(seed) {
    size_t total, free;
    if (cudaMemGetInfo(&free, &total)) {
      std::cout << "NO CONNECTED GPU" << std::endl;
      throw;
    }

    footprint_lb = (size_t)(free*0.01);
    footprint_ub = (size_t)(free*0.05);
  }

  Random_Problem_Generator() : Random_Problem_Generator(std::mt19937::default_seed) {}

  Problem generate() {
    Aspect_Dims aspects(aspect_max, rng);

    size_t n_lb = aspects.N_from_footprint(footprint_lb);
    size_t n_ub = aspects.N_from_footprint(footprint_ub);

    double flop_lb = aspects.calculate_flops(n_lb);
    double flop_ub = aspects.calculate_flops(n_ub);

    std::uniform_real_distribution dist(flop_lb, flop_ub);
    double flopcount = dist(rng);
    size_t N = aspects.N_from_flops(flopcount);
    auto [m,k,n] = aspects.get_dims(N);

    std::uniform_int_distribution<> bool_dist(0,1);
    cublasOperation_t opA = bool_dist(rng) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t opB = bool_dist(rng) ? CUBLAS_OP_N : CUBLAS_OP_T;

    return Problem(m, k, n, opA, opB);
  }
};

//Problem_Set generate_problems

int main(int argc, char *argv[]) {
  if (argc != 2) { 
    std::cout << "Expected one command line arg: num_problems" << std::endl;
    return 1;
  }

  int num_problems = atoi(argv[1]);

  Random_Problem_Generator gen;
  for (int i = 0; i < num_problems; i++) {
    auto problem = gen.generate();
    problem.opA = CUBLAS_OP_N; problem.opB = CUBLAS_OP_N;
    std::cout << problem << std::endl;
    problem.opA = CUBLAS_OP_N; problem.opB = CUBLAS_OP_T;
    std::cout << problem << std::endl;
    problem.opA = CUBLAS_OP_T; problem.opB = CUBLAS_OP_N;
    std::cout << problem << std::endl;
    problem.opA = CUBLAS_OP_T; problem.opB = CUBLAS_OP_T;
    std::cout << problem << std::endl;
  }
}
