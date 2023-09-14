#include "problemset.h"
#include <planning_system.h>
#include <iostream>
#include <stack>
#include <variant>

class GPU_Stack_Buffer {
  std::stack<size_t> stack;

  size_t size;
  double *ptr;

public:
  GPU_Stack_Buffer(size_t size) : size(size) { gpuAssert(cudaMalloc(&ptr, size)); }
  ~GPU_Stack_Buffer() { gpuAssert(cudaFree(ptr)); }

  void pop() { stack.pop(); }

  double* alloc(size_t doubles) {
    doubles = (doubles/32)*32;

    size_t pos = (stack.size() == 0) ? 0 : stack.top();
    if ((pos+doubles)*sizeof(double) > size) {
      std::cout << "OVER-ALLOCATION" << std::endl;
      throw;
    }

    stack.push(pos+doubles);

    return &ptr[pos];
  }

  Matrix allocate_matrix(int m, int n) {
    size_t size = (size_t)(m)*(size_t)(n);
    Workspace space(alloc(size), size);

    Matrix A(space, m, n, m);
    return A;
  }

  size_t remaining_space() {
    if (size <= stack.top()) return 0;
    return size - stack.top();
  }
};

size_t avail_gpu_mem() {
  size_t free, total;
  gpuAssert(cudaMemGetInfo(&free, &total));
  return free;
}

class Runner {
  GPU_Stack_Buffer mem;
  cublasHandle_t handle;
  Stream s;
  bool smart = false;
  GEMM_Planner planner;

  std::vector<Predicate<std::pair<GEMM_Options, GEMM_Inputs>>> make_preds() {
    if (!smart) return {};
    std::vector<Predicate<std::pair<GEMM_Options, GEMM_Inputs>>> ret;
    ret.emplace_back(exclude_option(TRANS,TRANS));
    ret.emplace_back(exclude_option(TRANS,NOTRANS));
    return ret;
  }

public:
  

  Runner(bool smart = false) : mem((size_t)(((double)avail_gpu_mem())*0.9)), smart(smart),
                               planner(make_preds()){
    cublasCreate(&handle);
    cublasSetStream(handle,s);
  }

  ~Runner() { gpuAssert(cudaDeviceSynchronize()); cublasDestroy(handle); }

  GEMM_Options get_plan(GEMM_Inputs inputs) {
    if (!smart) return GEMM_Options(NOTRANS, NOTRANS);
    return planner.create_plan(inputs);
  }

  void run_problems(Problem_Set &problems, int reps) {
  
    for (auto &problem : problems.get_problems()) {
      int m = problem.m;
      int k = problem.k;
      int n = problem.n;
      GEMM_Options plan(NOTRANS, NOTRANS);
  
      Matrix A, B, C;
      A = (problem.opA == CUBLAS_OP_N) ? mem.allocate_matrix(m,k) : mem.allocate_matrix(k,m);
      B = (problem.opB == CUBLAS_OP_N) ? mem.allocate_matrix(k,n) : mem.allocate_matrix(n,k);
      C = mem.allocate_matrix(m,n);
  
      GEMM_Inputs inputs(handle, problem.opA, problem.opB, A, B, C,
                         1.0, 0.0, Workspace());
  
      std::cout << "Run problem " << problem << std::endl;
  
      for (int i = 0; i < reps; i++) {
        auto plan = get_plan(inputs);

        if (planner.calculate_workspace(plan,inputs)*sizeof(double) > mem.remaining_space()) {
          std::cout << "Insufficient memory for input " << problem << ", skipping" << std::endl;
          continue;
        }
        std::cout << "Running " << plan << std::endl;

        size_t ws = planner.calculate_workspace(plan,inputs)*sizeof(double);
        inputs.space = Workspace(mem.alloc(ws), ws);

        if (i == 0) planner.warmup(plan,inputs,s);
        planner.execute(plan, inputs, s);
        mem.pop();
      }
      gpuAssert(cudaDeviceSynchronize());
  
      problem.flop_rate = planner.get_floprate(inputs);
  
      mem.pop();
      mem.pop();
      mem.pop();
      problems.dump();
    }
  }
};


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
  Runner runner(true);
  Problem_Set problems(filename);
  // TODO check for duplicate dimensions when using smart measurement
  runner.run_problems(problems, reps);
}
