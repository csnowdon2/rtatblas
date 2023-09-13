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
  GPU_Stack_Buffer(size_t size) : size(size) { cudaMalloc(&ptr, size); }
  ~GPU_Stack_Buffer() { cudaFree(ptr); }

  void pop() { stack.pop(); }

  double* alloc(size_t doubles) {
    doubles = (doubles/32)*32;

    size_t pos = (stack.size() == 0) ? 0 : stack.top();
    std::cout << "Alloc " << pos << " to " << pos+doubles << "/" << size/sizeof(double) << std::endl;
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
  cudaMemGetInfo(&free, &total);
  return free;
}

void run_problems(Problem_Set &problems) {
  cublasHandle_t handle;
  cublasCreate(&handle);

  Stream s;
  cublasSetStream(handle, s);

  GPU_Stack_Buffer mem((size_t)(((double)avail_gpu_mem())*0.9));
  GEMM_Planner planner;

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

    if (planner.calculate_workspace(plan,inputs)*sizeof(double) > mem.remaining_space()) {
      std::cout << "Insufficient memory for input " << problem << ", skipping" << std::endl;
      continue;
    }

    size_t ws = planner.calculate_workspace(plan,inputs)*sizeof(double);
    inputs.space = Workspace(mem.alloc(ws), ws);

    std::cout << "Run problem " << problem << std::endl;
    planner.warmup(plan, inputs, s);

    for (int i = 0; i < 10; i++) 
      planner.execute(plan, inputs, s);
    cudaDeviceSynchronize();

    problem.flop_rate = planner.get_floprate(plan, inputs);

    mem.pop();
    mem.pop();
    mem.pop();
    mem.pop();
    problems.dump();
  }
}

int main(int argc, char *argv[]) {
  if (argc != 2) { 
    std::cout << "Expected one command line arg: filename" << std::endl;
    return 1;
  }

  std::string filename(argv[1]);

  Problem_Set problems(filename);
  run_problems(problems);
}
