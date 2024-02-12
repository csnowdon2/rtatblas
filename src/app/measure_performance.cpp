#include <iostream>
#include <planning_system.h>
#include <string>

using namespace rtat;

cublasOperation_t read_op(std::string op) {
  if (op == "N") return CUBLAS_OP_N;
  if (op == "T") return CUBLAS_OP_T;
  throw;
}

Workspace allocate_workspace(size_t size) {
  double *ptr;
  gpuAssert(cudaMalloc(&ptr, size));
  return Workspace(ptr, size);
}

Matrix allocate_matrix(size_t m, size_t n) {
  size_t size = m*n*sizeof(double);
  return Matrix(allocate_workspace(size), m, n, m);
}

int main(int argc, char *argv[]) {
  if (argc != 7) {
    std::cout << "Expected 6 parameters: m k n reps opA opB" << std::endl;
    return 1;
  }

  int m = std::stoi(std::string(argv[1]));
  int k = std::stoi(std::string(argv[2]));
  int n = std::stoi(std::string(argv[3]));
  int reps = std::stoi(std::string(argv[4]));

  auto opA = read_op(std::string(argv[5]));
  auto opB = read_op(std::string(argv[6]));

  GEMM_Planner planner;

  cublasHandle_t handle;
  cublasCreate(&handle);

  Stream s;
  cublasSetStream(handle, s);

  auto plans = GEMM_Options::enumerate();

  auto A = opA == CUBLAS_OP_N ? allocate_matrix(m,k) : allocate_matrix(k,m);
  auto B = opB == CUBLAS_OP_N ? allocate_matrix(k,n) : allocate_matrix(n,k);
  auto C = allocate_matrix(m,n);
  GEMM_Inputs inputs(handle, opA, opB, A, B, C, 1.0, 0.0, Workspace());
  
  {
    size_t workspace_req = 0;
    for (auto &plan : plans)
      workspace_req = std::max(workspace_req, 
                               planner.calculate_workspace(plan,inputs)*sizeof(double));

    inputs.space = allocate_workspace(workspace_req);
  }

  // DO WARMUP

  for (auto &plan : plans)
    for (int i = 0; i < reps; i++) 
      planner.execute(plan, inputs, s);

  planner.dump_analytics();
  return 0;
}
