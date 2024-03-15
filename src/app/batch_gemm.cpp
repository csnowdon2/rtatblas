#include "runner.h"

int main(int argc, char *argv[]) {
  using namespace rtat;
  if (argc != 7) { 
    std::cout << "Expected command line args: m k n opA opB reps" << std::endl;
    return 1;
  }

  cublasHandle_t handle;
  cublasCreate(&handle);
  const int reps = 10;
  int m = atoi(argv[1]);
  int k = atoi(argv[2]);
  int n = atoi(argv[3]);
  //cublasOperation_t opA = argv[4][0] == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T;
  //cublasOperation_t opB = argv[5][0] == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T;

  GPU_Stack_Buffer mem((size_t)(((double)avail_gpu_mem())*0.9));

  {
    Matrix A = mem.allocate_matrix(m,k);
    Matrix B = mem.allocate_matrix(k,n);
    Matrix C = mem.allocate_matrix(m,n);

    for (int i = 0; i < reps; i++) {
      std::unique_ptr<MatrixOp> Aop = std::make_unique<NoOp>(A);
      std::unique_ptr<MatrixOp> Bop = std::make_unique<NoOp>(B);
      std::unique_ptr<MatrixOp> Cop = std::make_unique<NoOp>(C);

      BatchMatrixMult mult(std::move(Aop), std::move(Bop), std::move(Cop), false, false, 1.0, 0.0);

      auto t1 = std::chrono::high_resolution_clock::now();
      mult.execute(handle, Workspace(), Workspace());
      gpuAssert(cudaDeviceSynchronize());
      auto t2 = std::chrono::high_resolution_clock::now();
      double t = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()/1000000.0;
      std::cout << i << t << "s" << std::endl;
    }
  }

  return 0;
}
