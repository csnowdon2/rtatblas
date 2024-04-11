#include "runner.h"
#include <random>

int main(int argc, char *argv[]) {
  using namespace rtat;
  if (argc != 7) { 
    std::cout << "Expected command line args: m k n mblock kblock nblock" << std::endl;
    return 1;
  }

  cublasHandle_t handle;
  cublasCreate(&handle);
  const int reps = 10;
  int m = atoi(argv[1]);
  int k = atoi(argv[2]);
  int n = atoi(argv[3]);

  int mblock = atoi(argv[4]);
  int kblock = atoi(argv[5]);
  int nblock = atoi(argv[6]);
  //cublasOperation_t opA = argv[4][0] == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T;
  //cublasOperation_t opB = argv[5][0] == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T;

  GPU_Stack_Buffer mem((size_t)(((double)avail_gpu_mem())*0.9));

  {
    // Warmup
    Matrix A = mem.allocate_matrix(10,10);
    Matrix B = mem.allocate_matrix(10,10);
    Matrix C = mem.allocate_matrix(10,10);
    std::unique_ptr<MatrixOp> Aop = std::make_unique<NoOp>(A);
    std::unique_ptr<MatrixOp> Bop = std::make_unique<NoOp>(B);
    std::unique_ptr<MatrixOp> Cop = std::make_unique<NoOp>(C);
    MatrixMult mult(std::move(Aop), std::move(Bop), std::move(Cop), false, false, 1.0, 0.0);
    mult.execute(handle, Workspace(), Workspace());
  }

  std::mt19937 rng;

  Matrix A = mem.allocate_matrix(m,k);
  Matrix B = mem.allocate_matrix(k,n);
  Matrix C = mem.allocate_matrix(m,n);
  //for (int mblock=64; mblock<=mblock_max; mblock += 128) {
  //  for (int nblock=64; nblock<=nblock_max; nblock += 128) {

      auto t1 = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < reps; i++) {
        std::unique_ptr<MatrixOp> Aop = std::make_unique<NoOp>(A);
        std::unique_ptr<MatrixOp> Bop = std::make_unique<NoOp>(B);
        std::unique_ptr<MatrixOp> Cop = std::make_unique<NoOp>(C);

        TiledMatrixMult mult(std::move(Aop), std::move(Bop), std::move(Cop), 
                             false, false, 1.0, 0.0, mblock, kblock, nblock);

        mult.execute(handle, Workspace(), Workspace());
        gpuAssert(cudaDeviceSynchronize());
      }
      auto t2 = std::chrono::high_resolution_clock::now();
      double t = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()/1000000.0;
      double floprate = reps*((size_t)2)*((size_t)m)*((size_t)k)*((size_t)n)/t*(1e-12);
      std::cout << "\tmblock=" << mblock << "\tnblock=" << nblock << "\tkblock=" << kblock << " " << floprate << " TFLOP/s" << std::endl;
  //  }
  //}

  return 0;
}
