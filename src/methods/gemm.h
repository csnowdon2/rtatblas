#pragma once
#include <string>
#include <vector>
#include <matrixop.h>
#include "executor.h"
#include "base_options.h"

namespace rtat {

struct GEMM_Inputs {
  cublasHandle_t handle;
  cublasOperation_t transa; cublasOperation_t transb;
  const Matrix A;
  const Matrix B;
        Matrix C;
  const double alpha; const double beta;

  GEMM_Inputs(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
              const Matrix A, const Matrix B, Matrix C, double alpha, double beta)
        : handle(handle), transa(transa), transb(transb), A(A), B(B), C(C), 
          alpha(alpha), beta(beta) {}

  size_t m() {return C.dims().m;}
  size_t n() {return C.dims().n;}
  size_t k() {return (transa == CUBLAS_OP_N) ? A.dims().n : A.dims().m;}
};


struct GEMM_Key {
  cublasOperation_t transa; cublasOperation_t transb;
  int m; int n; int k;

  GEMM_Key(GEMM_Inputs i) : transa(i.transa), transb(i.transb), 
                            m(i.m()), n(i.n()), k(i.k()) {}

  GEMM_Key(cublasOperation_t transa, cublasOperation_t transb,
           int m, int k, int n) : transa(transa), transb(transb), 
                                  m(m), n(n), k(k) {}

  operator std::string() const;
  bool operator<(const GEMM_Key&) const;
  friend std::ostream& operator<<(std::ostream&, const GEMM_Key&); 
};


struct GEMM_Options {
  BLAS_Op transa;
  Pad_Op  pada;
  BLAS_Op transb;
  Pad_Op  padb;
  BLAS_Op transc;
  Pad_Op  padc;

  GEMM_Options() = default;
  GEMM_Options(BLAS_Op transa, Pad_Op pada,
               BLAS_Op transb, Pad_Op padb,
               BLAS_Op transc, Pad_Op padc) :
    transa(transa), pada(pada), 
    transb(transb), padb(padb),
    transc(transc), padc(padc) {}

  static GEMM_Options default_opts() {
    return GEMM_Options();
  }

  static std::vector<GEMM_Options> enumerate();

  operator std::string() const;

  bool operator<(const GEMM_Options&) const;

  friend std::ostream& operator<<(std::ostream&, const GEMM_Options);
  friend std::istream& operator>>(std::istream&, GEMM_Options&); 

  std::unique_ptr<MatrixOp> form_operation(GEMM_Inputs);
};


class GEMM_Executor : public Executor<GEMM_Inputs, GEMM_Key, GEMM_Options> {
protected:
  void warmup(GEMM_Inputs params, [[maybe_unused]] GEMM_Options opts,
              [[maybe_unused]] Stream s) override {
    size_t n = 8;
    double *A, *B, *C;
    gpuAssert(cudaMalloc(&A, n*n*sizeof(double)));
    gpuAssert(cudaMalloc(&B, n*n*sizeof(double)));
    gpuAssert(cudaMalloc(&C, n*n*sizeof(double)));

    std::vector<cublasOperation_t> ops = {CUBLAS_OP_N, CUBLAS_OP_T};

    for (auto &opA : ops) {
      for (auto &opB : ops) {
        double alpha = 1.0;
        double beta = 0.0;
        cublasDgemm(params.handle, opA, opB, n,n,n, &alpha, A,n,B,n, &beta, C,n);
        cublasDgeam(params.handle, opA, opB, n,n, &alpha, A, n, &beta, B, n, C, n);
      }
    }
    gpuAssert(cudaDeviceSynchronize());
    gpuAssert(cudaFree(A));
    gpuAssert(cudaFree(B));
    gpuAssert(cudaFree(C));
  }
};

}
