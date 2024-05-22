#pragma once
#include <string>
#include <vector>
#include <matrixop.h>
#include <executor.h>
#include "base_options.h"

namespace rtat {

template<typename T>
struct TRSM_Inputs {
  cublasHandle_t handle;
  cublasSideMode_t side;
  cublasFillMode_t uplo;
  cublasOperation_t trans; 
  cublasDiagType_t diag;
  const Matrix<T> A;
        Matrix<T> B;
  const T alpha; 

  TRSM_Inputs(cublasHandle_t handle, cublasSideMode_t side, 
              cublasFillMode_t uplo, cublasOperation_t trans, 
              cublasDiagType_t diag, 
              const Matrix<T> A, Matrix<T> B, T alpha)
        : handle(handle), side(side), uplo(uplo), trans(trans), 
          diag(diag), A(A), B(B), alpha(alpha){}

  size_t m() {return B.dims().m;}
  size_t n() {return B.dims().n;}
};


struct TRSM_Key {
  cublasSideMode_t side;
  cublasFillMode_t uplo;
  cublasOperation_t trans; 
  cublasDiagType_t diag;
  int m; int n;

  TRSM_Key(cublasSideMode_t side, cublasFillMode_t uplo, 
           cublasOperation_t trans, cublasDiagType_t diag,
           int m, int n) : side(side), uplo(uplo),
                           trans(trans), diag(diag),
                           m(m), n(n) {}

  template<typename T>
  TRSM_Key(TRSM_Inputs<T> i) : 
    TRSM_Key(i.side, i.uplo, i.trans, i.diag, i.m(), i.n()) {}


  operator std::string() const;
  bool operator<(const TRSM_Key&) const;
  friend std::ostream& operator<<(std::ostream&, const TRSM_Key&); 
};


struct TRSM_Options {
  Bool_Op swap_side;
  Bool_Op transpose_A;

  TRSM_Options() = default;
  TRSM_Options(Bool_Op swap_side, Bool_Op transpose_A) :
    swap_side(swap_side), transpose_A(transpose_A) {}

  static TRSM_Options default_opts() {
    return TRSM_Options();
  }

  static std::vector<TRSM_Options> enumerate();

  operator std::string() const;

  bool operator<(const TRSM_Options&) const;

  friend std::ostream& operator<<(std::ostream&, const TRSM_Options);
  friend std::istream& operator>>(std::istream&, TRSM_Options&); 

  template<typename T>
  std::unique_ptr<MatrixOp<T>> form_operation(TRSM_Inputs<T>);
};


template<typename T>
class TRSM_Executor : public Executor<TRSM_Inputs<T>, TRSM_Key, TRSM_Options> {
protected:
  void warmup(TRSM_Inputs<T> params, [[maybe_unused]] TRSM_Options opts,
              [[maybe_unused]] Stream s) override {
    size_t n = 8;
    double *A, *B;
    gpuAssert(cudaMalloc(&A, n*n*sizeof(double)));
    gpuAssert(cudaMalloc(&B, n*n*sizeof(double)));

    std::vector<cublasOperation_t> ops = {CUBLAS_OP_N, CUBLAS_OP_T};

    for (auto side_left : {false,true}) {
      for (auto lower : {false,true}) {
        for (auto trans : {false,true}) {
          double alpha = 1.0;
          double beta = 0.0;
          cublasDtrsm(params.handle, 
              side_left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT,
              lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER,
              trans ? CUBLAS_OP_T : CUBLAS_OP_N,
              CUBLAS_DIAG_NON_UNIT,
              n,n,&alpha,A,n,B,n);
          cublasDgeam(params.handle, CUBLAS_OP_N, CUBLAS_OP_T, n,n, &alpha, A, n, &beta, B, n, A, n);
        }
      }
    }
    gpuAssert(cudaDeviceSynchronize());
    gpuAssert(cudaFree(A));
    gpuAssert(cudaFree(B));
  }
};

}
