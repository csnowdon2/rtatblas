#pragma once
#include <string>
#include <vector>
#include <matrixop.h>
#include <executor.h>
#include "base_options.h"

namespace rtat {

template<typename T>
struct TRSM_Inputs {
  using Scalar = T;

  gpu::blasHandle_t handle;
  BLAS_Side side;
  BLAS_Fill_Mode uplo;
  BLAS_Operation trans; 
  BLAS_Diag diag;
  const Matrix<T> A;
        Matrix<T> B;
  const T alpha; 

  TRSM_Inputs(gpu::blasHandle_t handle, BLAS_Side side, 
              BLAS_Fill_Mode uplo, BLAS_Operation trans, 
              BLAS_Diag diag, 
              const Matrix<T> A, Matrix<T> B, T alpha)
        : handle(handle), side(side), uplo(uplo), trans(trans), 
          diag(diag), A(A), B(B), alpha(alpha){}

  size_t m() {return B.dims().m;}
  size_t n() {return B.dims().n;}
};


struct TRSM_Key {
  BLAS_Side side;
  BLAS_Fill_Mode uplo;
  BLAS_Operation trans; 
  BLAS_Diag diag;
  int m; int n;

  TRSM_Key(BLAS_Side side, BLAS_Fill_Mode uplo, 
           BLAS_Operation trans, BLAS_Diag diag,
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
    size_t n = 128;
    double *A, *B;
    gpuAssert(gpu::Malloc(&A, n*n*sizeof(double)));
    gpuAssert(gpu::Malloc(&B, n*n*sizeof(double)));

    for (auto side_left : {false,true}) {
      for (auto lower : {false,true}) {
        for (auto trans : {false,true}) {
          double alpha = 1.0;
          double beta = 0.0;
          gpu::blasDtrsm(params.handle, 
              side_left ? gpu::BLAS_SIDE_LEFT : gpu::BLAS_SIDE_RIGHT,
              lower ? gpu::BLAS_FILL_MODE_LOWER : gpu::BLAS_FILL_MODE_UPPER,
              trans ? gpu::BLAS_OP_T : gpu::BLAS_OP_N,
              gpu::BLAS_DIAG_NON_UNIT,
              n,n,&alpha,A,n,B,n);
          gpu::blasDgeam(params.handle, gpu::BLAS_OP_N, gpu::BLAS_OP_T, n,n, &alpha, A, n, &beta, B, n, A, n);
        }
      }
    }
    gpuAssert(gpu::DeviceSynchronize());
    gpuAssert(gpu::Free(A));
    gpuAssert(gpu::Free(B));
  }
};

}
