#pragma once
#include <string>
#include <vector>
#include <matrixop.h>
#include <executor.h>
#include "base_options.h"

namespace rtat {

template<typename T>
struct SYRK_Inputs {
  using Scalar = T;

  gpu::blasHandle_t handle;
  BLAS_Fill_Mode uplo;
  BLAS_Operation trans; 
  const Matrix<T> A;
        Matrix<T> C;
  const T alpha; 
  const T beta; 

  SYRK_Inputs(gpu::blasHandle_t handle, BLAS_Fill_Mode uplo, 
              BLAS_Operation trans, 
              const Matrix<T> A, Matrix<T> C, 
              T alpha, T beta)
        : handle(handle), uplo(uplo), trans(trans), 
          A(A), C(C), alpha(alpha), beta(beta) {}

  size_t n() {return C.dims().m;}
  size_t k() {return trans == gpu::BLAS_OP_N ? A.dims().n : A.dims().m;}
};


struct SYRK_Key {
  BLAS_Fill_Mode uplo;
  BLAS_Operation trans; 
  int n; int k;

  SYRK_Key(BLAS_Fill_Mode uplo, BLAS_Operation trans,
           int n, int k) : uplo(uplo), trans(trans), 
                           n(n), k(k) {}

  template<typename T>
  SYRK_Key(SYRK_Inputs<T> i) : 
    SYRK_Key(i.uplo, i.trans, i.n(), i.k()) {}


  operator std::string() const;
  bool operator<(const SYRK_Key&) const;
  friend std::ostream& operator<<(std::ostream&, const SYRK_Key&); 
};


struct SYRK_Options {
  Bool_Op transpose_A;
  Bool_Op transpose_C;

  SYRK_Options() = default;
  SYRK_Options(Bool_Op transpose_A, Bool_Op transpose_C) :
    transpose_A(transpose_A), transpose_C(transpose_C) {}

  static SYRK_Options default_opts() {
    return SYRK_Options();
  }

  static std::vector<SYRK_Options> enumerate();

  operator std::string() const;

  bool operator<(const SYRK_Options&) const;

  friend std::ostream& operator<<(std::ostream&, const SYRK_Options);
  friend std::istream& operator>>(std::istream&, SYRK_Options&); 

  template<typename T>
  std::unique_ptr<MatrixOp<T>> form_operation(SYRK_Inputs<T>);
};


template<typename T>
class SYRK_Executor : public Executor<SYRK_Inputs<T>, SYRK_Key, SYRK_Options> {
protected:
  void warmup(SYRK_Inputs<T> params, [[maybe_unused]] SYRK_Options opts,
              [[maybe_unused]] Stream s) override {
    size_t n = 64;
    double *A, *C;
    gpuAssert(gpu::Malloc(&A, n*n*sizeof(double)));
    gpuAssert(gpu::Malloc(&C, n*n*sizeof(double)));

    for (auto lower : {false,true}) {
      for (auto trans : {false,true}) {
        double alpha = 1.0;
        double beta = 0.0;
        auto status = gpu::blasDsyrk(params.handle,
          lower ? gpu::BLAS_FILL_MODE_LOWER : gpu::BLAS_FILL_MODE_UPPER,
          trans ? gpu::BLAS_OP_T : gpu::BLAS_OP_N,
          n, n, 
          &alpha,
          A, n,
          &beta,
          C, n);
        if (status != gpu::BLAS_STATUS_SUCCESS) {
          std::cout << "Fuc" << std::endl;
        }
        status = gpu::blasDgeam(params.handle, gpu::BLAS_OP_N, gpu::BLAS_OP_T, 
            n,n, &alpha, A, n, &beta, C, n, A, n);
        if (status != gpu::BLAS_STATUS_SUCCESS) {
          std::cout << "Fuc" << std::endl;
        }
      }
    }
    gpuAssert(gpu::DeviceSynchronize());
    gpuAssert(gpu::Free(A));
    gpuAssert(gpu::Free(C));
  }
};

}

