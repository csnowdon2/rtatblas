#include <gtest/gtest.h>
#include <gemm.h>
#include "common.h"

class GEMM_Executor_Test : public BLAS_Test {};

TEST_F(GEMM_Executor_Test, Correctness) {
  GEMM_Executor<double> exec;

  int m = 70;
  int n = 45;
  int k = 62;


  double alpha = 1.0;


  for (auto &opts : GEMM_Options::enumerate()) {
    TestMatrix<double> A(m,k,m);
    TestMatrix<double> B(k,n,k);
    TestMatrix<double> C(m,n,m);
    GEMM_Inputs<double> inputs(handle, CUBLAS_OP_N, CUBLAS_OP_N, A, B, C, alpha, 0.0);

    size_t ws = exec.calculate_workspace(inputs, opts);
    ManagedWorkspace space(ws);

    exec.execute(inputs, opts, space, s);

    C.download();
    EXPECT_TRUE(!C.is_zero());
    test_gemm(A, B, C, -alpha, 1.0, false, false);
    EXPECT_TRUE(C.is_zero());
  }
  for (auto &opts : GEMM_Options::enumerate()) {
    TestMatrix<double> A(m,k,m);
    TestMatrix<double> B(n,k,n);
    TestMatrix<double> C(m,n,m);
    GEMM_Inputs<double> inputs(handle, CUBLAS_OP_N, CUBLAS_OP_T, A, B, C, alpha, 0.0);

    size_t ws = exec.calculate_workspace(inputs, opts);
    ManagedWorkspace space(ws);

    exec.execute(inputs, opts, space, s);

    C.download();
    EXPECT_TRUE(!C.is_zero());
    test_gemm(A, B, C, -alpha, 1.0, false, true);
    EXPECT_TRUE(C.is_zero());
  }
  for (auto &opts : GEMM_Options::enumerate()) {
    TestMatrix<double> A(k,m,k);
    TestMatrix<double> B(k,n,k);
    TestMatrix<double> C(m,n,m);
    GEMM_Inputs<double> inputs(handle, CUBLAS_OP_T, CUBLAS_OP_N, A, B, C, alpha, 0.0);

    size_t ws = exec.calculate_workspace(inputs, opts);
    ManagedWorkspace space(ws);

    exec.execute(inputs, opts, space, s);

    C.download();
    EXPECT_TRUE(!C.is_zero());
    test_gemm(A, B, C, -alpha, 1.0, true, false);
    EXPECT_TRUE(C.is_zero());
  }
  for (auto &opts : GEMM_Options::enumerate()) {
    TestMatrix<double> A(k,m,k);
    TestMatrix<double> B(n,k,n);
    TestMatrix<double> C(m,n,m);
    GEMM_Inputs<double> inputs(handle, CUBLAS_OP_T, CUBLAS_OP_T, A, B, C, alpha, 0.0);

    size_t ws = exec.calculate_workspace(inputs, opts);
    ManagedWorkspace space(ws);

    exec.execute(inputs, opts, space, s);

    C.download();
    EXPECT_TRUE(!C.is_zero());
    test_gemm(A, B, C, -alpha, 1.0, true, true);
    EXPECT_TRUE(C.is_zero());
  }
  SUCCEED();
}
