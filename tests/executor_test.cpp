#include <gtest/gtest.h>
#include <gemm.h>
#include "common.h"

class GEMM_Executor_Test : public BLAS_Test {};

TEST_F(GEMM_Executor_Test, Correctness_Double) {
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

TEST_F(GEMM_Executor_Test, Correctness_Single) {
  GEMM_Executor<float> exec;

  int m = 70;
  int n = 45;
  int k = 62;


  float alpha = 1.0;


  for (bool transa : {true,false}) {
    for (bool transb : {true,false}) {
      for (auto &opts : GEMM_Options::enumerate()) {
        int Am = transa ? k : m;
        int An = transa ? m : k;
        int Bm = transb ? n : k;
        int Bn = transb ? k : n;

        TestMatrix<float> A(Am,An,Am);
        TestMatrix<float> B(Bm,Bn,Bm);
        TestMatrix<float> C(m,n,m);

        GEMM_Inputs<float> inputs(handle, 
            transa ? CUBLAS_OP_T : CUBLAS_OP_N, 
            transb ? CUBLAS_OP_T : CUBLAS_OP_N, 
            A, B, C, alpha, 0.0);

        size_t ws = exec.calculate_workspace(inputs, opts);
        ManagedWorkspace space(ws);

        exec.execute(inputs, opts, space, s);

        C.download();
        EXPECT_TRUE(!C.is_zero());
        test_gemm<float>(A, B, C, -alpha, 1.0, transa, transb);
        EXPECT_TRUE(C.is_zero());
      }
    }
  }
}
