#include <gtest/gtest.h>
#include <gemm.h>
#include <trsm.h>
#include <syrk.h>
#include "common.h"
#include "gpu-api.h"

class GEMM_Executor_Test : public BLAS_Test {};
class TRSM_Executor_Test : public BLAS_Test {};
class SYRK_Executor_Test : public BLAS_Test {};

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

TEST_F(TRSM_Executor_Test, TRSM_Correctness_Double) {
  TRSM_Executor<double> trsm_exec;
  GEMM_Executor<double> gemm_exec;

  int m = 435;
  int n = 527;

  for (auto side_left : {false,true}) {
    for (auto lower : {false,true}) {
      for (auto unit_diag : {false,true}) {
        for (auto trans : {false,true}) {
          for (auto &opts : TRSM_Options::enumerate()) {
            TestMatrix<double> A(m,m,m);
            int m_X = side_left ? m : n;
            int n_X = side_left ? n : m;
            TestMatrix<double> X(m_X,n_X,m_X);
            TestMatrix<double> B(m_X,n_X,m_X);
            if (lower) {
              for (int i=0; i<m; i++) {
                for (int j=0; j<m; j++) {
                  if (i > j) 
                    A.host_vector[i*A.ld+j] = 0.0;
                  if (unit_diag && i == j)
                    A.host_vector[i*A.ld+j] = 1.0;
                  // Force diagonal dominance
                  if (!unit_diag && i == j)
                    A.host_vector[i*A.ld+j] += m+1;
                  if (i < j && unit_diag)
                    A.host_vector[i*A.ld+j] /= m;
                }
              }
            } else {
              for (int i=0; i<m; i++) {
                for (int j=0; j<m; j++) {
                  if (i < j) 
                    A.host_vector[i*A.ld+j] = 0.0;
                  if (unit_diag && i == j)
                    A.host_vector[i*A.ld+j] = 1.0;
                  // Force diagonal dominance
                  if (!unit_diag && i == j)
                    A.host_vector[i*A.ld+j] += m+1;
                  if (i > j && unit_diag)
                    A.host_vector[i*A.ld+j] /= m;
                }
              }
            }
            A.upload();
            B.upload();
            X.upload();

            // B := AX
            if (side_left) {
              GEMM_Inputs<double> inputs(handle, 
                  trans ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, 
                  A, X, B, 1.0, 0.0);
              gemm_exec.execute(
                  inputs, GEMM_Options(), Workspace(), s);
            } else {
              GEMM_Inputs<double> inputs(handle, 
                  CUBLAS_OP_N, trans ? CUBLAS_OP_T : CUBLAS_OP_N, 
                  X, A, B, 1.0, 0.0);
              gemm_exec.execute(
                  inputs, GEMM_Options(), Workspace(), s);
            }

            // Solve AX := B
            {
              TRSM_Inputs<double> inputs(handle, 
                side_left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT, 
                lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER,
                trans ? CUBLAS_OP_T : CUBLAS_OP_N, 
                unit_diag ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT, 
                A, B, 1.0);

              size_t ws = 
                trsm_exec.calculate_workspace(inputs, opts);
              ManagedWorkspace space(ws);

              trsm_exec.execute(inputs, opts, space, s);
            }

            B.download();
            X.download();
            double delta = diff(B,X);
            bool check = delta < 1e-10;
            EXPECT_TRUE(check);
            if (!check)
              std::cout << "delta=" << delta << std::endl;
          }
        }
      }
    }
  }
}

TEST_F(SYRK_Executor_Test, SYRK_Correctness_Double) {
  SYRK_Executor<double> syrk_exec;
  GEMM_Executor<double> gemm_exec;

  int n = 213;
  int k = 74;

  for (auto lower : {false,true}) {
    for (auto trans : {false,true}) {
      for (auto &opts : SYRK_Options::enumerate()) {
        TestMatrix<double> C(n,n,n);
        int m_A = trans ? k : n;
        int n_A = trans ? n : k;
        TestMatrix<double> A(m_A,n_A,m_A);
        for (int i=0; i<n; i++) {
          for (int j=0; j<n; j++) {
            if (i > j) 
              C.host_vector[i*C.ld+j] = C.host_vector[j*C.ld+i];
            // Force diagonal dominance
          }
        }
        A.upload();
        C.upload();

        // C := op(A)op(A)^T
        {
          GEMM_Inputs<double> inputs(handle, 
              trans ? CUBLAS_OP_T : CUBLAS_OP_N, 
              trans ? CUBLAS_OP_N : CUBLAS_OP_T, 
              A, A, C, 1.0, 0.0);
          gemm_exec.execute(
              inputs, GEMM_Options(), Workspace(), s);
        }

        // C := C - op(A)op(A)^T
        {
          SYRK_Inputs<double> inputs(handle, 
            lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER,
            trans ? CUBLAS_OP_T : CUBLAS_OP_N, 
            A, C, -1.0, 1.0);

          size_t ws = 
            syrk_exec.calculate_workspace(inputs, opts);
          ManagedWorkspace space(ws);

          syrk_exec.execute(inputs, opts, space, s);
        }

        C.download();
        // Zero out un-used triangle
        for (int i=0; i<n; i++) {
          for (int j=0; j<n; j++) {
            if (lower && i > j) 
              C.host_vector[i*C.ld+j] = 0.0;
            if (!lower && i < j) 
              C.host_vector[i*C.ld+j] = 0.0;
          }
        }

        EXPECT_TRUE(C.is_zero());
      }
    }
  }
}
