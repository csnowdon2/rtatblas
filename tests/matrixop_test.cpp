#include <gtest/gtest.h>
#include <matrixop.h>
#include <gpu-api.h>
#include <iostream>
#include "common.h"

class MatrixOp_Test : public BLAS_Test {};


TEST_F(MatrixOp_Test, MoveTest) {
  const int n = 12;
  const int m = 10;
  TestMatrix<double> A(m,n,m);

  std::unique_ptr<MatrixOp<double>> Aop = std::make_unique<NoOp<double>>(A);
  MatrixMove Bop(std::move(Aop), 1.0, false, 32);

  auto dims = Bop.dims();
  TestMatrix<double> B(dims.m,dims.n,dims.ld);

  ManagedWorkspace scratch(Bop.scratch_space_req_bytes());

  Bop.execute(handle, B.workspace(), scratch);
  B.download();

  std::cout << "A" << std::endl;
  A.print();
  std::cout << "B" << std::endl;
  B.print();
  ASSERT_TRUE(A == B);
}

TEST_F(MatrixOp_Test, MatMulTest) {
  int m = 26;
  int k = 34;
  int n = 27;

  TestMatrix<double> A(m,k,m);
  TestMatrix<double> B(k,n,k);
  TestMatrix<double> C(m,n,m);

  {
    std::unique_ptr<MatrixOp<double>> Aop = std::make_unique<NoOp<double>>(A);
    std::unique_ptr<MatrixOp<double>> Bop = std::make_unique<NoOp<double>>(B);
    std::unique_ptr<MatrixOp<double>> Cop = std::make_unique<NoOp<double>>(C);

    MatrixMult mult(std::move(Aop), std::move(Bop), std::move(Cop), false, false, 1.0, 0.0);
    ASSERT_EQ(mult.output_space_req(), 0);
    mult.execute(handle, Workspace(), ManagedWorkspace(mult.scratch_space_req_bytes()));

    C.download();
  }

  test_gemm(A, B, C, -1.0, 1.0, false, false);
  ASSERT_TRUE(C.is_zero());
}

TEST_F(MatrixOp_Test, TNMulTest) {
  int m = 25;
  int k = 14;
  int n = 32;

  TestMatrix<double> A(k,m,k);
  TestMatrix<double> B(k,n,k);
  TestMatrix<double> C(m,n,m);

  // do the multiply
  {
    std::unique_ptr<MatrixOp<double>> Aop = std::make_unique<NoOp<double>>(A);
    std::unique_ptr<MatrixOp<double>> Bop = std::make_unique<NoOp<double>>(B);
    std::unique_ptr<MatrixOp<double>> Cop = std::make_unique<NoOp<double>>(C);

    Cop = 
        std::make_unique<MatrixMult<double>>(std::move(Aop), std::move(Bop), std::move(Cop), 
                                     true, false, 2.0, 0.0);

    Aop = std::make_unique<NoOp<double>>(A);
    Bop = std::make_unique<NoOp<double>>(B);

    Aop = std::make_unique<MatrixMove<double>>(std::move(Aop), -2.0, true, 8);

    Cop = std::make_unique<MatrixMult<double>>(std::move(Aop), std::move(Bop), std::move(Cop),
                    false, false, 1.0, 1.0);


    ASSERT_EQ(Cop->output_space_req(), 0);
    ManagedWorkspace scratch(Cop->scratch_space_req_bytes());
    Cop->execute(handle, Workspace(), scratch);
    C.download();
  }

  ASSERT_TRUE(C.is_zero());
}

TEST_F(MatrixOp_Test, TTMulTest) {
  int m = 23;
  int k = 42;
  int n = 61;
  
  TestMatrix<double> A(k,m,k);
  TestMatrix<double> B(n,k,n);
  TestMatrix<double> C(m,n,m);

  // do the multiply
  {
    std::unique_ptr<MatrixOp<double>> Aop = std::make_unique<NoOp<double>>(A);
    std::unique_ptr<MatrixOp<double>> Bop = std::make_unique<NoOp<double>>(B);
    std::unique_ptr<MatrixOp<double>> Cop = std::make_unique<NoOp<double>>(C);

    Cop = 
        std::make_unique<MatrixMult<double>>(std::move(Aop), std::move(Bop), std::move(Cop), 
                                     true, true, 2.0, 0.0);

    Aop = std::make_unique<NoOp<double>>(A);
    Bop = std::make_unique<NoOp<double>>(B);

    Aop = std::make_unique<MatrixMove<double>>(std::move(Aop), 0.5, true, 16);
    Bop = std::make_unique<MatrixMove<double>>(std::move(Bop), -4.0, true, 8);

    Cop = std::make_unique<MatrixMult<double>>(std::move(Aop), std::move(Bop), std::move(Cop),
                    false, false, 1.0, 1.0);

    ASSERT_EQ(Cop->output_space_req(), 0);
    ManagedWorkspace scratch(Cop->scratch_space_req_bytes());
    Cop->execute(handle, Workspace(), scratch);
    C.download();
  }

  ASSERT_TRUE(C.is_zero());
}

TEST_F(MatrixOp_Test, TiledMatMulTest) {
  int m = 1024;
  int k = 1024;
  int n = 1024;

  TestMatrix<double> A(m,k,m);
  TestMatrix<double> B(k,n,k);
  TestMatrix<double> C(m,n,m);

  {
    std::unique_ptr<MatrixOp<double>> Aop = std::make_unique<NoOp<double>>(A);
    std::unique_ptr<MatrixOp<double>> Bop = std::make_unique<NoOp<double>>(B);
    std::unique_ptr<MatrixOp<double>> Cop = std::make_unique<NoOp<double>>(C);

    TiledMatrixMult mult(std::move(Aop), std::move(Bop), std::move(Cop), 
                         false, false, 1.0, 0.0, 128, 128, 128);
    ASSERT_EQ(mult.output_space_req(), 0);
    mult.execute(handle, Workspace(), ManagedWorkspace(mult.scratch_space_req_bytes()));
  }
  {
    std::unique_ptr<MatrixOp<double>> Aop = std::make_unique<NoOp<double>>(A);
    std::unique_ptr<MatrixOp<double>> Bop = std::make_unique<NoOp<double>>(B);
    std::unique_ptr<MatrixOp<double>> Cop = std::make_unique<NoOp<double>>(C);

    MatrixMult mult(std::move(Aop), std::move(Bop), std::move(Cop), false, false, -1.0, 1.0);
    mult.execute(handle, Workspace(), ManagedWorkspace(mult.scratch_space_req_bytes()));
  }

  C.download();
  ASSERT_TRUE(C.is_zero());
}
