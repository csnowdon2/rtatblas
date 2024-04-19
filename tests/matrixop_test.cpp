#include <gtest/gtest.h>
#include <matrixop.h>
#include <gpu-api.h>
#include <iostream>
#include "common.h"

class MatrixOp_Test : public BLAS_Test {};


TEST_F(MatrixOp_Test, MoveTest) {
  const int n = 12;
  const int m = 10;
  TestMatrix A(m,n,m);

  std::unique_ptr<MatrixOp> Aop = std::make_unique<NoOp>(A);
  MatrixMove Bop(std::move(Aop), 1.0, false, 32);

  auto dims = Bop.dims();
  TestMatrix B(dims.m,dims.n,dims.ld);

  ManagedWorkspace scratch(Bop.scratch_space_req());

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

  TestMatrix A(m,k,m);
  TestMatrix B(k,n,k);
  TestMatrix C(m,n,m);

  {
    std::unique_ptr<MatrixOp> Aop = std::make_unique<NoOp>(A);
    std::unique_ptr<MatrixOp> Bop = std::make_unique<NoOp>(B);
    std::unique_ptr<MatrixOp> Cop = std::make_unique<NoOp>(C);

    MatrixMult mult(std::move(Aop), std::move(Bop), std::move(Cop), false, false, 1.0, 0.0);
    ASSERT_EQ(mult.output_space_req(), 0);
    mult.execute(handle, Workspace(), ManagedWorkspace(mult.scratch_space_req()));

    C.download();
  }

  test_gemm(A, B, C, -1.0, 1.0, false, false);
  ASSERT_TRUE(C.is_zero());
}

TEST_F(MatrixOp_Test, TNMulTest) {
  int m = 25;
  int k = 14;
  int n = 32;

  TestMatrix A(k,m,k);
  TestMatrix B(k,n,k);
  TestMatrix C(m,n,m);

  // do the multiply
  {
    std::unique_ptr<MatrixOp> Aop = std::make_unique<NoOp>(A);
    std::unique_ptr<MatrixOp> Bop = std::make_unique<NoOp>(B);
    std::unique_ptr<MatrixOp> Cop = std::make_unique<NoOp>(C);

    Cop = 
        std::make_unique<MatrixMult>(std::move(Aop), std::move(Bop), std::move(Cop), 
                                     true, false, 2.0, 0.0);

    Aop = std::make_unique<NoOp>(A);
    Bop = std::make_unique<NoOp>(B);

    Aop = std::make_unique<MatrixMove>(std::move(Aop), -2.0, true, 8);

    Cop = std::make_unique<MatrixMult>(std::move(Aop), std::move(Bop), std::move(Cop),
                    false, false, 1.0, 1.0);


    ASSERT_EQ(Cop->output_space_req(), 0);
    ManagedWorkspace scratch(Cop->scratch_space_req());
    Cop->execute(handle, Workspace(), scratch);
    C.download();
  }

  ASSERT_TRUE(C.is_zero());
}

TEST_F(MatrixOp_Test, TTMulTest) {
  int m = 23;
  int k = 42;
  int n = 61;
  
  TestMatrix A(k,m,k);
  TestMatrix B(n,k,n);
  TestMatrix C(m,n,m);

  // do the multiply
  {
    std::unique_ptr<MatrixOp> Aop = std::make_unique<NoOp>(A);
    std::unique_ptr<MatrixOp> Bop = std::make_unique<NoOp>(B);
    std::unique_ptr<MatrixOp> Cop = std::make_unique<NoOp>(C);

    Cop = 
        std::make_unique<MatrixMult>(std::move(Aop), std::move(Bop), std::move(Cop), 
                                     true, true, 2.0, 0.0);

    Aop = std::make_unique<NoOp>(A);
    Bop = std::make_unique<NoOp>(B);

    Aop = std::make_unique<MatrixMove>(std::move(Aop), 0.5, true, 16);
    Bop = std::make_unique<MatrixMove>(std::move(Bop), -4.0, true, 8);

    Cop = std::make_unique<MatrixMult>(std::move(Aop), std::move(Bop), std::move(Cop),
                    false, false, 1.0, 1.0);

    ASSERT_EQ(Cop->output_space_req(), 0);
    ManagedWorkspace scratch(Cop->scratch_space_req());
    Cop->execute(handle, Workspace(), scratch);
    C.download();
  }

  ASSERT_TRUE(C.is_zero());
}

TEST_F(MatrixOp_Test, TiledMatMulTest) {
  int m = 1024;
  int k = 1024;
  int n = 1024;

  TestMatrix A(m,k,m);
  TestMatrix B(k,n,k);
  TestMatrix C(m,n,m);

  {
    std::unique_ptr<MatrixOp> Aop = std::make_unique<NoOp>(A);
    std::unique_ptr<MatrixOp> Bop = std::make_unique<NoOp>(B);
    std::unique_ptr<MatrixOp> Cop = std::make_unique<NoOp>(C);

    TiledMatrixMult mult(std::move(Aop), std::move(Bop), std::move(Cop), 
                         false, false, 1.0, 0.0, 128, 128, 128);
    ASSERT_EQ(mult.output_space_req(), 0);
    mult.execute(handle, Workspace(), ManagedWorkspace(mult.scratch_space_req()));
  }
  {
    std::unique_ptr<MatrixOp> Aop = std::make_unique<NoOp>(A);
    std::unique_ptr<MatrixOp> Bop = std::make_unique<NoOp>(B);
    std::unique_ptr<MatrixOp> Cop = std::make_unique<NoOp>(C);

    MatrixMult mult(std::move(Aop), std::move(Bop), std::move(Cop), false, false, -1.0, 1.0);
    mult.execute(handle, Workspace(), ManagedWorkspace(mult.scratch_space_req()));
  }

  C.download();
  ASSERT_TRUE(C.is_zero());
}
