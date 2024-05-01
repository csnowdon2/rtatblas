#include <gtest/gtest.h>
#include <planning_system.h>
#include "common.h"

class Planning_Test : public BLAS_Test {};

TEST_F(Planning_Test, GEMM_Correctness) {
  GEMM_Planner planner;

  int m = 23;
  int n = 16;
  int k = 35;

  TestMatrix A(m,k,m);
  TestMatrix B(k,n,k);
  TestMatrix C(m,n,m);

  double alpha = 1.0;

  GEMM_Inputs inputs(handle, CUBLAS_OP_N, CUBLAS_OP_N, A, B, C, alpha, 0.0);

  for (int i=0; i<10; i++) {
    for (auto &plan : GEMM_Options::enumerate()) {
      size_t ws = planner.calculate_workspace(inputs, plan);
      ManagedWorkspace space(ws);

      planner.execute(inputs, plan, space, s);

      C.download();
      test_gemm(A, B, C, -alpha, 1.0, false, false);

      EXPECT_TRUE(C.is_zero());
    }
  }
}

TEST_F(Planning_Test, Hello) {
  // This isn't really testing anything?
  GEMM_Planner planner;

  size_t m = 423;
  size_t n = 125;
  size_t k = 318;

  TestMatrix A(m,k,m);
  TestMatrix B(k,n,k);
  TestMatrix C(m,n,m);

  double alpha = 1.0;

  GEMM_Inputs inputs(handle, CUBLAS_OP_N, CUBLAS_OP_N, A, B, C, alpha, 0.0);

  ManagedWorkspace space(1024);
  for (int j=0; j<2; j++) {
    size_t reps = 100;
    for (size_t i=0; i<reps; i++) {
      GEMM_Options plan = planner.create_plan(inputs);

      size_t req = planner.calculate_workspace(inputs, plan)*sizeof(double);
      space.grow_to_fit(req);

      planner.execute(inputs, plan, space, s);
    }
    gpuAssert(cudaDeviceSynchronize());
    //planner.dump_analytics();
  }
}

// Check that every plan can run without workspace
TEST_F(Planning_Test, Plan_Degradation) {
  GEMM_Planner planner;

  size_t m = 69;
  size_t n = 123;
  size_t k = 42;

  TestMatrix A(m,k,m);
  TestMatrix B(k,n,k);
  TestMatrix C(m,n,m);

  double alpha = 1.0;
  GEMM_Inputs inputs(handle, CUBLAS_OP_N, CUBLAS_OP_N, A, B, C, alpha, 0.0);

  for (auto &plan : GEMM_Options::enumerate()) {
    planner.execute(inputs, plan, Workspace(), s);

    C.download();
    test_gemm(A, B, C, -alpha, 1.0, false, false);

    ASSERT_TRUE(C.is_zero());
  }
}
