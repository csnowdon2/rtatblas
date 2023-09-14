#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <iostream>
#include "../src/planning_system.h"
#include "common.h"


TEST(Planning_Test, Hello) {
  GEMM_Planner planner;
  cublasHandle_t handle;

  cublasCreate(&handle);
  cudaDeviceSynchronize();

  Stream s;
  cublasSetStream(handle, s);

  int m = 42;
  int n = 35;
  int k = 60;

  TestMatrix A(m,k,m);
  TestMatrix B(k,n,k);
  TestMatrix C(m,n,m);

  double alpha = 1.0;

  GEMM_Inputs inputs(handle, CUBLAS_OP_N, CUBLAS_OP_N, A, B, C, alpha, 0.0, Workspace());

  for (int i=0; i<10; i++) {
    for (auto &plan : GEMM_Options::enumerate()) {
      size_t ws = planner.calculate_workspace(plan, inputs)*sizeof(double);
      double *space;
      cudaMalloc(&space, ws);
      inputs.space = Workspace(space, ws);

      for (int j=0; j<10; j++)
        planner.execute(plan, inputs, s);

      C.download();
      test_gemm(A, B, C, -alpha, 1.0, false, false);

      for (int k = 0; k < C.m; k++) 
        for (int j = 0; j < C.n; j++) 
          ASSERT_NEAR(C.host_vector[j*C.ld+k], 0.0, 1e-10);
      cudaFree(space);
    }
  }

  cublasDestroy(handle);
  SUCCEED();
}

TEST(Plan_Create_Test, Hello) {
  GEMM_Planner planner;
  cublasHandle_t handle;
  cublasCreate(&handle);

  Stream s;
  cublasSetStream(handle, s);

  size_t m = 12000;
  size_t n = 3840;
  size_t k = 12000;

  TestMatrix A(m,k,m);
  TestMatrix B(k,n,k);
  TestMatrix C(m,n,m);


  double alpha = 1.0;

  GEMM_Inputs inputs(handle, CUBLAS_OP_N, CUBLAS_OP_N, A, B, C, alpha, 0.0, Workspace());

  size_t ws = 0;
  double *space = nullptr;
  for (int j=0; j<2; j++) {
    auto t1 = std::chrono::high_resolution_clock::now();
    size_t reps = 100;
    for (int i=0; i<reps; i++) {
      GEMM_Options plan = planner.create_plan(inputs);
      //std::cout << plan << std::endl;
      size_t req = planner.calculate_workspace(plan, inputs)*sizeof(double);
      if (req > ws) {
        cudaDeviceSynchronize();
        ws = req;
        if (space)
          cudaFree(space);
        cudaMalloc(&space, ws);
        inputs.space = Workspace(space, ws);
      }
      inputs.space = Workspace(space, ws);

      planner.execute(plan, inputs, s);
    }
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    double t = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
    std::cout << "Avg flop rate=" << reps*m*n*k*2/(t*1e9) << "TFLOP/s" << std::endl;
    planner.dump_analytics();
  }

  cublasDestroy(handle);
  SUCCEED();
}
