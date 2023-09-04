#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <iostream>
#include "../src/planning_system.h"

class TestMatrix {
public:
  TestMatrix(size_t m, size_t n, size_t ld) : m(m), n(n), ld(ld) {
    if (ld < m) throw "Bad matrix ld";
    host_vector.resize(footprint());
    cudaMalloc((void**)&dev_ptr, footprint()*sizeof(double));

    randomize_host();
    upload();
  }

  ~TestMatrix() {cudaFree(dev_ptr);}

  size_t m;
  size_t n;
  size_t ld;

  size_t footprint() {return ld*n;}

  double* dev_ptr;
  std::vector<double> host_vector;

  void upload() {
    cudaMemcpy(dev_ptr, host_vector.data(), host_vector.size()*sizeof(double), cudaMemcpyHostToDevice);
  }

  void download() {
    cudaMemcpy(host_vector.data(), dev_ptr, host_vector.size()*sizeof(double), cudaMemcpyDeviceToHost);
  }

  void randomize_host() {
    std::uniform_real_distribution<double> unif(-1.0, 1.0);
    std::default_random_engine re;

    for (auto &x : host_vector) x = unif(re);
  }

  Matrix matrix() {
    return Matrix(Workspace(dev_ptr, footprint()), m, n, ld);
  }

  operator Matrix() {return matrix();}
};

void test_gemm(TestMatrix &A, TestMatrix &B, TestMatrix &C, double alpha, double beta, bool transa, bool transb) {
  auto ixA = [&](int i, int j) {return transa ? i*A.ld+j : j*A.ld+i;};
  auto ixB = [&](int i, int j) {return transb ? i*B.ld+j : j*B.ld+i;};

  int k = transa ? A.m : A.n;
  for (int i = 0; i < C.m; i++) {
    for (int j = 0; j < C.n; j++) {
      C.host_vector[j*C.ld+i] *= beta;
      for (int l = 0; l < k; l++) {
        C.host_vector[j*C.ld+i] += alpha*A.host_vector[ixA(i,l)]*B.host_vector[ixB(l,j)];
      }
    }
  }
}

TEST(Planning_Test, Hello) {
  GEMM_Planner planner;
  cublasHandle_t handle;

  cublasCreate(&handle);
  cudaDeviceSynchronize();

  Stream s;
  cublasSetStream(handle, s);
  //cublasSetStream(handle, s);

  int m = 42;
  int n = 35;
  int k = 60;

  TestMatrix A(m,k,m);
  TestMatrix B(k,n,k);
  TestMatrix C(m,n,m);


  double alpha = 1.0;

  GEMM_Inputs inputs(handle, CUBLAS_OP_N, CUBLAS_OP_N, A, B, C, alpha, 0.0, Workspace());

  GEMM_Options plan(TRANS, NOTRANS);
  std::vector<GEMM_Options> plans;
  plans.emplace_back(NOTRANS,NOTRANS);
  plans.emplace_back(TRANS,NOTRANS);
  plans.emplace_back(NOTRANS,TRANS);
  plans.emplace_back(TRANS,TRANS);

  for (int i=0; i<10; i++) {
    for (auto &plan : plans) {
      size_t ws = planner.calculate_workspace(plan, inputs)*sizeof(double);
      double *space;
      cudaMalloc(&space, ws);
      inputs.space = Workspace(space, ws);

      for (int i=0; i<10; i++)
        planner.execute(plan, inputs, s);

      C.download();
      test_gemm(A, B, C, -alpha, 1.0, false, false);

      for (int i = 0; i < C.m; i++) {
        for (int j = 0; j < C.n; j++) {
          ASSERT_NEAR(C.host_vector[j*C.ld+i], 0.0, 1e-10);
        }
      }
      cudaFree(space);
    }
    //planner.dump_analytics();
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
