#include <gtest/gtest.h>
#include <random>
#include <matrixop.h>
#include <cuda_runtime_api.h>
#include <iostream>

std::pair<Workspace, Workspace> allocate_workspace(MatrixOp &op) {
  double *output_ptr, *scratch_ptr;
  size_t out_size = op.output_space_req();
  size_t scratch_size = op.scratch_space_req();
  cudaMalloc((void**)&output_ptr, out_size*sizeof(double));
  cudaMalloc((void**)&scratch_ptr, scratch_size*sizeof(double));
  Workspace output(output_ptr, out_size);
  Workspace scratch(scratch_ptr, scratch_size);
  return std::make_pair(output, scratch);
}


TEST(MatrixOp_Test, Hello) {
  const int n = 12;
  const int m = 10;
  std::vector<double> Ah(m*n);
  std::vector<double> Bh(m*n);

  cublasHandle_t handle;
  cublasCreate(&handle);

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      Ah[j*m+i] = i*100+j;
    }
  }

  double *Ad, *Bd, *scratch;
  cudaMalloc((void**)&Ad, m*n*sizeof(double));

  cudaMemcpy(Ad, Ah.data(), m*n*sizeof(double), cudaMemcpyHostToDevice);


  Matrix A(Workspace(Ad, m*n), m, n, m);
  std::unique_ptr<MatrixOp> Aop = std::make_unique<NoOp>(A);

  MatrixMove Bop(std::move(Aop), 1.0, true, 1);

  size_t output_size = Bop.output_space_req()*sizeof(double);
  cudaMalloc((void**)&Bd, output_size);

  size_t scratch_size = Bop.scratch_space_req()*sizeof(double);
  cudaMalloc((void**)&scratch, scratch_size);

  Matrix B = Bop.execute(handle, Workspace(Bd, output_size), Workspace(scratch, scratch_size));
  cudaMemcpy(Bh.data(), B.ptr(), output_size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      std::cout << Ah[j*m+i] << " ";

      //std::cout << Ah[i*n+j] << " " << Bh[i*n+j] << std::endl;
      //EXPECT_EQ(Ah[i*n+j], Bh[j*m+i]);
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < m; i++) {
      std::cout << Bh[i*n+j] << " ";

      //std::cout << Ah[i*n+j] << " " << Bh[i*n+j] << std::endl;
      //EXPECT_EQ(Ah[i*n+j], Bh[j*m+i]);
    }
    std::cout << std::endl;
  }
  cudaFree(Ad);
  cudaFree(Bd);
  cudaFree(scratch);
  cublasDestroy(handle);
}

TEST(MatrixOp_Test, MatMulTest) {
  int m = 5;
  int k = 4;
  int n = 3;

  std::vector<double> hA(m*k, 0);
  std::vector<double> hB(n*k, 0);
  std::vector<double> hC(m*n, 0);
  std::vector<double> hC_control(m*n, 0);

  cublasHandle_t handle;
  cublasCreate(&handle);

  {
    std::uniform_real_distribution<double> unif(-1.0, 1.0);
    std::default_random_engine re;

    for (auto &x : hA) x = unif(re);
    for (auto &x : hB) x = unif(re);
  }

  for (int i = 0; i < m; i++) 
    for (int j = 0; j < n; j++) 
      for (int l = 0; l < k; l++) 
        hC_control[j*m+i] += hA[l*m+i]*hB[j*k+l];


  double *dA, *dB, *dC;
  cudaMalloc((void**)&dA, hA.size()*sizeof(double));
  cudaMalloc((void**)&dB, hB.size()*sizeof(double));
  cudaMalloc((void**)&dC, hC.size()*sizeof(double));

  cudaMemcpy(dA, hA.data(), hA.size()*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB.data(), hB.size()*sizeof(double), cudaMemcpyHostToDevice);

  // do the multiply
  {
    Matrix A(Workspace(dA, m*k), m, k, m);
    Matrix B(Workspace(dB, n*k), k, n, k);
    Matrix C(Workspace(dC, m*n), m, n, m);

    std::unique_ptr<MatrixOp> Aop = std::make_unique<NoOp>(A);
    std::unique_ptr<MatrixOp> Bop = std::make_unique<NoOp>(B);
    std::unique_ptr<MatrixOp> Cop = std::make_unique<NoOp>(C);

    MatrixMult mult(std::move(Aop), std::move(Bop), std::move(Cop), false, false, 1.0, 0.0);
    ASSERT_EQ(mult.workspace_req(), 0);
    mult.execute(handle, Workspace(), Workspace());

    cudaMemcpy(hC.data(), dC, hC.size()*sizeof(double), cudaMemcpyDeviceToHost);
  }

  for (int i = 0; i < m; i++) 
    for (int j = 0; j < n; j++) 
      EXPECT_NEAR(hC[j*m+i], hC_control[j*m+i], 1e-10);

}

TEST(MatrixOp_Test, TNMulTest) {
  int m = 5;
  int k = 4;
  int n = 3;

  std::vector<double> hA(m*k, 0);
  std::vector<double> hB(n*k, 0);
  std::vector<double> hC(m*n, 0);

  cublasHandle_t handle;
  cublasCreate(&handle);

  {
    std::uniform_real_distribution<double> unif(-1.0, 1.0);
    std::default_random_engine re;

    for (auto &x : hA) x = unif(re);
    for (auto &x : hB) x = unif(re);
  }

  double *dA, *dB, *dC;
  cudaMalloc((void**)&dA, hA.size()*sizeof(double));
  cudaMalloc((void**)&dB, hB.size()*sizeof(double));
  cudaMalloc((void**)&dC, hC.size()*sizeof(double));

  cudaMemcpy(dA, hA.data(), hA.size()*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB.data(), hB.size()*sizeof(double), cudaMemcpyHostToDevice);

  // do the multiply
  {
    Matrix A(Workspace(dA, m*k), k, m, k);
    Matrix B(Workspace(dB, n*k), k, n, k);
    Matrix C(Workspace(dC, m*n), m, n, m);

    std::unique_ptr<MatrixOp> Aop = std::make_unique<NoOp>(A);
    std::unique_ptr<MatrixOp> Bop = std::make_unique<NoOp>(B);
    std::unique_ptr<MatrixOp> Cop = std::make_unique<NoOp>(C);

    Cop = 
        std::make_unique<MatrixMult>(std::move(Aop), std::move(Bop), std::move(Cop), 
                                     true, false, 2.0, 0.0);

    Aop = std::make_unique<NoOp>(A);
    Bop = std::make_unique<NoOp>(B);

    Aop = transpose_matrix(std::move(Aop), -2.0, 8);

    Cop = std::make_unique<MatrixMult>(std::move(Aop), std::move(Bop), std::move(Cop),
                    false, false, 1.0, 1.0);

    Workspace output, scratch;
    std::tie(output, scratch) = allocate_workspace(*Cop);
    Cop->execute(handle, output, scratch);

    cudaMemcpy(hC.data(), dC, hC.size()*sizeof(double), cudaMemcpyDeviceToHost);
  }

  for (int i = 0; i < m; i++) 
    for (int j = 0; j < n; j++) 
      EXPECT_NEAR(hC[j*m+i], 0.0, 1e-10);
}

TEST(MatrixOp_Test, TTMulTest) {
  int m = 5;
  int k = 4;
  int n = 3;

  std::vector<double> hA(m*k, 0);
  std::vector<double> hB(n*k, 0);
  std::vector<double> hC(m*n, 0);

  cublasHandle_t handle;
  cublasCreate(&handle);

  {
    std::uniform_real_distribution<double> unif(-1.0, 1.0);
    std::default_random_engine re;

    for (auto &x : hA) x = unif(re);
    for (auto &x : hB) x = unif(re);
  }

  double *dA, *dB, *dC;
  cudaMalloc((void**)&dA, hA.size()*sizeof(double));
  cudaMalloc((void**)&dB, hB.size()*sizeof(double));
  cudaMalloc((void**)&dC, hC.size()*sizeof(double));

  cudaMemcpy(dA, hA.data(), hA.size()*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB.data(), hB.size()*sizeof(double), cudaMemcpyHostToDevice);

  // do the multiply
  {
    Matrix A(Workspace(dA, m*k), k, m, k);
    Matrix B(Workspace(dB, n*k), n, k, n);
    Matrix C(Workspace(dC, m*n), m, n, m);

    std::unique_ptr<MatrixOp> Aop = std::make_unique<NoOp>(A);
    std::unique_ptr<MatrixOp> Bop = std::make_unique<NoOp>(B);
    std::unique_ptr<MatrixOp> Cop = std::make_unique<NoOp>(C);

    Cop = 
        std::make_unique<MatrixMult>(std::move(Aop), std::move(Bop), std::move(Cop), 
                                     true, true, 2.0, 0.0);

    Aop = std::make_unique<NoOp>(A);
    Bop = std::make_unique<NoOp>(B);

    Aop = transpose_matrix(std::move(Aop), 0.5, 16);
    Bop = transpose_matrix(std::move(Bop), -4.0, 8);

    Cop = std::make_unique<MatrixMult>(std::move(Aop), std::move(Bop), std::move(Cop),
                    false, false, 1.0, 1.0);

    Workspace output, scratch;
    std::tie(output, scratch) = allocate_workspace(*Cop);
    Cop->execute(handle, output, scratch);

    cudaMemcpy(hC.data(), dC, hC.size()*sizeof(double), cudaMemcpyDeviceToHost);
  }

  for (int i = 0; i < m; i++) 
    for (int j = 0; j < n; j++) 
      EXPECT_NEAR(hC[j*m+i], 0.0, 1e-10);
}
