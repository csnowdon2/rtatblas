#include <gtest/gtest.h>
#include <matrixop.h>
#include <cuda_runtime_api.h>
#include <iostream>



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
