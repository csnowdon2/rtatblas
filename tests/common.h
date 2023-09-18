#pragma once
#include <gpu-api.h>

class BLAS_Test : public ::testing::Test {
protected:
  virtual ~BLAS_Test() = default;

  Stream s;
  cublasHandle_t handle;
  virtual void SetUp() {
    cublasCreate(&handle);
    cublasSetStream(handle, s);
  }
  virtual void TearDown() {
    cublasDestroy(handle);
  }
};

class ManagedWorkspace : public Workspace {
public:
  ManagedWorkspace(size_t doubles) : Workspace() {
    gpuAssert(cudaMalloc(&ptr, doubles*sizeof(double)));
    count = doubles;
  }
  ~ManagedWorkspace() { gpuAssert(cudaFree(ptr)); }

  void grow_to_fit(size_t doubles) {
    if (size() < doubles) {
      gpuAssert(cudaDeviceSynchronize());
      gpuAssert(cudaFree(ptr));
      gpuAssert(cudaMalloc(&ptr, doubles));
      count = doubles;
    }
  }
};

class TestMatrix {
public:
  TestMatrix(size_t m, size_t n) : TestMatrix(m,n,m) {}

  TestMatrix(size_t m, size_t n, size_t ld) : m(m), n(n), ld(ld), space(footprint()), 
                                              host_vector(footprint()) {
    if (ld < m) throw "Bad matrix ld";

    randomize_host();
    upload();
  }

  size_t m;
  size_t n;
  size_t ld;
  ManagedWorkspace space;

  size_t footprint() {return ld*n;}

  std::vector<double> host_vector;

  void upload() {
    gpuAssert(cudaMemcpy(space, host_vector.data(), host_vector.size()*sizeof(double), 
                         cudaMemcpyHostToDevice));
  }

  void download() {
    gpuAssert(cudaMemcpy(host_vector.data(), space, host_vector.size()*sizeof(double), 
                         cudaMemcpyDeviceToHost));
  }

  void randomize_host() {
    std::uniform_real_distribution<double> unif(-1.0, 1.0);
    std::default_random_engine re;

    for (auto &x : host_vector) x = unif(re);
  }
  
  void zero_host() {
    for (auto &x : host_vector) x = 0.0;
  }

  void insert(Matrix A) {
    if (A.dims().m != m || A.dims().n != n || A.dims().ld != ld) {
      std::cout << "Bad matrix insert" << std::endl;
      throw;
    }

    std::cout << "Size = " << footprint()*sizeof(double) << std::endl;
    gpuAssert(cudaMemcpy(space, A.ptr(), footprint()*sizeof(double), 
                         cudaMemcpyDeviceToDevice));
  }

  Matrix matrix() {
    return Matrix(space, m, n, ld);
  }

  friend bool operator==(const TestMatrix &A, const TestMatrix &B) {
    if (A.m != B.m || A.n != B.n) return false;

    for (int i = 0; i < A.m; i++) 
      for (int j = 0; j < A.n; j++) 
        if (abs(A.host_vector[j*A.ld+i]-B.host_vector[j*B.ld+i]) > 1e-10) return false;

    return true;
  }

  bool is_zero() {
    for (int i = 0; i < m; i++) 
      for (int j = 0; j < n; j++) 
        if (abs(host_vector[j*ld+i]) > 1e-10) return false;
    return true;
  }

  void print() {
    for (int i = 0; i < ld; i++) {
      for (int j = 0; j < n; j++) {
        std::cout << host_vector[j*ld+i] << " ";
      }
      std::cout << std::endl;
    }
  }

  Workspace workspace() {return space;}

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
