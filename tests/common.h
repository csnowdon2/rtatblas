#pragma once
#include <gtest/gtest.h>
#include <gpu-api.h>
#include <matrixop.h>
#include <random>

using namespace rtat;

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
  ManagedWorkspace(size_t bytes) : Workspace() {
    gpuAssert(cudaMalloc(&ptr, bytes));
    count = bytes;
  }
  ~ManagedWorkspace() { gpuAssert(cudaFree(ptr)); }

  template<typename T>
  void grow_to_fit(size_t new_count) {
    if (size<T>() < new_count) {
      gpuAssert(cudaDeviceSynchronize());
      gpuAssert(cudaFree(ptr));
      gpuAssert(cudaMalloc(&ptr, new_count));
      count = new_count;
    }
  }
};

template<typename T>
class TestMatrix {
public:
  TestMatrix(size_t m, size_t n) : TestMatrix(m,n,m) {}

  TestMatrix(size_t m, size_t n, size_t ld) : m(m), n(n), ld(ld), space(footprint()*sizeof(T)), 
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

  std::vector<T> host_vector;

  void upload() {
    gpuAssert(cudaMemcpy(space, host_vector.data(), host_vector.size()*sizeof(T), 
                         cudaMemcpyHostToDevice));
  }

  void download() {
    gpuAssert(cudaMemcpy(host_vector.data(), space, host_vector.size()*sizeof(T), 
                         cudaMemcpyDeviceToHost));
  }

  void randomize_host() {
    std::uniform_real_distribution<T> unif(-1.0, 1.0);
    static std::default_random_engine re;

    for (auto &x : host_vector) x = unif(re);
  }
  
  void zero_host() {
    for (auto &x : host_vector) x = 0.0;
  }

  void insert(Matrix<T> A) {
    if (A.dims().m != m || A.dims().n != n || A.dims().ld != ld) {
      std::cout << "Bad matrix insert" << std::endl;
      throw;
    }

    std::cout << "Size = " << footprint()*sizeof(T) << std::endl;
    gpuAssert(cudaMemcpy(space, A.ptr(), footprint()*sizeof(T), 
                         cudaMemcpyDeviceToDevice));
  }

  Matrix<T> matrix() {
    return Matrix<T>(space, m, n, ld);
  }

  friend bool operator==(const TestMatrix &A, const TestMatrix &B) {
    T epsilon = 0;
    if constexpr(std::is_same_v<T,float>) {
      epsilon = 1e-4;
    } else if constexpr(std::is_same_v<T,double>) {
      epsilon = 1e-10;
    }

    if (A.m != B.m || A.n != B.n) return false;

    for (size_t i = 0; i < A.m; i++) 
      for (size_t j = 0; j < A.n; j++) 
        if (abs(A.host_vector[j*A.ld+i]-B.host_vector[j*B.ld+i]) > epsilon) return false;

    return true;
  }

  bool is_zero() {
    T epsilon = 0;
    if constexpr(std::is_same_v<T,float>) {
      epsilon = 1e-4;
    } else if constexpr(std::is_same_v<T,double>) {
      epsilon = 1e-10;
    }
    for (size_t i = 0; i < m; i++) 
      for (size_t j = 0; j < n; j++) 
        if (abs(host_vector[j*ld+i]) > epsilon) return false;
    return true;
  }

  void print() {
    for (size_t i = 0; i < ld; i++) {
      for (size_t j = 0; j < n; j++) {
        std::cout << host_vector[j*ld+i] << " ";
      }
      std::cout << std::endl;
    }
  }

  Workspace workspace() {return space;}

  operator Matrix<T>() {return matrix();}
};

template<typename T>
inline void test_gemm(TestMatrix<T> &A, TestMatrix<T> &B, TestMatrix<T> &C, T alpha, T beta, bool transa, bool transb) {
  auto ixA = [&](int i, int j) {return transa ? i*A.ld+j : j*A.ld+i;};
  auto ixB = [&](int i, int j) {return transb ? i*B.ld+j : j*B.ld+i;};

  size_t k = transa ? A.m : A.n;
  // Error check
  {
    size_t k_B = transb ? B.n : B.m;
    if (k != k_B) {
      std::cout << "test_gemm k mismatch " << k << "=/=" << k_B << std::endl;
      throw("test_gemm k mismatch");
    }

    size_t m = transa ? A.n : A.m;
    if (m != C.m) {
      std::cout << "test_gemm m mismatch " << m << "=/=" << C.m << std::endl;
      throw("test_gemm m mismatch");
    }

    size_t n = transb ? B.m : B.n;
    if (n != C.n) {
      std::cout << "test_gemm n mismatch " << n << "=/=" << C.n << std::endl;
      throw("test_gemm n mismatch");
    }
  }

  for (size_t i = 0; i < C.m; i++) {
    for (size_t j = 0; j < C.n; j++) {
      C.host_vector[j*C.ld+i] *= beta;
      for (size_t l = 0; l < k; l++) {
        C.host_vector[j*C.ld+i] += alpha*A.host_vector[ixA(i,l)]*B.host_vector[ixB(l,j)];
      }
    }
  }
}

template<typename T>
inline void test_trsm(TestMatrix<T> &A, TestMatrix<T> &B, 
                      bool side_left, bool lower, bool, 
                      bool trans, T alpha) {
  auto ixA = [&](size_t i, size_t j) -> T& {return trans ? A.host_vector[j*A.ld+i] : A.host_vector[i*A.ld+j];};
  auto ixB = [&](size_t i, size_t j) -> T& {
    //std::cout << i << " " << j << " ld=" << B.ld <<  std::endl;
    //if (i >= B.n) throw("ABDOUA");
    //if (j >= B.m) throw("ABDOUA");
    return B.host_vector[i*B.ld+j];
  };
  if (trans) lower = !lower;

  int n = B.n;
  int m = B.m;
  if (side_left && !lower) {
    for (int j=0; j<n; j++) {
      for (int i=0; i<m; i++) {
        ixB(j,i) *= alpha;
      }

      for (int k=m-1; k>=0; k--) {
        ixB(j,k) /= ixA(k,k);

        for (int i=0; i<k; i++) {
          ixB(j,i) -= ixB(j,k)*ixA(k,i);
        }
      }
    }
  } else if (side_left && lower) {
    for (int j=0; j<n; j++) {
      for (int i=0; i<m; i++) {
        ixB(j,i) *= alpha;
      }

      for (int k=0; k<m; k++) {
        ixB(j,k) /= ixA(k,k);

        for (int i=k+1; i<m; i++) {
          ixB(j,i) -= ixB(j,k)*ixA(k,i);
        }
      }
    }
  } else if (!side_left && !lower) {
    for (int j=0; j<n; j++) {
      for (int i=0; i<m; i++) {
        ixB(j,i) *= alpha;
      }

      for (int k=0; k<j; k++) {
        for (int i=0; i<m; i++) {
          std::cout << "HELLO" << std::endl;
          std::cout << ixB(j,i) << " -= " << ixB(k,i) << "*" << ixA(j,k) << std::endl;
          ixB(j,i) -= ixB(k,i)*ixA(j,k);
        }
      }
      for (int i=0; i<m; i++)
        ixB(j,i) /= ixA(j,j);
    }
  } else if (!side_left && lower) {
    for (int j=n-1; j>=0; j--) {
      for (int i=0; i<m; i++) {
        ixB(j,i) *= alpha;
      }

      for (int k=j+1; k<n; k++) {
        for (int i=0; i<m; i++) {
          ixB(j,i) -= ixB(k,i)*ixA(j,k);
        }
      }
      for (int i=0; i<m; i++)
        ixB(j,i) /= ixA(j,j);
    }
  }
}
