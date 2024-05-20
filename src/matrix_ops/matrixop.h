#pragma once
#include <gpu-api.h>
#include "matrix.h"
#include <iostream>
#include <vector>
#include <memory>

namespace rtat {

// Represents a matrix that may not have been computed yet, an 
// abstract description of a matrix computation. It can be 
// concretized by providing space for the matrix and an execution 
// context.

template<typename T>
inline cublasStatus_t gpuTgemm(cublasHandle_t handle, 
                               bool transa, bool transb,
                               Matrix<T> A, Matrix<T> B, Matrix<T> C,
                               const T alpha, const T beta) {
  int m = C.dims().m;
  int n = C.dims().n;
  int k = transa ? A.dims().m : A.dims().n;
  if constexpr(std::is_same_v<T,double>) {
    return cublasDgemm(handle,
                transa ? CUBLAS_OP_T : CUBLAS_OP_N,
                transb ? CUBLAS_OP_T : CUBLAS_OP_N,
                m, n, k,
                &alpha,
                A.ptr(), A.dims().ld,
                B.ptr(), B.dims().ld,
                &beta,
                C.ptr(), C.dims().ld);
  } else if constexpr(std::is_same_v<T,float>) {
    return cublasSgemm(handle,
                transa ? CUBLAS_OP_T : CUBLAS_OP_N,
                transb ? CUBLAS_OP_T : CUBLAS_OP_N,
                m, n, k,
                &alpha,
                A.ptr(), A.dims().ld,
                B.ptr(), B.dims().ld,
                &beta,
                C.ptr(), C.dims().ld);
  } else {
    static_assert(!sizeof(T), "GEMM is only double and float");
  }
}

template<typename T>
inline cublasStatus_t gpuTtrsm(cublasHandle_t handle, 
                               bool side_left, bool lower, 
                               bool trans, bool unit_diag,
                               Matrix<T> A, Matrix<T> B, 
                               const T alpha) {
  int m = B.dims().m;
  int n = B.dims().n;
  if constexpr(std::is_same_v<T,double>) {
    return cublasDtrsm(handle,
                side_left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT,
                lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER,
                trans ? CUBLAS_OP_T : CUBLAS_OP_N,
                unit_diag ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
                m, n,
                &alpha,
                A.ptr(), A.dims().ld,
                B.ptr(), B.dims().ld);
  } else if constexpr(std::is_same_v<T,float>) {
    return cublasStrsm(handle,
                side_left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT,
                lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER,
                trans ? CUBLAS_OP_T : CUBLAS_OP_N,
                unit_diag ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
                m, n,
                &alpha,
                A.ptr(), A.dims().ld,
                B.ptr(), B.dims().ld);
  } else {
    static_assert(!sizeof(T), "TRSM is only double and float");
  }
}

template<typename T>
inline cublasStatus_t gpuTgeam(cublasHandle_t handle, 
                               bool transa, bool transb,
                               Matrix<T> A, Matrix<T> B, Matrix<T> C,
                               const T alpha, 
                               const T beta) {
  if constexpr(std::is_same_v<T,double>) {
    return cublasDgeam(handle,
                transa ? CUBLAS_OP_T : CUBLAS_OP_N, 
                transb ? CUBLAS_OP_T : CUBLAS_OP_N, 
                B.dims().m, B.dims().n,
                &alpha,
                A.ptr(), A.dims().ld,
                &beta,
                B.ptr(), B.dims().ld,
                C.ptr(), C.dims().ld);
  } else if constexpr(std::is_same_v<T,float>) {
    return cublasSgeam(handle,
                transa ? CUBLAS_OP_T : CUBLAS_OP_N, 
                transb ? CUBLAS_OP_T : CUBLAS_OP_N, 
                B.dims().m, B.dims().n,
                &alpha,
                A.ptr(), A.dims().ld,
                &beta,
                B.ptr(), B.dims().ld,
                C.ptr(), C.dims().ld);
  } else {
    static_assert(!sizeof(T), "GEAM is only double and float");
  }
}

template<typename T>
class MatrixOp {
protected:
  std::vector<std::unique_ptr<MatrixOp>> operands;
  int output_operand = -1;
public:
  MatrixOp(const MatrixOp&) = delete;
  MatrixOp(MatrixOp&&) = default;
  MatrixOp& operator=(MatrixOp&&) = default;

  MatrixOp(std::vector<std::unique_ptr<MatrixOp>> operands)
          : MatrixOp(std::move(operands), -1) {}

  MatrixOp(std::vector<std::unique_ptr<MatrixOp>> operands, int output_operand)
          : operands(std::move(operands)), output_operand(output_operand) {}

  virtual ~MatrixOp() = default;

  virtual Matrix<T> execute(cublasHandle_t handle, Workspace out_space, Workspace scratch_space) = 0;
  virtual size_t output_space_req()  const = 0;
  virtual MatrixDims dims() const = 0;

  size_t scratch_space_req() const {
    return workspace_req() - output_space_req();    
  }

  size_t scratch_space_req_bytes() const {
    return sizeof(T)*scratch_space_req();
  }

  size_t workspace_req() const { 
    size_t out_space = output_space_req();
    size_t operand_space = 0;
    size_t extra_space = 0;

    for (int i = 0; i < (int)operands.size(); i++) {
      auto &op = operands[i];
      if (i != output_operand)
        operand_space += op->output_space_req();
      extra_space = std::max(extra_space, operand_space + op->scratch_space_req());
    }

    return out_space + extra_space;
  }

  size_t workspace_req_bytes() const { 
    return sizeof(T)*workspace_req();
  }


  std::vector<Matrix<T>> compute_operands(cublasHandle_t handle,
                                       Workspace out_space, Workspace scratch_space) {
    // TODO compute operands in decreasing order of space requirements
    std::vector<Matrix<T>> output;
    for (int i = 0; i < (int)operands.size(); i++) {
      auto &operand = operands[i];
      Workspace operand_space;

      if (i == output_operand) {
        operand_space = out_space;
      } else {
        operand_space = scratch_space.peel<T>(operand->output_space_req());
      }

      output.emplace_back(operand->execute(handle, operand_space, scratch_space));
    }

    return output;
  }

  std::vector<Matrix<T>> compute_operands(cublasHandle_t handle, Workspace scratch_space) {
    return compute_operands(handle, Workspace(), scratch_space);
  }
};

template<typename T>
class NoOp : public MatrixOp<T> {
  Matrix<T> A;
public:
  NoOp(Matrix<T> A) : MatrixOp<T>({}), A(A) {}

  Matrix<T> execute([[maybe_unused]] cublasHandle_t handle, [[maybe_unused]] Workspace out_space, [[maybe_unused]] Workspace scratch_space) override {
    return A;
  }

  size_t output_space_req()  const override {return 0;}
  MatrixDims dims() const override {return A.dims();}
};

template<typename T>
class ScratchMatrix : public MatrixOp<T> {
  size_t m, n, ld;
public:
  ScratchMatrix(MatrixDims dims) : ScratchMatrix(dims.m,dims.n,dims.ld) {}
  ScratchMatrix(size_t m, size_t n, size_t ld) : MatrixOp<T>({}), m(m), n(n), ld(ld) {}
  ScratchMatrix(std::unique_ptr<MatrixOp<T>> &op, int pad) 
    : MatrixOp<T>({}), m(op->dims().m), n(op->dims().n), ld(((op->dims().m+pad-1)/pad)*pad) {}

  // Some of this is replicated from MatrixMove, should think about 
  // how to nicely collapse common things. Maybe make a ConcreteMatrix 
  // class and have both inherit.
  size_t output_space_req() const override {
    return dims().footprint();
  }

  MatrixDims dims() const override {
    return MatrixDims(m,n,ld);
  }

  Matrix<T> execute([[maybe_unused]] cublasHandle_t handle, Workspace out_space, [[maybe_unused]] Workspace scratch_space) override {
    return Matrix<T>(out_space, dims());
  }
};

template<typename T>
class MatrixAccumulate : public MatrixOp<T> {
  T alpha, beta;
  bool transpose;
public:
  MatrixAccumulate(std::unique_ptr<MatrixOp<T>> Aop, std::unique_ptr<MatrixOp<T>> Bop,
                   T alpha, T beta, bool transpose) : MatrixOp<T>({}, 1) ,
                     alpha(alpha), beta(beta), transpose(transpose) {
    int Am = Aop->dims().m;
    int An = Aop->dims().n;
    int Bm = Bop->dims().m;
    int Bn = Bop->dims().n;

    this->operands.push_back(std::move(Aop));
    this->operands.push_back(std::move(Bop));

    bool bad = false;
    if (transpose) {
      bad = bad || (Am != Bn || An != Bm);
    } else {
      bad = bad || (Am != Bm || An != Bn);
    }

    if (bad) {
      std::cout << "Bad matrix accumulate, Adims=" << Am << "," << An
                                      << " Bdims=" << Bm << "," << Bn 
                               << " " << (transpose ? "trans" : "notrans") << std::endl;
      throw;
    }
  }

  size_t output_space_req() const override { return 0; }

  MatrixDims dims() const override { return this->operands[1]->dims(); }

  Matrix<T> execute(cublasHandle_t handle, Workspace out_space, Workspace scratch_space) override {
    if (out_space.size<T>() < output_space_req() || 
        scratch_space.size<T>() < this->scratch_space_req()) {
      std::cout << "NOT ENOUGH SPACE" << std::endl;
      throw "Not enough space";
    }

    auto matrices = this->compute_operands(handle, out_space, scratch_space);
    Matrix<T> A = matrices[0];
    Matrix<T> B = matrices[1];

    gpuTgeam(handle, transpose, false, A, B, B, alpha, beta);
    return B;
  }
};

template<typename T>
class MatrixMove : public MatrixOp<T> {
private:
  T alpha;
  bool transpose;
  size_t pad;
public:
  MatrixMove(std::unique_ptr<MatrixOp<T>> Aop, T alpha, bool transpose, size_t pad)
      : MatrixOp<T>({}), alpha(alpha), transpose(transpose), pad(pad) {
    this->operands.push_back(std::move(Aop));
  }

  size_t output_space_req() const override { return dims().footprint(); }

  MatrixDims dims() const override {
    auto &Aop = this->operands[0];
    size_t m = transpose ? Aop->dims().n : Aop->dims().m;
    size_t n = transpose ? Aop->dims().m : Aop->dims().n;
    size_t ld = ((m+pad-1)/pad)*pad;
    return MatrixDims(m,n,ld);
  };

  Matrix<T> execute(cublasHandle_t handle, Workspace out_space, Workspace scratch_space) override {
    if (out_space.size<T>() < output_space_req() || 
        scratch_space.size<T>() < this->scratch_space_req()) {
      std::cout << "NOT ENOUGH SPACE" << std::endl;
      throw "Not enough space";
    }
    auto matrices = this->compute_operands(handle, out_space, scratch_space);
    Matrix<T> A = matrices[0];
    Matrix<T> B(out_space, dims());

    T beta = 0.0;
    gpuTgeam<T>(handle, transpose, false,
                A, B, B, alpha, beta);
    return B;
  }
};


template<typename T>
class MatrixMult : public MatrixOp<T> {
protected:
  bool transa, transb;
  T alpha, beta;
public:
  MatrixMult(std::unique_ptr<MatrixOp<T>> Aop, std::unique_ptr<MatrixOp<T>> Bop,
             std::unique_ptr<MatrixOp<T>> Cop, bool transa, bool transb, 
             T alpha, T beta) : MatrixOp<T>({}, 2), transa(transa), transb(transb),
                                          alpha(alpha), beta(beta) {
    int kA = transa ? Aop->dims().m : Aop->dims().n;
    int kB = transb ? Bop->dims().n : Bop->dims().m;
    if (kA != kB) {
      std::cout << "Bad matrix mult, kA=" << kA << " kB=" << kB << std::endl;
      throw;
    }
    size_t mA = transa ? Aop->dims().n : Aop->dims().m;
    size_t nB = transb ? Bop->dims().m : Bop->dims().n;
    if (mA != Cop->dims().m || nB != Cop->dims().n) {
      std::cout << "Bad matrix mult, mA=" << mA << ", mC=" << Cop->dims().m
                <<                ", nB=" << nB << ", nC=" << Cop->dims().n << std::endl;
    }
    
    this->operands.push_back(std::move(Aop));
    this->operands.push_back(std::move(Bop));
    this->operands.push_back(std::move(Cop));
  }

  MatrixDims dims() const override {
    auto &Cop = this->operands[2];
    return Cop->dims();
  }

  size_t output_space_req() const override {return 0;}

  virtual Matrix<T> execute(cublasHandle_t handle, Workspace out_space, Workspace scratch_space) override {

    auto matrices = this->compute_operands(handle, out_space, scratch_space);

    Matrix<T> &A = matrices[0];
    Matrix<T> &B = matrices[1];
    Matrix<T> &C = matrices[2];

    gpuTgemm<T>(handle, transa, transb, A, B, C, alpha, beta);
    return C;
  }

};

template<typename T>
class MatrixMultAlloc : public MatrixOp<T> {
  bool transa, transb;
  T alpha;
  size_t pad;
public:
  MatrixMultAlloc(std::unique_ptr<MatrixOp<T>> Aop, std::unique_ptr<MatrixOp<T>> Bop,
                  bool transa, bool transb, T alpha, size_t pad) 
              : MatrixOp<T>({}), transa(transa), transb(transb), alpha(alpha), pad(pad) {
    int kA = transa ? Aop->dims().m : Aop->dims().n;
    int kB = transb ? Bop->dims().n : Bop->dims().m;
    if (kA != kB) {
      std::cout << "Bad matrix mult, kA=" << kA << " kB=" << kB << std::endl;
      throw;
    }
    
    this->operands.push_back(std::move(Aop));
    this->operands.push_back(std::move(Bop));
  }

  MatrixDims dims() const override {
    auto &Aop = this->operands[0];
    auto &Bop = this->operands[1];
    size_t m = transa ? Aop->dims().n : Aop->dims().m;
    size_t n = transb ? Bop->dims().m : Bop->dims().n;
    size_t ld = ((m+pad-1)/pad)*pad;
    return MatrixDims(m,n,ld);
  }

  size_t output_space_req() const override {return dims().footprint();}

  Matrix<T> execute(cublasHandle_t handle, Workspace out_space, Workspace scratch_space) override {
    auto matrices = this->compute_operands(handle, out_space, scratch_space);

    Matrix<T> &A = matrices[0];
    Matrix<T> &B = matrices[1];
    Matrix<T> C(out_space, dims());

    T beta = 0.0;
    gpuTgemm<T>(handle, transa, transb, A, B, C, alpha, beta);
    return C;
  }

};

template<typename T>
class MatrixTrs : public MatrixOp<T> {
protected:
  bool side_left, lower, trans, unit_diag;
  T alpha;
public:
  MatrixTrs(std::unique_ptr<MatrixOp<T>> Aop, std::unique_ptr<MatrixOp<T>> Bop,
      bool side_left, bool lower, bool trans, bool unit_diag,
             T alpha) : MatrixOp<T>({}, 1), side_left(side_left),
                        lower(lower), trans(trans), 
                        unit_diag(unit_diag), alpha(alpha) {
    int nB = Bop->dims().n;
    int mB = Bop->dims().m;
    if ((side_left && mB != Aop->dims().m) || 
        (!side_left && nB != Aop->dims().m) || 
        (Aop->dims().m != Aop->dims().n)) {
      std::cout << "Bad matrix trs, mA=" << Aop->dims().m << " nA=" << Aop->dims().n << std::endl;
      std::cout << "                mB=" << Bop->dims().m << " nB=" << Bop->dims().n << std::endl;
      std::cout << "                " << (side_left ? "LEFT" : "RIGHT") << std::endl;
      throw;
    }
    
    this->operands.push_back(std::move(Aop));
    this->operands.push_back(std::move(Bop));
  }

  MatrixDims dims() const override {
    auto &Bop = this->operands[1];
    return Bop->dims();
  }

  size_t output_space_req() const override {return 0;}

  virtual Matrix<T> execute(cublasHandle_t handle, Workspace out_space, Workspace scratch_space) override {

    auto matrices = this->compute_operands(handle, out_space, scratch_space);

    Matrix<T> &A = matrices[0];
    Matrix<T> &B = matrices[1];

    gpuTtrsm<T>(handle, side_left, lower, trans, unit_diag, A, B, alpha);
    return B;
  }

};


// template<typename T>
// class BatchMatrixMult : public MatrixOp<T> {
//   bool transa, transb;
//   T alpha, beta;
// public:
//   BatchMatrixMult(std::unique_ptr<MatrixOp<T>> Aop, std::unique_ptr<MatrixOp<T>> Bop,
//              std::unique_ptr<MatrixOp<T>> Cop, bool transa, bool transb, 
//              T alpha, T beta) : MatrixOp<T>({}, 2), transa(transa), transb(transb),
//                                           alpha(alpha), beta(beta) {
//     int kA = transa ? Aop->dims().m : Aop->dims().n;
//     int kB = transb ? Bop->dims().n : Bop->dims().m;
//     if (kA != kB) {
//       std::cout << "Bad matrix mult, kA=" << kA << " kB=" << kB << std::endl;
//       throw;
//     }
//     size_t mA = transa ? Aop->dims().n : Aop->dims().m;
//     size_t nB = transb ? Bop->dims().m : Bop->dims().n;
//     if (mA != Cop->dims().m || nB != Cop->dims().n) {
//       std::cout << "Bad matrix mult, mA=" << mA << ", mC=" << Cop->dims().m
//                 <<                ", nB=" << nB << ", nC=" << Cop->dims().n << std::endl;
//     }
//     
//     this->operands.push_back(std::move(Aop));
//     this->operands.push_back(std::move(Bop));
//     this->operands.push_back(std::move(Cop));
//   }
// 
//   MatrixDims dims() const override {
//     auto &Cop = this->operands[2];
//     return Cop->dims();
//   }
// 
//   size_t output_space_req() const override {return 0;}
// 
//   Matrix<T> execute(cublasHandle_t handle, Workspace out_space, Workspace scratch_space) override {
// 
//     auto opA = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
//     auto opB = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
//     auto matrices = compute_operands(handle, out_space, scratch_space);
// 
//     Matrix<T> &A = matrices[0];
//     Matrix<T> &B = matrices[1];
//     Matrix<T> &C = matrices[2];
// 
//     int m = C.dims().m;
//     int n = C.dims().n;
//     int k = transa ? A.dims().m : A.dims().n;
// 
//     const int mblock = 4;
//     const int block_count = m/mblock;
//     if (m % mblock != 0) throw "aaa";
// 
//     std::vector<T*> Ablocks;
//     std::vector<T*> Bblocks;
//     std::vector<T*> Cblocks;
//     for (int i = 0; i < m; i += mblock) {
//       Ablocks.push_back(&A.ptr()[i]);
//       Bblocks.push_back(B.ptr());
//       Cblocks.push_back(&C.ptr()[i]);
//     }
// 
//     cublasDgemmBatched(handle, opA, opB,
//                        mblock, n, k,
//                        &alpha,
//                        Ablocks.data(), A.dims().ld,
//                        Bblocks.data(), B.dims().ld,
//                        &beta,
//                        Cblocks.data(), C.dims().ld,
//                        block_count);
//     return C;
//   }
// };

template<typename T>
class TiledMatrixMult : public MatrixMult<T> {
  int mblock, nblock, kblock;
public:
  TiledMatrixMult(std::unique_ptr<MatrixOp<T>> Aop, std::unique_ptr<MatrixOp<T>> Bop,
                  std::unique_ptr<MatrixOp<T>> Cop, bool transa, bool transb, 
                  T alpha, T beta, int mblock, int nblock, int kblock) 
        : MatrixMult<T>(std::move(Aop), std::move(Bop), std::move(Cop), transa, transb, alpha, beta),
          mblock(mblock), nblock(nblock), kblock(kblock) {}

  Matrix<T> execute(cublasHandle_t handle, Workspace out_space, Workspace scratch_space) override {

    auto matrices = this->compute_operands(handle, out_space, scratch_space);

    Matrix<T> &A = matrices[0];
    Matrix<T> &B = matrices[1];
    Matrix<T> &C = matrices[2];

    int m = C.dims().m;
    int n = C.dims().n;
    int k = this->transa ? A.dims().m : A.dims().n;

    for (int a = 0; a < m; a += mblock) {
      for (int b = 0; b < n; b += nblock) {
        for (int c = 0; c < k; c += kblock) {
          int msize = std::min(mblock, m-a);
          int nsize = std::min(nblock, n-b);
          int ksize = std::min(kblock, k-c);

          T bet = this->beta;
          if (c > 0) bet = 1.0;
          Matrix<T> Ablock = A.block(a, c, msize, ksize);
          Matrix<T> Bblock = B.block(c, b, ksize, nsize);
          Matrix<T> Cblock = C.block(a, b, msize, nsize);
          gpuTgemm<T>(handle, this->transa, this->transb, 
                      Ablock, Bblock, Cblock, this->alpha, bet);
        }
      }
    }
    return C;
  }
};


}
