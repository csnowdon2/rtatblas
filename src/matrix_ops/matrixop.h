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

  virtual Matrix execute(cublasHandle_t handle, Workspace out_space, Workspace scratch_space) = 0;
  virtual size_t output_space_req()  const = 0;
  virtual MatrixDims dims() const = 0;

  size_t scratch_space_req() const {
    return workspace_req() - output_space_req();    
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


  std::vector<Matrix> compute_operands(cublasHandle_t handle,
                                       Workspace out_space, Workspace scratch_space) {
    // TODO compute operands in decreasing order of space requirements
    std::vector<Matrix> output;
    for (int i = 0; i < (int)operands.size(); i++) {
      auto &operand = operands[i];
      Workspace operand_space;

      if (i == output_operand) {
        operand_space = out_space;
      } else {
        operand_space = scratch_space.peel(operand->output_space_req());
      }

      output.emplace_back(operand->execute(handle, operand_space, scratch_space));
    }

    return output;
  }

  std::vector<Matrix> compute_operands(cublasHandle_t handle, Workspace scratch_space) {
    return compute_operands(handle, Workspace(), scratch_space);
  }
};

class NoOp : public MatrixOp {
  Matrix A;
public:
  NoOp(Matrix A) : MatrixOp({}), A(A) {}

  Matrix execute([[maybe_unused]] cublasHandle_t handle, [[maybe_unused]] Workspace out_space, [[maybe_unused]] Workspace scratch_space) override {
    return A;
  }

  size_t output_space_req()  const override {return 0;}
  MatrixDims dims() const override {return A.dims();}
};

class ScratchMatrix : public MatrixOp {
  size_t m, n, ld;
public:
  ScratchMatrix(MatrixDims dims) : ScratchMatrix(dims.m,dims.n,dims.ld) {}
  ScratchMatrix(size_t m, size_t n, size_t ld) : MatrixOp({}), m(m), n(n), ld(ld) {}
  ScratchMatrix(std::unique_ptr<MatrixOp> &op, int pad) 
    : MatrixOp({}), m(op->dims().m), n(op->dims().n), ld(((op->dims().m+pad-1)/pad)*pad) {}

  // Some of this is replicated from MatrixMove, should think about 
  // how to nicely collapse common things. Maybe make a ConcreteMatrix 
  // class and have both inherit.
  size_t output_space_req() const override {
    return dims().footprint();
  }

  MatrixDims dims() const override {
    return MatrixDims(m,n,ld);
  }

  Matrix execute([[maybe_unused]] cublasHandle_t handle, Workspace out_space, [[maybe_unused]] Workspace scratch_space) override {
    return Matrix(out_space, dims());
  }
};

class MatrixAccumulate : public MatrixOp {
  double alpha, beta;
  bool transpose;
public:
  MatrixAccumulate(std::unique_ptr<MatrixOp> Aop, std::unique_ptr<MatrixOp> Bop,
                   double alpha, double beta, bool transpose) : MatrixOp({}, 1) ,
                     alpha(alpha), beta(beta), transpose(transpose) {
    int Am = Aop->dims().m;
    int An = Aop->dims().n;
    int Bm = Bop->dims().m;
    int Bn = Bop->dims().n;

    operands.push_back(std::move(Aop));
    operands.push_back(std::move(Bop));

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

  MatrixDims dims() const override { return operands[1]->dims(); }

  Matrix execute(cublasHandle_t handle, Workspace out_space, Workspace scratch_space) override {
    if (out_space.size() < output_space_req() || 
        scratch_space.size() < scratch_space_req()) {
      std::cout << "NOT ENOUGH SPACE" << std::endl;
      throw "Not enough space";
    }

    auto matrices = compute_operands(handle, out_space, scratch_space);
    Matrix A = matrices[0];
    Matrix B = matrices[1];

    size_t lda = A.dims().ld;
    size_t ldb = B.dims().ld;
    cublasDgeam(handle,
                transpose ? CUBLAS_OP_T : CUBLAS_OP_N, 
                CUBLAS_OP_N,
                B.dims().m, B.dims().n,
                &alpha,
                A.ptr(), lda,
                &beta,
                B.ptr(), ldb,
                B.ptr(), ldb);
    return B;
  }
};

class MatrixMove : public MatrixOp {
private:
  double alpha;
  bool transpose;
  size_t pad;
public:
  MatrixMove(std::unique_ptr<MatrixOp> Aop, double alpha, bool transpose, size_t pad)
      : MatrixOp({}), alpha(alpha), transpose(transpose), pad(pad) {
    operands.push_back(std::move(Aop));
  }

  size_t output_space_req() const override { return dims().footprint(); }

  MatrixDims dims() const override {
    auto &Aop = operands[0];
    size_t m = transpose ? Aop->dims().n : Aop->dims().m;
    size_t n = transpose ? Aop->dims().m : Aop->dims().n;
    size_t ld = ((m+pad-1)/pad)*pad;
    return MatrixDims(m,n,ld);
  };

  Matrix execute(cublasHandle_t handle, Workspace out_space, Workspace scratch_space) override {
    if (out_space.size() < output_space_req() || 
        scratch_space.size() < scratch_space_req()) {
      std::cout << "NOT ENOUGH SPACE" << std::endl;
      throw "Not enough space";
    }
    auto matrices = compute_operands(handle, out_space, scratch_space);
    Matrix A = matrices[0];
    Matrix B(out_space, dims());

    size_t lda = A.dims().ld;
    size_t ldb = B.dims().ld;
    double beta = 0.0;
    cublasDgeam(handle,
                transpose ? CUBLAS_OP_T : CUBLAS_OP_N, 
                CUBLAS_OP_N,
                B.dims().m, B.dims().n,
                &alpha,
                A.ptr(), lda,
                &beta,
                B.ptr(), ldb,
                B.ptr(), ldb);
    return B;
  }
};


class MatrixMult : public MatrixOp {
  bool transa, transb;
  double alpha, beta;
public:
  MatrixMult(std::unique_ptr<MatrixOp> Aop, std::unique_ptr<MatrixOp> Bop,
             std::unique_ptr<MatrixOp> Cop, bool transa, bool transb, 
             double alpha, double beta) : MatrixOp({}, 2), transa(transa), transb(transb),
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
    
    operands.push_back(std::move(Aop));
    operands.push_back(std::move(Bop));
    operands.push_back(std::move(Cop));
  }

  MatrixDims dims() const override {
    auto &Cop = operands[2];
    return Cop->dims();
  }

  size_t output_space_req() const override {return 0;}

  Matrix execute(cublasHandle_t handle, Workspace out_space, Workspace scratch_space) override {

    auto opA = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto opB = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto matrices = compute_operands(handle, out_space, scratch_space);

    Matrix &A = matrices[0];
    Matrix &B = matrices[1];
    Matrix &C = matrices[2];

    int m = C.dims().m;
    int n = C.dims().n;
    int k = transa ? A.dims().m : A.dims().n;
    cublasDgemm(handle,
                opA, opB,
                m, n, k,
                &alpha,
                A.ptr(), A.dims().ld,
                B.ptr(), B.dims().ld,
                &beta,
                C.ptr(), C.dims().ld);
    return C;
  }

};

class MatrixMultAlloc : public MatrixOp {
  bool transa, transb;
  double alpha;
  size_t pad;
public:
  MatrixMultAlloc(std::unique_ptr<MatrixOp> Aop, std::unique_ptr<MatrixOp> Bop,
                  bool transa, bool transb, double alpha, size_t pad) 
              : MatrixOp({}), transa(transa), transb(transb), alpha(alpha), pad(pad) {
    int kA = transa ? Aop->dims().m : Aop->dims().n;
    int kB = transb ? Bop->dims().n : Bop->dims().m;
    if (kA != kB) {
      std::cout << "Bad matrix mult, kA=" << kA << " kB=" << kB << std::endl;
      throw;
    }
    
    operands.push_back(std::move(Aop));
    operands.push_back(std::move(Bop));
  }

  MatrixDims dims() const override {
    auto &Aop = operands[0];
    auto &Bop = operands[1];
    size_t m = transa ? Aop->dims().n : Aop->dims().m;
    size_t n = transb ? Bop->dims().m : Bop->dims().n;
    size_t ld = ((m+pad-1)/pad)*pad;
    return MatrixDims(m,n,ld);
  }

  size_t output_space_req() const override {return dims().footprint();}

  Matrix execute(cublasHandle_t handle, Workspace out_space, Workspace scratch_space) override {
    auto opA = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto opB = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto matrices = compute_operands(handle, out_space, scratch_space);

    Matrix &A = matrices[0];
    Matrix &B = matrices[1];
    Matrix C(out_space, dims());

    int m = C.dims().m;
    int n = C.dims().n;
    int k = transa ? A.dims().m : A.dims().n;
    double beta = 0.0;
    cublasDgemm(handle,
                opA, opB,
                m, n, k,
                &alpha,
                A.ptr(), A.dims().ld,
                B.ptr(), B.dims().ld,
                &beta,
                C.ptr(), C.dims().ld);
    return C;
  }

};


class BatchMatrixMult : public MatrixOp {
  bool transa, transb;
  double alpha, beta;
public:
  BatchMatrixMult(std::unique_ptr<MatrixOp> Aop, std::unique_ptr<MatrixOp> Bop,
             std::unique_ptr<MatrixOp> Cop, bool transa, bool transb, 
             double alpha, double beta) : MatrixOp({}, 2), transa(transa), transb(transb),
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
    
    operands.push_back(std::move(Aop));
    operands.push_back(std::move(Bop));
    operands.push_back(std::move(Cop));
  }

  MatrixDims dims() const override {
    auto &Cop = operands[2];
    return Cop->dims();
  }

  size_t output_space_req() const override {return 0;}

  Matrix execute(cublasHandle_t handle, Workspace out_space, Workspace scratch_space) override {

    auto opA = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto opB = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto matrices = compute_operands(handle, out_space, scratch_space);

    Matrix &A = matrices[0];
    Matrix &B = matrices[1];
    Matrix &C = matrices[2];

    int m = C.dims().m;
    int n = C.dims().n;
    int k = transa ? A.dims().m : A.dims().n;

    const int mblock = 4;
    const int block_count = m/mblock;
    if (m % mblock != 0) throw "aaa";

    std::vector<double*> Ablocks;
    std::vector<double*> Bblocks;
    std::vector<double*> Cblocks;
    for (int i = 0; i < m; i += mblock) {
      Ablocks.push_back(&A.ptr()[i]);
      Bblocks.push_back(B.ptr());
      Cblocks.push_back(&C.ptr()[i]);
    }

    cublasDgemmBatched(handle, opA, opB,
                       mblock, n, k,
                       &alpha,
                       Ablocks.data(), A.dims().ld,
                       Bblocks.data(), B.dims().ld,
                       &beta,
                       Cblocks.data(), C.dims().ld,
                       block_count);
    return C;
  }
};

}
