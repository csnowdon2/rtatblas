#include <gpu-api.h>
#include "matrix.h"
#include <iostream>
#include <vector>
#include <memory>

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

    for (int i = 0; i < operands.size(); i++) {
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
    for (int i = 0; i < operands.size(); i++) {
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

  Matrix execute(cublasHandle_t handle, Workspace out_space, Workspace scratch_space) override {
    return A;
  }

  size_t output_space_req()  const override {return 0;}
  MatrixDims dims() const override {return A.dims();}
};


// Relocate a matrix in memory
class MatrixMove : public MatrixOp {
  double alpha = 1.0;
  bool transpose = false;
  size_t pad = 32;

public:
  MatrixMove(std::unique_ptr<MatrixOp> Aop, double alpha, bool transpose, size_t pad) 
      : MatrixOp({}), alpha(alpha), transpose(transpose), pad(pad) {
    operands.push_back(std::move(Aop));
  }


  size_t output_space_req() const override {
    return dims().footprint();
  }

  MatrixDims dims() const override {
    auto &Aop = operands[0];
    size_t m = transpose ? Aop->dims().n : Aop->dims().m;
    size_t n = transpose ? Aop->dims().m : Aop->dims().n;
    size_t ld = ((m+pad-1)/pad)*pad;

    return MatrixDims(m,n,ld);
  }

  Matrix execute(cublasHandle_t handle, Workspace out_space, Workspace scratch_space) override {
    if (out_space.size() < output_space_req() || 
        scratch_space.size() < scratch_space_req()) {
      std::cout << "NOT ENOUGH SPACE" << std::endl;
      throw "Not enough space";
    }

    auto matrices = compute_operands(handle, scratch_space);
    Matrix A = matrices[0];
    Matrix B(out_space, dims());

    double beta = 0.0;
    cublasDgeam(handle,
                transpose ? CUBLAS_OP_T : CUBLAS_OP_N, 
                CUBLAS_OP_N,
                B.dims().m, B.dims().n,
                &alpha,
                A.ptr(), A.dims().ld,
                &beta,
                B.ptr(), B.dims().ld,
                B.ptr(), B.dims().ld);
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
    operands.push_back(std::move(Aop));
    operands.push_back(std::move(Bop));
    operands.push_back(std::move(Cop));
  }

  MatrixDims dims() const override {
    auto &Cop = operands[2];
    return Cop->dims();
  }

  size_t output_space_req() const override {return 0;}

  Matrix execute(cublasHandle_t handle, Workspace out_space, Workspace scratch_space) {

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

std::unique_ptr<MatrixOp> transpose_matrix(std::unique_ptr<MatrixOp> matrix, double scale, size_t pad) {
  return std::make_unique<MatrixMove>(std::move(matrix), scale, true, pad);
}

std::unique_ptr<MatrixOp> pad_matrix(std::unique_ptr<MatrixOp> matrix, double scale, size_t pad) {
  return std::make_unique<MatrixMove>(std::move(matrix), scale, false, pad);
}
