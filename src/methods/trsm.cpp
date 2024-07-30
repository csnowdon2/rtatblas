#include "trsm.h"
#include "gpu-api.h"
#include "matrixop.h"
#include <sstream>
using namespace rtat;

// TRSM_Key implementation
TRSM_Key::operator std::string() const {
  std::stringstream ss;
  ss << side << "," << uplo << "," << trans << "," << diag
     << "," << m << "," << n;

  std::string ret;
  ss >> ret;
  return ret;
}

bool TRSM_Key::operator<(const TRSM_Key& rhs) const {
  return std::string(*this) < std::string(rhs);
}

std::ostream& operator<<(std::ostream& os, const TRSM_Key& dt) {
    os << std::string(dt);
    return os;
}


// TRSM_Options implementation
std::vector<TRSM_Options> TRSM_Options::enumerate() {
  std::vector<TRSM_Options> ret;

  for (auto swap : {Bool_Op(false), Bool_Op(true)})
    for (auto trans : {Bool_Op(false), Bool_Op(true)})
      ret.push_back(TRSM_Options(swap,trans));
  return ret;
}

TRSM_Options::operator std::string() const {
  std::stringstream ss;
  ss << std::string(swap_side);
  ss << std::string(transpose_A);

  std::string ret;
  ss >> ret;
  return ret;
}

bool TRSM_Options::operator<(const TRSM_Options& o) const {
  return std::string(*this) < std::string(o);
}

std::ostream& operator<<(std::ostream& os, const TRSM_Options opts) {
  os << std::string(opts); 
  return os;
}

std::istream& operator>>(std::istream &is, TRSM_Options &opts) {
  std::string s;
  is >> s;
  if (s.size() != 2) {
    is.setstate(std::ios::failbit);
    return is;
  }
    
  opts.swap_side = Bool_Op(s[0]);
  opts.transpose_A = Bool_Op(s[1]);

  return is;
}

template<typename T>
std::unique_ptr<MatrixOp<T>> TRSM_Options::form_operation(TRSM_Inputs<T> params) {

  std::unique_ptr<MatrixOp<T>> A = std::make_unique<NoOp<T>>(params.A);
  std::unique_ptr<MatrixOp<T>> B = std::make_unique<NoOp<T>>(params.B);

  if (transpose_A) {
    params.trans = !params.trans;
    params.uplo = !params.uplo;
    A = std::make_unique<MatrixMove<T>>(
        std::move(A), 1.0, true, 1);
  }

  if (swap_side) {
    // Transpose B
    params.trans = !params.trans;
    params.side = !params.side;
    std::unique_ptr<MatrixOp<T>> scratch = 
      std::make_unique<MatrixMove<T>>(std::move(B), 1.0, true, 1);
    scratch = std::make_unique<MatrixTrsAlloc<T>>(
        std::move(A), std::move(scratch), 
        params.side == CUBLAS_SIDE_LEFT,
        params.uplo == CUBLAS_FILL_MODE_LOWER,
        params.trans == CUBLAS_OP_T,
        params.diag == CUBLAS_DIAG_UNIT,
        params.alpha);

    B = std::make_unique<NoOp<T>>(params.B);

    return std::make_unique<MatrixAccumulate<T>>(
        std::move(scratch), std::move(B), 1.0, 0.0, true);
  } else {
    return std::make_unique<MatrixTrs<T>>(
        std::move(A), std::move(B), 
        params.side == CUBLAS_SIDE_LEFT,
        params.uplo == CUBLAS_FILL_MODE_LOWER,
        params.trans == CUBLAS_OP_T,
        params.diag == CUBLAS_DIAG_UNIT,
        params.alpha);
  }
}

template std::unique_ptr<MatrixOp<double>> 
  TRSM_Options::form_operation(TRSM_Inputs<double>);

template std::unique_ptr<MatrixOp<float>> 
  TRSM_Options::form_operation(TRSM_Inputs<float>);

