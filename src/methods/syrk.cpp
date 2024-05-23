#include "syrk.h"
#include "gpu-api.h"
#include "matrixop.h"
#include <sstream>
using namespace rtat;

// SYRK_Key implementation
SYRK_Key::operator std::string() const {
  std::stringstream ss;
  ss << uplo << "," << trans << "," << n << "," << k;

  std::string ret;
  ss >> ret;
  return ret;
}

bool SYRK_Key::operator<(const SYRK_Key& rhs) const {
  return std::string(*this) < std::string(rhs);
}

std::ostream& rtat::operator<<(std::ostream& os, const SYRK_Key& dt) {
    os << std::string(dt);
    return os;
}


// SYRK_Options implementation
std::vector<SYRK_Options> SYRK_Options::enumerate() {
  std::vector<SYRK_Options> ret;

  for (auto transa : {Bool_Op(false), Bool_Op(true)})
    for (auto transc : {Bool_Op(false), Bool_Op(true)})
      ret.push_back(SYRK_Options(transa,transc));
  return ret;
}

SYRK_Options::operator std::string() const {
  std::stringstream ss;
  ss << std::string(transpose_A);
  ss << std::string(transpose_C);

  std::string ret;
  ss >> ret;
  return ret;
}

bool SYRK_Options::operator<(const SYRK_Options& o) const {
  return std::string(*this) < std::string(o);
}

std::ostream& rtat::operator<<(std::ostream& os, const SYRK_Options opts) {
  os << std::string(opts); 
  return os;
}

std::istream& operator>>(std::istream &is, SYRK_Options &opts) {
  std::string s;
  is >> s;
  if (s.size() != 2) {
    is.setstate(std::ios::failbit);
    return is;
  }
    
  opts.transpose_A = Bool_Op(s[0]);
  opts.transpose_C = Bool_Op(s[1]);

  return is;
}

template<typename T>
std::unique_ptr<MatrixOp<T>> SYRK_Options::form_operation(SYRK_Inputs<T> params) {

  std::unique_ptr<MatrixOp<T>> A = std::make_unique<NoOp<T>>(params.A);
  std::unique_ptr<MatrixOp<T>> C = std::make_unique<NoOp<T>>(params.C);

  if (transpose_A) {
    params.trans = (params.trans == CUBLAS_OP_N)
      ? CUBLAS_OP_T
      : CUBLAS_OP_N;
    A = std::make_unique<MatrixMove<T>>(
        std::move(A), 1.0, true, 1);
  }

  if (transpose_C) {
    // Transpose B
    params.uplo = (params.uplo == CUBLAS_FILL_MODE_UPPER) 
      ? CUBLAS_FILL_MODE_LOWER
      : CUBLAS_FILL_MODE_UPPER;
    std::unique_ptr<MatrixOp<T>> scratch = std::make_unique<MatrixSyrkAlloc<T>>(
        std::move(A), 
        params.uplo == CUBLAS_FILL_MODE_LOWER,
        params.trans == CUBLAS_OP_T,
        params.alpha);

    return std::make_unique<MatrixAccumulate<T>>(
        std::move(scratch), std::move(C), 1.0, params.beta, true);
  } else {
    return std::make_unique<MatrixSyrk<T>>(
        std::move(A), std::move(C), 
        params.uplo == CUBLAS_FILL_MODE_LOWER,
        params.trans == CUBLAS_OP_T,
        params.alpha, params.beta);
  }
}

template std::unique_ptr<MatrixOp<double>> 
  SYRK_Options::form_operation(SYRK_Inputs<double>);

template std::unique_ptr<MatrixOp<float>> 
  SYRK_Options::form_operation(SYRK_Inputs<float>);


