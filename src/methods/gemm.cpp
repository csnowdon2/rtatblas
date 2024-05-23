#include "gemm.h"
#include <sstream>
using namespace rtat;

// GEMM_Key implementation
GEMM_Key::operator std::string() const {
  std::stringstream ss;
  ss << transa << ","
     << transb << ","
     << "," << m << "," << k << "," << n;

  std::string ret;
  ss >> ret;
  return ret;
}

bool GEMM_Key::operator<(const GEMM_Key& rhs) const {
  return std::string(*this) < std::string(rhs);
}

std::ostream& rtat::operator<<(std::ostream& os, const GEMM_Key& dt) {
    os << std::string(dt);
    return os;
}


// GEMM_Options implementation
std::vector<GEMM_Options> GEMM_Options::enumerate() {
  std::vector<GEMM_Options> ret;

  for (auto opA : {BLAS_Op::NOTRANS, BLAS_Op::TRANS})
    for (auto opB : {BLAS_Op::NOTRANS, BLAS_Op::TRANS})
      for (auto opC : {BLAS_Op::NOTRANS, BLAS_Op::TRANS})
        for (auto padA : {Pad_Op::NOPAD, Pad_Op::PAD})
          for (auto padB : {Pad_Op::NOPAD, Pad_Op::PAD})
            for (auto padC : {Pad_Op::NOPAD, Pad_Op::PAD})
              ret.push_back(GEMM_Options(opA,padA,opB,padB,opC,padC));
  return ret;
}

GEMM_Options::operator std::string() const {
  std::stringstream ss;
  ss << std::string(transa);
  ss << std::string(transb);
  ss << std::string(transc);
  ss << std::string(pada);
  ss << std::string(padb);
  ss << std::string(padc);

  std::string ret;
  ss >> ret;
  return ret;
}

bool GEMM_Options::operator<(const GEMM_Options& o) const {
  return std::string(*this) < std::string(o);
}

std::ostream& rtat::operator<<(std::ostream& os, const GEMM_Options opts) {
  os << std::string(opts); 
  return os;
}

std::istream& operator>>(std::istream &is, GEMM_Options &opts) {
  std::string s;
  is >> s;
  if (s.size() != 6) {
    is.setstate(std::ios::failbit);
    return is;
  }
    
  opts.transa = BLAS_Op({s[0]});
  opts.pada = Pad_Op({s[3]});
  opts.transb = BLAS_Op({s[1]});
  opts.padb = Pad_Op({s[4]});
  opts.transc = BLAS_Op({s[2]});
  opts.padc = Pad_Op({s[5]});

  return is;
}

template<typename T>
std::unique_ptr<MatrixOp<T>> GEMM_Options::form_operation(GEMM_Inputs<T> params) {

  std::unique_ptr<MatrixOp<T>> A = std::make_unique<NoOp<T>>(params.A);
  std::unique_ptr<MatrixOp<T>> B = std::make_unique<NoOp<T>>(params.B);
  std::unique_ptr<MatrixOp<T>> C = std::make_unique<NoOp<T>>(params.C);

  bool ta = transa == BLAS_Op::TRANS;
  bool tb = transb == BLAS_Op::TRANS;
  bool tc = transc == BLAS_Op::TRANS;
  bool pa = pada == Pad_Op::PAD;
  bool pb = padb == Pad_Op::PAD;
  bool pc = padc == Pad_Op::PAD;

  if (ta) 
    params.transa = !params.transa;
  if (ta || pa)
    A = std::make_unique<MatrixMove<T>>(
        std::move(A), 1.0, ta, pa ? 32 : 1);

  if (tb)
    params.transb = !params.transb;
  if (tb || pb)
    B = std::make_unique<MatrixMove<T>>(
        std::move(B), 1.0, tb, pb ? 32 : 1);

  if (tc) {
    auto scratch = std::make_unique<MatrixMultAlloc<T>>(
        std::move(B), std::move(A), 
        params.transb != CUBLAS_OP_T, 
        params.transa != CUBLAS_OP_T, 
        params.alpha, pc ? 32 : 1);

    return std::make_unique<MatrixAccumulate<T>>(
        std::move(scratch), std::move(C), 
        1.0, params.beta, true);
  } else if (pc) {
    auto scratch = std::make_unique<MatrixMultAlloc<T>>(
        std::move(A), std::move(B),
        params.transa == CUBLAS_OP_T, 
        params.transb == CUBLAS_OP_T, 
        params.alpha, 32);

    return std::make_unique<MatrixAccumulate<T>>(
        std::move(scratch), std::move(C), 
        1.0, params.beta, false);
  } else {
    return std::make_unique<MatrixMult<T>>(
        std::move(A), std::move(B), std::move(C), 
        params.transa == CUBLAS_OP_T, params.transb == CUBLAS_OP_T,
        params.alpha, params.beta);
  }
}

template std::unique_ptr<MatrixOp<double>> 
  GEMM_Options::form_operation(GEMM_Inputs<double>);

template std::unique_ptr<MatrixOp<float>> 
  GEMM_Options::form_operation(GEMM_Inputs<float>);
