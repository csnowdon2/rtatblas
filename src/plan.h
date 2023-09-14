#pragma once
#include <variant>
#include <vector>
#include <map>
#ifdef CUDA
#include <cublas_v2.h>
#else
#include <hipblas.h>
#endif
#include "options.h"


enum BLAS_Op {
  TRANS, NOTRANS
};

BLAS_Op switch_op(BLAS_Op op) {
  switch (op) {
    case TRANS:
      return NOTRANS;
    case NOTRANS:
      return TRANS;
  }
}

using Trans_Opt = Option<BLAS_Op, TRANS, NOTRANS>;

class GEMM_Options : public Options<Trans_Opt, Trans_Opt> {
public:
  using Options<Trans_Opt, Trans_Opt>::Options;
  Trans_Opt transa() const {return std::get<0>(*this);}
  Trans_Opt transb() const {return std::get<1>(*this);}

  friend std::ostream& operator<<(std::ostream& os, const GEMM_Options opts) {
    os << (opts.transa() == NOTRANS ? "N" : "T");
    os << (opts.transb() == NOTRANS ? "N" : "T");
    return os;
  }
};

