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
  TRANS, PAD, NOTRANS
};

char op_to_char(BLAS_Op op) {
  switch (op) {
    case TRANS: return 'T';
    case PAD: return 'P';
    case NOTRANS: return 'N';
  }
}

using Trans_Opt = Option<BLAS_Op, TRANS, PAD, NOTRANS>;

class GEMM_Options : public Options<Trans_Opt, Trans_Opt> {
public:
  using Options<Trans_Opt, Trans_Opt>::Options;
  Trans_Opt transa() const {return std::get<0>(*this);}
  Trans_Opt transb() const {return std::get<1>(*this);}

  friend std::ostream& operator<<(std::ostream& os, const GEMM_Options opts) {
    os << op_to_char(opts.transa());
    os << op_to_char(opts.transb());
    return os;
  }
};

