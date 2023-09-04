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

using Trans_Opt = Option<bool, TRANS, NOTRANS>;

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

