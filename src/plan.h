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
  NOTRANS, TRANS
};

enum Pad_Op {
  NOPAD, PAD
};

char op_to_char(BLAS_Op op) {
  switch (op) {
    case TRANS: return 'T';
    case NOTRANS: return 'N';
  }
}

char op_to_char(Pad_Op op) {
  switch (op) {
    case PAD: return 'P';
    case NOPAD: return 'N';
  }
}

using Trans_Opt = Option<BLAS_Op, NOTRANS, TRANS>;
using Pad_Opt = Option<Pad_Op, NOPAD, PAD>;

class GEMM_Options : public Options<Trans_Opt, Pad_Opt, Trans_Opt, Pad_Opt, Trans_Opt, Pad_Opt> {
public:
  using Options<Trans_Opt, Pad_Opt, Trans_Opt, Pad_Opt, Trans_Opt, Pad_Opt>::Options;
  Trans_Opt transa() const {return std::get<0>(*this);}
  Trans_Opt transb() const {return std::get<2>(*this);}
  Trans_Opt transc() const {return std::get<4>(*this);}
  Pad_Opt pada() const {return std::get<1>(*this);}
  Pad_Opt padb() const {return std::get<3>(*this);}
  Pad_Opt padc() const {return std::get<5>(*this);}

  friend std::ostream& operator<<(std::ostream& os, const GEMM_Options opts) {
    os << op_to_char(opts.transa());
    os << op_to_char(opts.transb());
    os << op_to_char(opts.transc());
    os << op_to_char(opts.pada());
    os << op_to_char(opts.padb());
    os << op_to_char(opts.padc());
  
    return os;
  }
};

