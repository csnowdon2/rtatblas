#pragma once

#include <sstream>
#include <ios>
#ifdef CUDA
#include <cublas_v2.h>
#else
#include <hipblas.h>
#endif
#include "options.h"

namespace rtat {

enum BLAS_Op {
  NOTRANS, TRANS
};

enum Pad_Op {
  NOPAD, PAD
};

inline char op_to_char(BLAS_Op op) {
  switch (op) {
    case TRANS: return 'T';
    case NOTRANS: return 'N';
    default: throw;
  }
}

inline char op_to_char(Pad_Op op) {
  switch (op) {
    case PAD: return 'P';
    case NOPAD: return 'N';
    default: throw;
  }
}

inline BLAS_Op char_to_blas_op(char c) {
  switch (c) {
    case 'N': return NOTRANS;
    case 'T': return TRANS;
    default: throw;
  }
}

inline Pad_Op char_to_pad_op(char c) {
  switch (c) {
    case 'N': return NOPAD;
    case 'P': return PAD;
    default: throw;
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

  operator std::string() const {
    std::stringstream ss;
    ss << op_to_char(transa());
    ss << op_to_char(transb());
    ss << op_to_char(transc());
    ss << op_to_char(pada());
    ss << op_to_char(padb());
    ss << op_to_char(padc());

    std::string ret;
    ss >> ret;
    return ret;
  }

  friend std::ostream& operator<<(std::ostream& os, const GEMM_Options opts) {
    os << std::string(opts); 
    return os;
  }

  friend std::istream& operator>>(std::istream &is, GEMM_Options &opts) {
    std::string s;
    is >> s;
    if (s.size() != 6) {
      is.setstate(std::ios::failbit);
      return is;
    }
      
    std::get<0>(opts) = char_to_blas_op(s[0]);
    std::get<1>(opts) = char_to_pad_op(s[3]);
    std::get<2>(opts) = char_to_blas_op(s[1]);
    std::get<3>(opts) = char_to_pad_op(s[4]);
    std::get<4>(opts) = char_to_blas_op(s[2]);
    std::get<5>(opts) = char_to_pad_op(s[5]);

    return is;
  }
};

}
