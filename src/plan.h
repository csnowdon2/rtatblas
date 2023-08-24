#pragma once
#include <variant>
#include <vector>
#include <map>
#include <cublas_v2.h>
#include "options.h"


enum BLAS_Op {
  TRANS, NOTRANS
};

using Trans_Opt = Option<bool, TRANS, NOTRANS>;

class GEMM_Options : public Options<Trans_Opt, Trans_Opt> {
public:
  Trans_Opt transa() {return std::get<0>(*this);}
  Trans_Opt transb() {return std::get<1>(*this);}
};

