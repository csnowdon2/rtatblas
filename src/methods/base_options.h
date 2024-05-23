#pragma once
#include <stdexcept>
#include <string>
#include <vector>

namespace rtat {

class BLAS_Op {
public:
  enum _BLAS_Op {
    NOTRANS, TRANS
  };
  _BLAS_Op op;

  BLAS_Op() : op(NOTRANS) {}

  bool operator==(_BLAS_Op o) { return op == o; }

  operator std::string() const {
    switch (op) {
      case TRANS: return "T";
      case NOTRANS: return "N";
      default: throw;
    }
  }

  BLAS_Op operator!() {
    switch (op) {
      case NOTRANS:
        return BLAS_Op(TRANS);
      case TRANS:
        return BLAS_Op(NOTRANS);
    }
  }

  BLAS_Op(_BLAS_Op op) : op(op) {}

  BLAS_Op(std::string c) {
    if (c.size() > 1) throw;
    switch (c[0]) {
      case 'N': op = NOTRANS;
      case 'T': op = TRANS;
      default: throw;
    }
  }

  std::vector<BLAS_Op> enumerate() {
    return {BLAS_Op(NOTRANS), BLAS_Op(TRANS)};
  }
};

class Pad_Op {
public:
  enum _Pad_Op {
    NOPAD, PAD
  };
  _Pad_Op op;

  Pad_Op() : op(NOPAD) {}

  bool operator==(_Pad_Op o) { return op == o; }

  operator std::string() const {
    switch (op) {
      case PAD: return "P";
      case NOPAD: return "N";
      default: throw;
    }
  }

  Pad_Op operator!() {
    switch (op) {
      case NOPAD:
        return Pad_Op(PAD);
      case PAD:
        return Pad_Op(NOPAD);
    }
  }

  Pad_Op(_Pad_Op op) : op(op) {}

  Pad_Op(std::string c) {
    if (c.size() > 1) throw;
    switch (c[0]) {
      case 'N': op = NOPAD;
      case 'P': op = PAD;
      default: throw;
    }
  }

  std::vector<Pad_Op> enumerate() {
    return {Pad_Op(NOPAD), Pad_Op(PAD)};
  }
};

class Bool_Op {
public:
  bool op;

  Bool_Op() : op(false) {}

  operator bool() {return op;}

  Bool_Op operator!() {return Bool_Op(!op);}

  operator std::string() const {
    if (op) return "T";
    return "F";
  }

  Bool_Op(bool op) : op(op) {}

  Bool_Op(std::string c) {
    if (c == "T") {
      op = true;
    } else if (c == "F") {
      op = false;
    } else {
      std::string err = "Invalid bool op string " + c;
      throw std::runtime_error(err);
    }
  }

  std::vector<Bool_Op> enumerate() {
    return {Bool_Op(false), Bool_Op(true)};
  }
};
}
