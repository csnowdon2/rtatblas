#pragma once
#include "workspace.h"
// There should be no inheritance relationship between Matrix and MatrixOp.
// Accessing a MatrixOp requires a handle, accessing a Matrix does not.

namespace rtat {

struct MatrixDims {
  size_t m, n, ld;
  MatrixDims() {}
  MatrixDims(size_t m, size_t n, size_t ld) : m(m), n(n), ld(ld) {
    if (ld < m) throw "ld < m in MatrixDims";
  }

  size_t footprint() const {return ld*n;}

  friend std::ostream& operator<<(std::ostream& os, const MatrixDims dims) {
    os << "(" << dims.m << "," << dims.n << "," << dims.ld << ")";
    return os;
  }
};

template<typename T>
class Matrix {
  Workspace home;
  MatrixDims dimensions;
public:

  Matrix() {}

  Matrix(Workspace home, MatrixDims dimensions)
      : home(home), dimensions(dimensions) {
    if (home.size<T>() < home.size<T>())
      throw "MATRIX WORKSPACE TOO SMALL";
  }

  Matrix(Workspace home, size_t m, size_t n, size_t ld) 
      : Matrix(home, MatrixDims(m,n,ld)) {}

  Matrix block(int row, int col, int nrows, int ncols) {
    MatrixDims block_dims(nrows, ncols, dims().ld);

    size_t offset = row+col*dims().ld;
    Workspace block_home(home, offset, block_dims.footprint()*sizeof(T));

    return Matrix(block_home, block_dims);

  }

  size_t footprint() const {return dimensions.footprint();}

  const MatrixDims dims() const {return dimensions;}

  T* ptr() {return (T*)home;}
};

}
