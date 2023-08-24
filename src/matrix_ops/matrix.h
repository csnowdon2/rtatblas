#include "workspace.h"
// There should be no inheritance relationship between Matrix and MatrixOp.
// Accessing a MatrixOp requires a handle, accessing a Matrix does not.


struct MatrixDims {
  size_t m, n, ld;
  MatrixDims(size_t m, size_t n, size_t ld) : m(m), n(n), ld(ld) {
    if (ld < m) throw;
  }

  const size_t footprint() const {return ld*n;}
};

class Matrix {
  Workspace home;
  MatrixDims dimensions;
public:

  Matrix(Workspace home, MatrixDims dimensions)
      : home(home), dimensions(dimensions) {}

  Matrix(Workspace home, size_t m, size_t n, size_t ld) 
      : Matrix(home, MatrixDims(m,n,ld)) {}

  const size_t footprint() const {return dimensions.footprint();}

  const MatrixDims dims() const {return dimensions;}

  double* const ptr() {return home.ptr;}
};
