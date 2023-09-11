#include <cstdlib>
// Captures user-provided workspace. Non-owning sized pointer.

class Matrix;

class Workspace {
  friend class Matrix;
  size_t count;
  double* ptr;
public:

  Workspace(double* ptr, size_t count) : ptr(ptr), count(count) {}
  Workspace() : Workspace(nullptr, 0) {}

  Workspace(Workspace other, size_t offset, size_t count) 
      : Workspace(&other.ptr[offset], count) {
    if (offset+count > other.count) {
      std::cout << "WORKSPACE OFFSET ERROR" << std::endl;
      throw;
    }
  }

  Workspace peel(size_t size) {
    Workspace newspace(ptr, size);
    ptr += size;
    count -= size;
    return newspace;
  }

  double& operator[](size_t ix) {return ptr[ix];}

  size_t size() {return count;}
};
