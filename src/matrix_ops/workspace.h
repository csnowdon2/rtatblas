#pragma once
#include <cstdlib>
#include <iostream>
// Captures user-provided workspace. Non-owning sized pointer.

namespace rtat {

template<typename T>
class Matrix;

class Workspace {
protected:
  size_t count;
  char* ptr;
public:

  template<typename T>
  Workspace(T* ptr, size_t count) : count(count*sizeof(T)), ptr((char*)ptr) {}

  Workspace() : count(0), ptr(nullptr) {}
  virtual ~Workspace() = default;

  Workspace(Workspace other, size_t offset, size_t count) 
      : Workspace(&other.ptr[offset], count) {
    if (offset+count > other.count) {
      std::cout << "WORKSPACE OFFSET ERROR" << std::endl;
      throw;
    }
  }

  template<typename T>
  Workspace peel(size_t size) {
    size_t bytes = size*sizeof(T);

    Workspace newspace(ptr, bytes);
    ptr += bytes;
    count -= bytes;
    return newspace;
  }

  template<typename T>
  operator T*() {return (T*)ptr;}

  template<typename T>
  size_t size() {return count/sizeof(T);}
};

}
