#pragma once
#include <cstdlib>
#include <iostream>
#include <gpu-api.h>
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

  Workspace(Workspace &other, size_t offset, size_t count) 
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

class ManagedWorkspace : public Workspace {
public:
  ManagedWorkspace(size_t bytes) : Workspace() {
    gpuAssert(gpu::Malloc(&ptr, bytes));
    count = bytes;
  }
  ~ManagedWorkspace() { gpuAssert(gpu::Free(ptr)); }

  template<typename T>
  void grow_to_fit(size_t new_count) {
    if (size<T>() < new_count) {
      gpuAssert(gpu::DeviceSynchronize());
      gpuAssert(gpu::Free(ptr));
      gpuAssert(gpu::Malloc(&ptr, new_count*sizeof(T)));
      count = new_count*sizeof(T);
    }
  }
};

}
