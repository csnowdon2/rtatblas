#pragma once
#include "problemset.h"
#include <planning_system.h>
#include <iostream>
#include <stack>
#include <rtat.h>

namespace rtat {

class GPU_Stack_Buffer {
  std::stack<size_t> stack;
  Device_RNG rng;

  size_t size;
  double *ptr;

public:
  GPU_Stack_Buffer(size_t size) : size(size) { gpuAssert(cudaMalloc(&ptr, size)); }
  ~GPU_Stack_Buffer() { gpuAssert(cudaFree(ptr)); }

  void pop() { stack.pop(); }

  template<typename T>
  T* alloc(size_t count) {
    count = (count/(512/sizeof(T)))*(512/sizeof(T));

    size_t pos = (stack.size() == 0) ? 0 : stack.top();
    if ((pos+count)*sizeof(T) > size) {
      std::cout << "OVER-ALLOCATION" << std::endl;
      std::cout << "At " << pos*sizeof(T) << "/" << size << std::endl;
      std::cout << "Requested " << count*sizeof(T) << std::endl;
      throw;
    }

    stack.push(pos+count*sizeof(T));

    rng.uniform<T>((T*)&ptr[pos], count);

    return (T*)&ptr[pos];
  }

  template<typename T>
  Matrix<T> allocate_matrix(int m, int n) {
    size_t size = (size_t)(m)*(size_t)(n);
    Workspace space(alloc<T>(size), size);

    Matrix<T> A(space, m, n, m);
    return A;
  }

  size_t remaining_space() {
    if (size <= stack.top()) return 0;
    return size - stack.top();
  }
};

inline size_t avail_gpu_mem() {
  size_t free, total;
  gpuAssert(cudaMemGetInfo(&free, &total));
  return free;
}

template<typename T>
class Runner {
protected:
  GPU_Stack_Buffer mem;
  cublasHandle_t handle;
  Stream s;
  Planning_System<GEMM_Executor<T>> planner;

public:
  Planning_System<GEMM_Executor<T>>& get_planner() { return planner; }
  void json_output(std::ostream &os) { os << std::setw(2) << planner.make_statistics().json(); }
  void json_output(std::string filename) { std::ofstream file(filename); json_output(file); }
  

  Runner() : mem((size_t)(((T)avail_gpu_mem())*0.9)) {
    cublasCreate(&handle);
    cublasSetStream(handle,s);

    //if (const char* predfilename = std::getenv("PREDFILE")) 
    //  load_predicates(std::string(predfilename));
  }

  virtual ~Runner() { gpuAssert(cudaDeviceSynchronize()); cublasDestroy(handle); }

  virtual GEMM_Options get_plan(GEMM_Inputs<T>) {
    return GEMM_Options(BLAS_Op::NOTRANS, Pad_Op::NOPAD, 
        BLAS_Op::NOTRANS, Pad_Op::NOPAD, 
        BLAS_Op::NOTRANS, Pad_Op::NOPAD);
  }

  void sync() { s.synchronize(); }

  void run_problems(Problem_Set &problems, int reps) {
  
    // TODO Use the existing workspace class rather than the Stack thing, 
    // it does the same thing but will be more elegant (if it works how 
    // I think it does)
    for (auto &problem : problems.get_problems()) {
      size_t m = problem.m;
      size_t k = problem.k;
      size_t n = problem.n;
  
      Matrix<T> A, B, C;
      A = (problem.opA == CUBLAS_OP_N) ? mem.allocate_matrix<T>(m,k) : mem.allocate_matrix<T>(k,m);
      B = (problem.opB == CUBLAS_OP_N) ? mem.allocate_matrix<T>(k,n) : mem.allocate_matrix<T>(n,k);
      C = mem.allocate_matrix<T>(m,n);
  
      GEMM_Inputs<T> inputs(handle, problem.opA, problem.opB, A, B, C, 1.0, 0.0);
  
      std::cout << "Run problem " << problem << std::endl;
  
      for (int i = 0; i < reps; i++) {
        auto plan = get_plan(inputs);

        size_t ws = planner.calculate_workspace(inputs,plan);
        if (ws > mem.remaining_space()) {
          std::cout << "Insufficient memory for input " << problem << ", skipping" << std::endl;
          continue;
        }
        std::cout << "Running " << plan << std::endl;

        Workspace space(mem.alloc<T>(ws), ws);

        planner.execute(inputs, plan, space, s);
        mem.pop();
      }
      gpuAssert(cudaDeviceSynchronize());
  
      mem.pop();
      mem.pop();
      mem.pop();
    }
  }
};

template <typename T>
class SmartRunner : public Runner<T> {
public:
  GEMM_Options get_plan(GEMM_Inputs<T> inputs) override {
    return this->planner.create_plan(inputs);
  }
};

template <typename T>
class RoundRobinRunner : public Runner<T> {
  int i=-1;
public:
  GEMM_Options get_plan(GEMM_Inputs<T>) override {
    const auto ops = GEMM_Options::enumerate();
    i = (i+1)%ops.size();
    return ops[i];
  }
};

}
