#pragma once
#include "problemset.h"
#include "json_planner.h"
#include <planning_system.h>
#include <iostream>
#include <stack>

namespace rtat {

class GPU_Stack_Buffer {
  std::stack<size_t> stack;

  size_t size;
  double *ptr;

public:
  GPU_Stack_Buffer(size_t size) : size(size) { gpuAssert(cudaMalloc(&ptr, size)); }
  ~GPU_Stack_Buffer() { gpuAssert(cudaFree(ptr)); }

  void pop() { stack.pop(); }

  double* alloc(size_t doubles) {
    doubles = (doubles/32)*32;

    size_t pos = (stack.size() == 0) ? 0 : stack.top();
    if ((pos+doubles)*sizeof(double) > size) {
      std::cout << "OVER-ALLOCATION" << std::endl;
      std::cout << "At " << pos*sizeof(double) << "/" << size << std::endl;
      std::cout << "Requested " << doubles*sizeof(double) << std::endl;
      throw;
    }

    stack.push(pos+doubles);

    return &ptr[pos];
  }

  Matrix allocate_matrix(int m, int n) {
    size_t size = (size_t)(m)*(size_t)(n);
    Workspace space(alloc(size), size);

    Matrix A(space, m, n, m);
    return A;
  }

  size_t remaining_space() {
    if (size <= stack.top()) return 0;
    return size - stack.top();
  }
};

size_t avail_gpu_mem() {
  size_t free, total;
  gpuAssert(cudaMemGetInfo(&free, &total));
  return free;
}

class Runner {
protected:
  GPU_Stack_Buffer mem;
  cublasHandle_t handle;
  Stream s;
  JSON_Planner<GEMM_Planner> planner;

  //std::vector<Predicate<std::pair<GEMM_Options, GEMM_Inputs>>> make_preds() {
  //  std::vector<Predicate<std::pair<GEMM_Options, GEMM_Inputs>>> ret;
  //  ret.emplace_back(exclude_option(CUBLAS_OP_T,CUBLAS_OP_T));
  //  ret.emplace_back(exclude_option(CUBLAS_OP_T,CUBLAS_OP_N));
  //  return ret;
  //}

public:
  void json_output(std::ostream &os) { planner.json_output(os); }
  void json_output(std::string filename) { planner.json_output(filename); }
  

  Runner() : mem((size_t)(((double)avail_gpu_mem())*0.9)), 
             planner({}, 1){
    cublasCreate(&handle);
    cublasSetStream(handle,s);

    if (const char* predfilename = std::getenv("PREDFILE")) 
      load_predicates(std::string(predfilename));
  }

  virtual ~Runner() { gpuAssert(cudaDeviceSynchronize()); cublasDestroy(handle); }

  virtual GEMM_Options get_plan([[maybe_unused]] GEMM_Inputs inputs) {
    return GEMM_Options(NOTRANS, NOPAD, NOTRANS, NOPAD, NOTRANS, NOPAD);
  }

  void sync() { s.synchronize(); }

  void print_analytics() { planner.dump_analytics(); } 

  void print_top_n(int n) {
    planner.dump_top_n(n);
  }

  void print_bottom_n(int n) {
    planner.dump_bottom_n(n);
  }

  void load_predicates(std::string filename) {
    if (filename.size() == 0) return;
    std::cout << "LOADING PREDFILE " << filename << std::endl;
    std::ifstream is(filename);
    if (!is.good()) {
      std::cout << "File failed to open" << std::endl;
      throw;
    }
    planner.load_predicates(is);
  }

  virtual void run_problems(Problem_Set &problems, int reps) {
  
    // TODO Use the existing workspace class rather than the Stack thing, 
    // it does the same thing but will be more elegant (if it works how 
    // I think it does)
    for (auto &problem : problems.get_problems()) {
      size_t m = problem.m;
      size_t k = problem.k;
      size_t n = problem.n;
      GEMM_Options plan(NOTRANS, NOPAD, NOTRANS, NOPAD, NOTRANS, NOPAD);
  
      Matrix A, B, C;
      A = (problem.opA == CUBLAS_OP_N) ? mem.allocate_matrix(m,k) : mem.allocate_matrix(k,m);
      B = (problem.opB == CUBLAS_OP_N) ? mem.allocate_matrix(k,n) : mem.allocate_matrix(n,k);
      C = mem.allocate_matrix(m,n);
  
      GEMM_Inputs inputs(handle, problem.opA, problem.opB, A, B, C,
                         1.0, 0.0, Workspace());
  
      std::cout << "Run problem " << problem << std::endl;
  
      for (int i = 0; i < reps; i++) {
        auto plan = get_plan(inputs);

        if (planner.calculate_workspace(plan,inputs)*sizeof(double) > mem.remaining_space()) {
          std::cout << "Insufficient memory for input " << problem << ", skipping" << std::endl;
          continue;
        }
        std::cout << "Running " << plan << std::endl;

        size_t ws = planner.calculate_workspace(plan,inputs);
        inputs.space = Workspace(mem.alloc(ws), ws);

        if (i == 0) {planner.warmup(plan,inputs,s); gpuAssert(cudaDeviceSynchronize());}
        //planner.warmup(plan,inputs,s);
        planner.execute(plan, inputs, s);
        mem.pop();
      }
      gpuAssert(cudaDeviceSynchronize());
  
      mem.pop();
      mem.pop();
      mem.pop();

      problem.flop_rate = planner.get_floprate(inputs);
      problems.dump();
    }
  }
};

class SmartRunner : public Runner {
public:
  GEMM_Options get_plan(GEMM_Inputs inputs) override {
    return planner.create_plan(inputs);
  }
};

class RoundRobinRunner : public Runner {
  int i=-1;
public:
  GEMM_Options get_plan([[maybe_unused]] GEMM_Inputs inputs) override {
    const auto ops = GEMM_Options::enumerate();
    i = (i+1)%ops.size();
    return ops[i];
  }
};

class ExhaustiveRunner : public Runner {
  GEMM_Options current_plan;
public:
  GEMM_Options get_plan([[maybe_unused]] GEMM_Inputs inputs) override {
    return current_plan;
  }

  void run_problems(Problem_Set &problems, int reps) override {
    for (auto &plan : GEMM_Options::enumerate()) {
      current_plan = plan;
      Runner::run_problems(problems, reps);
    }
  }
};
}
