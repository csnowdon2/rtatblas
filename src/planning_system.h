#pragma once
#include <map>
#include <gpu-api.h>
#include <performance_record.h>
#include "options.h"
#include "plan.h"

struct Matrix {
  double *ptr;
  size_t m, n, ld;

  Matrix (double *ptr, size_t m, size_t n, size_t ld) : ptr(ptr), m(m), n(n), ld(ld) {}
  Matrix() {}

  size_t footprint() {return n*ld;}
};

struct Workspace {
  Workspace(double* ptr, size_t size) : ptr(ptr), size(size) {}
  Workspace() : ptr(nullptr), size(0) {}
  Workspace(Workspace w, size_t offset, size_t size) :
      Workspace(&w[offset], size) {
    if (offset + size > w.size) throw;
  }

  double * const ptr;
  const size_t size;

  double &operator[](size_t ix) {return ptr[ix];}
};

template<typename Input_Params, typename Input_Key, typename Opts>
class Planning_System {
public:
  // Can probably come up with a sensible default that could 
  // be overridden. e.g. just consider walltimes naively 
  // vs using flop rates to figure out absolute performance.
  // That can come later though, we'll do FLOP rates in 
  // implementation class for now.
  virtual Opts create_plan(Input_Params params) = 0;

  // Don't do any planning, just take a plan and go with it, 
  // measuring performance on the way.
  void execute(Opts opts, Input_Params params, Stream s) {

    Analytics &an = analytics[Input_Key(params)];

    an.performance_data[opts].measure([&]() {
      internal_execute(opts, params, s);
    }, s);
  }

  virtual ~Planning_System() = default;

protected:
  // Actually do the computation
  virtual void internal_execute(Opts opts, Input_Params params, Stream s) = 0;

private:
  // Add workspace by inheritance?

  class Analytics {
    Analytics() {
      for (auto &opts : Opts::enumerate()) {
        performance_data.emplace(std::make_pair(opts, Performance_Record()));
      }
    }

    std::map<Opts, Performance_Record> performance_data;
  };

  std::map<Input_Key, Analytics> analytics;
};


struct GEMM_Inputs {
  cublasHandle_t handle;
  cublasOperation_t transa; cublasOperation_t transb;
  const Matrix A;
  const Matrix B;
        Matrix C;
  const double alpha; const double beta;
  Workspace space;
  
  size_t m() {return C.m;}
  size_t n() {return C.n;}
  size_t k() {return (transa == CUBLAS_OP_T) ? A.m : A.n;}
};

struct GEMM_Key {
  cublasOperation_t transa; cublasOperation_t transb;
  int m; int k; int n;
  int lda; int ldb; int ldc;

  GEMM_Key(GEMM_Inputs i) : transa(i.transa), transb(i.transb), 
                            m(i.m()), n(i.n()), k(i.k()), 
                            lda(i.A.ld), ldb(i.B.ld), ldc(i.C.ld) {}
};



class GEMM_Planner : public Planning_System<GEMM_Inputs, GEMM_Key, GEMM_Options> {
public:
  //GEMM_Options create_plans(GEMM_Inputs params) {
  //}

  //size_t calculate_workspace(GEMM_Inputs params, GEMM_Options opts) {
  //  
  //}

private:

  void internal_execute(GEMM_Options opts, GEMM_Inputs params, Stream s) override {
    
    size_t workspace_offset = 0;
    const Matrix &A = params.A;
    const Matrix &B = params.B;
    const Matrix &C = params.C;

    Matrix left = A;
    Matrix right = B;
    if (opts.transa() == TRANS) {
      left.ptr = &params.space[workspace_offset];
      left.m = A.n;
      left.n = A.m;
      left.ld = ((left.m+31)/32)*32;
      workspace_offset += ((left.footprint()+511)/512)*512;
    }
    if (opts.transb() == TRANS) {
      right.ptr = &params.space[workspace_offset];
      right.m = B.n;
      right.n = B.m;
      right.ld = ((right.m+31)/32)*32;
      workspace_offset += ((right.footprint()+511)/512)*512;
    }
      // Computation object. Stores all input info required, can be 
      // modified e.g. to change transposes and padding. Can be 
      // queried for required output buffer size. Takes output buffer 
      // and returns matrix object

    // Options tell us what to do to the input
    // Need workspace to do transposes according to plan. Add to class 
    // with methods to manage workspace?
    // Need to iron out what plan means: perform transposes, or switch to 
    // given transposes?
  }
};
