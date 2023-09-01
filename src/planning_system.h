#pragma once
#include "matrix_ops/matrixop.h"
#include <map>
#include <gpu-api.h>
#include <performance_record.h>
#include <iostream>
#include "options.h"
#include "plan.h"

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

    auto key = Input_Key(params);
    Analytics &an = analytics[key];

    an.performance_data[opts].measure([&](Stream &str) {
      internal_execute(opts, params, str);
    }, s);
  }

  virtual ~Planning_System() = default;

  void dump_analytics() {
    for (auto &[key, an] : analytics) {
      std::cout << "KEY=" << key << std::endl;
      for (auto &[opts, rec] : an.performance_data) {
        std::cout << opts << " AVG=" << rec.get_time() << " STD=" << rec.get_std() << std::endl;
        rec.print();
      }
    }
  }

protected:
  // Actually do the computation
  virtual void internal_execute(Opts opts, Input_Params params, Stream s) = 0;

  class Analytics {
    public:
    Analytics() {
      for (auto &opts : Opts::enumerate()) {
        performance_data.emplace(std::make_pair(opts, Performance_Record(true)));
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

  GEMM_Inputs(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
              const Matrix A, const Matrix B, Matrix C, double alpha, double beta, 
              Workspace space) 
        : handle(handle), transa(transa), transb(transb), A(A), B(B), C(C), 
          alpha(alpha), beta(beta), space(space) {}

  size_t m() {return C.dims().m;}
  size_t n() {return C.dims().n;}
  size_t k() {return (transa == CUBLAS_OP_N) ? A.dims().n : A.dims().m;}
};

struct GEMM_Key {
  cublasOperation_t transa; cublasOperation_t transb;
  int m; int k; int n;
  //int lda; int ldb; int ldc;

  GEMM_Key(GEMM_Inputs i) : transa(i.transa), transb(i.transb), 
                            m(i.m()), n(i.n()), k(i.k()) {}
  //                          lda(i.A.dims().ld), ldb(i.B.dims().ld), ldc(i.C.dims().ld) {}

  friend bool operator<(const GEMM_Key &l, const GEMM_Key &r) {
    if (l.transa < r.transa) return true;
    if (l.transa > r.transa) return false;
    if (l.transb < r.transb) return true;
    if (l.transb > r.transb) return false;
    if (l.m < r.m) return true;
    if (l.m > r.m) return false;
    if (l.k < r.k) return true;
    if (l.k > r.k) return false;
    if (l.n < r.n) return true;
    if (l.n > r.n) return false;
    //if (l.lda < r.lda) return true;
    //if (l.lda > r.lda) return false;
    //if (l.ldb < r.ldb) return true;
    //if (l.ldb > r.ldb) return false;
    //if (l.ldc < r.ldc) return true;
    //if (l.ldc > r.ldc) return false;
    return false;
  }

  friend std::ostream& operator<<(std::ostream& os, const GEMM_Key& dt) {
      os << (dt.transa == CUBLAS_OP_N ? "N" : "T")
         << (dt.transb == CUBLAS_OP_N ? "N" : "T")
         << " m=" << dt.m
         << " n=" << dt.n
         << " k=" << dt.k;
      return os;
  }
};



class GEMM_Planner : public Planning_System<GEMM_Inputs, GEMM_Key, GEMM_Options> {
public:
  GEMM_Options create_plan(GEMM_Inputs params) override {
    // TODO actually come up with a method to determine a plan
    auto &an = analytics[GEMM_Key(params)];

    GEMM_Options plan(NOTRANS, NOTRANS);
    int min_count = 10000;
    for (auto &[opts, rec] : an.performance_data)
      if (rec.count() < min_count) min_count = rec.count();

    if (min_count < 3) 
      for (auto &[opts, rec] : an.performance_data)
        if (rec.count() == min_count) return opts;

    float ms = std::numeric_limits<float>::infinity();
    for (auto &[opts, rec] : an.performance_data) {
      rec.synchronous = false;
      float t = rec.get_time();
      if (t < ms) {
        plan = opts;
        ms = t;
      }
    }

    return plan;
  }

  size_t calculate_workspace(GEMM_Options opts, GEMM_Inputs params) {
    auto mult = form_operation(opts, params);
    return mult.workspace_req();
  }

  bool acceptable_plan(GEMM_Options opts, GEMM_Inputs params) {
    auto mult = form_operation(opts, params);
    return mult.workspace_req() <= params.space.size();
  }

  MatrixMult form_operation(GEMM_Options opts, GEMM_Inputs params) {

    std::unique_ptr<MatrixOp> A = std::make_unique<NoOp>(params.A);
    std::unique_ptr<MatrixOp> B = std::make_unique<NoOp>(params.B);
    std::unique_ptr<MatrixOp> C = std::make_unique<NoOp>(params.C);

    Workspace space = params.space;

    if (opts.transa() == TRANS) {
      A = transpose_matrix(std::move(A), 1.0, 32);
      params.transa = (params.transa == CUBLAS_OP_N) ? CUBLAS_OP_T : CUBLAS_OP_N;
    }
    if (opts.transb() == TRANS) {
      B = transpose_matrix(std::move(B), 1.0, 32);
      params.transb = (params.transb == CUBLAS_OP_N) ? CUBLAS_OP_T : CUBLAS_OP_N;
    }


    MatrixMult mult(std::move(A), std::move(B), std::move(C), 
                    params.transa == CUBLAS_OP_T, params.transb == CUBLAS_OP_T,
                    params.alpha, params.beta);
    return mult;
  }

private:

  void internal_execute(GEMM_Options opts, GEMM_Inputs params, Stream s) override {
    auto mult = form_operation(opts, params);
    if (mult.workspace_req() > params.space.size()) throw "Insufficient workspace";
    mult.execute(params.handle, Workspace(), params.space);
    // What to do if workspace is insufficient?
  }
};
