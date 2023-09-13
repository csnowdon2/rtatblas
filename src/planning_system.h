#pragma once
#include <functional>
#include "matrix_ops/matrixop.h"
#include <map>
#include <gpu-api.h>
#include <performance_record.h>
#include <iostream>
#include "options.h"
#include "plan.h"

template<typename T>
using Predicate = std::function<bool(T)>;

template<typename Input_Params, typename Input_Key, typename Opts>
class Planning_System {
public:
  Planning_System() = default;
  Planning_System(std::vector<Predicate<std::pair<Opts, Input_Params>>> predicates) 
      : predicates(predicates) {}
  // Can probably come up with a sensible default that could 
  // be overridden. e.g. just consider walltimes naively 
  // vs using flop rates to figure out absolute performance.
  // That can come later though, we'll do FLOP rates in 
  // implementation class for now.
  virtual Opts create_plan(Input_Params params) = 0;

  // Don't do any planning, just take a plan and go with it, 
  // measuring performance on the way.
  void execute(Opts opts, Input_Params params, Stream s) {

    if (!warm) {
      warmup(opts, params, s);
      warm = true;
    }

    Analytics &an = get_analytics(params);

    an.performance_data[opts].measure([&](Stream &str) {
      internal_execute(opts, params, str);
    }, s);
  }

  void warmup(Opts opts, Input_Params params, Stream s) {

    Analytics &an = get_analytics(params);

    internal_execute(opts, params, s);
  }

  virtual ~Planning_System() = default;

  void dump_analytics() {
    for (auto &[key, an] : analytics) {
      std::cout << "KEY=" << key << std::endl;
      for (auto &[opts, rec] : an.performance_data) {
        std::cout << opts << " AVG=" << rec.get_time() << " STD=" << rec.get_std() << std::endl;
        rec.print();
      }
      std::cout << std::endl;
    }
  }

protected:
  // Actually do the computation
  virtual void internal_execute(Opts opts, Input_Params params, Stream s) = 0;

  class Analytics {
    public:
    Analytics(std::vector<Predicate<Opts>> predicates) {
      for (auto &opts : Opts::enumerate()) {
        bool good = true;
        for (auto &pred : predicates) {
          if (!pred(opts)) {
            good = false;
            break;
          }
        }
        if (good)
          performance_data.emplace(std::make_pair(opts, Performance_Record(true)));
      }
    }
    Analytics() : Analytics({}) {}

    std::map<Opts, Performance_Record> performance_data;
  };

  Analytics &get_analytics(Input_Params params) {
    if (!analytics.count(Input_Key(params))) {
      std::vector<Predicate<Opts>> preds;
      for (auto &pred : predicates)
        preds.push_back([&pred,&params](Opts opts) -> bool 
            {return pred(std::make_pair(opts,params));});
      analytics.emplace(std::make_pair(Input_Key(params), preds));
    }
    return analytics.find(Input_Key(params))->second;
  }

  std::vector<Predicate<std::pair<Opts, Input_Params>>> predicates;
  std::map<Input_Key, Analytics> analytics;
  bool warm = false;
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


Predicate<std::pair<GEMM_Options, GEMM_Inputs>>
    exclude_option(BLAS_Op opA, BLAS_Op opB) {
  return [opA, opB](std::pair<GEMM_Options, GEMM_Inputs> p) -> bool {
    auto &opts = p.first;
    auto &params = p.second;
    auto opA_ = (params.transa == CUBLAS_OP_N) ? NOTRANS : TRANS;
    auto opB_ = (params.transb == CUBLAS_OP_N) ? NOTRANS : TRANS;

    if (opts.transa() == TRANS) opA_ = switch_op(opA_);
    if (opts.transb() == TRANS) opB_ = switch_op(opB_);

    return !(opA == opA_ && opB == opB_);
  };
}

class GEMM_Planner : public Planning_System<GEMM_Inputs, GEMM_Key, GEMM_Options> {
  int tests_until_converge = 1;
public:
  GEMM_Planner(std::vector<Predicate<std::pair<GEMM_Options, GEMM_Inputs>>> predicates, 
               int tests_until_converge) : Planning_System(predicates), 
                                           tests_until_converge(tests_until_converge) {
  }
  GEMM_Planner(std::vector<Predicate<std::pair<GEMM_Options, GEMM_Inputs>>> predicates) 
      : GEMM_Planner(predicates, 1) {}
  GEMM_Planner() : GEMM_Planner({}, 1) {}

  GEMM_Options create_plan(GEMM_Inputs params) override {
    // TODO actually come up with a method to determine a plan
    auto &an = get_analytics(params);

    //GEMM_Options plan(NOTRANS, NOTRANS);
    GEMM_Options plan = an.performance_data.begin()->first;
    int min_count = 10000;
    for (auto &[opts, rec] : an.performance_data)
      if (rec.count() < min_count) min_count = rec.count();

    if (min_count < tests_until_converge) 
      for (auto &[opts, rec] : an.performance_data)
        if (rec.count() == min_count) return opts;

    float ms = std::numeric_limits<float>::infinity();
    for (auto &[opts, rec] : an.performance_data) {
      float t = rec.get_time();
      rec.synchronous = false;
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
    return calculate_workspace(opts, params) <= params.space.size();
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

  double get_floprate(GEMM_Options opts, GEMM_Inputs params) {
    Analytics &an = get_analytics(params);

    double secs = (double)(an.performance_data[opts].get_time())/1000.0;
    double tflops = 2*params.m()*params.k()*params.n()/1e12;

    return tflops/secs;
  }

  double get_floprate(GEMM_Inputs params) {
    Analytics &an = get_analytics(params);

    size_t count = 0;
    float total = 0.0;
    for (auto &[opts,rec] : an.performance_data) {
      if (rec.count() > 0) {
        rec.flush();
        count += rec.count();
        total += rec.count()*rec.get_time();
      }
    }
    std::cout << "Count = " << count << ", Total = " << total << std::endl;
    double secs = (double)(total/count)/1000.0;
    double tflops = 2*params.m()*params.k()*params.n()/1e12;
    std::cout << "Rate = " << tflops/secs << std::endl;

    return tflops/secs;
  }

private:

  void internal_execute(GEMM_Options opts, GEMM_Inputs params, Stream s) override {
    auto mult = form_operation(opts, params);
    if (mult.workspace_req() > params.space.size()) {
      std::cout << "INSUFFICIENT WORKSPACE" << std::endl;
      throw "Insufficient workspace";
    }
    mult.execute(params.handle, Workspace(), params.space);
    // What to do if workspace is insufficient?
  }
};
