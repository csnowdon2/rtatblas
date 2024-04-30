#pragma once

#include <map>
#include <iostream>

#include <gpu-api.h>
#include <timer_bank.h>
#include <numeric>

#include "matrix_ops/matrixop.h"
#include "plan.h"
#include "predicates.h"

namespace rtat {


template<typename Key, typename Opts>
class Option_Filter {
  Predicate<std::pair<Opts,Key>> filter;
public:

  Option_Filter(Predicate<std::pair<Opts,Key>> filter) 
    : filter(filter) {}

  Option_Filter() 
    : filter([](std::pair<Opts, Key>) { return true; }) {}

  std::vector<Opts> apply(Key key) const {
    std::vector<Opts> ret;

    for (auto &opts : Opts::enumerate()) {
      if (filter(std::make_pair(opts, key)))
        ret.push_back(opts);
    }

    return ret;
  }
};

template<typename Params, typename Key, typename Opts>
class Executor {
public:
  typedef Params Params_T;
  typedef Key    Key_T;
  typedef Opts   Opts_T;

  Executor() = default;
  virtual ~Executor() = default;

  virtual void execute(Params params, Opts opts, 
                       Workspace space, Stream s, 
                       Device_Timer::Mode sync = Device_Timer::ASYNCHRONOUS) {

    if (!warm) {
      warmup(params, opts, s);
      warm = true;
    }

    Device_Timer timer([&](const Stream &str) {
      internal_execute(params, opts, space, str);
    }, s, sync);

    timer_log[params][opts].append(timer);
  }


  std::map<Key, std::map<Opts, Timer_Bank>>& get_timings() {return timer_log;}
  std::map<Opts, Timer_Bank>& get_timings(Key key) {return timer_log[key];}

  virtual size_t calculate_workspace(Params, Opts) = 0;
protected:
  virtual void internal_execute(Params, Opts, Workspace, Stream) = 0;
  virtual void warmup(Params, Opts, Stream) = 0;
  std::map<Key, std::map<Opts, Timer_Bank>> timer_log;  

  bool warm = false;
};


template<typename Executor_Type> 
class Planning_System {
protected:
  using Params = typename Executor_Type::Params_T;
  using Key    = typename Executor_Type::Key_T;
  using Opts   = typename Executor_Type::Opts_T;

  Executor_Type executor;

  const Option_Filter<Key, Opts> opt_filter;

  Opts degrade_plan(Params, Opts, Workspace) {
    return Opts::default_opts();
  }

  size_t tests_until_converge = 1;
  std::map<Key, Opts> converged_plans;
public:
  Planning_System() = default;
  Planning_System(Option_Filter<Key, Opts> opt_filter) 
      : opt_filter(opt_filter) {}

  virtual ~Planning_System() = default;

  virtual Opts create_plan(Key key) {
    if (converged_plans.count(key))
      return converged_plans[key];

    std::map<Opts, Timer_Bank> &timings = executor.get_timings(key);

    // Find un-used times
    auto opt_set = opt_filter.apply(key);
    for (auto &opts : opt_set) {
      if (timings[opts].size() < tests_until_converge)
        return opts;
    }

    // Choose best time 
    Opts best_opts;
    float best_time = std::numeric_limits<float>::max();
    for (auto &opts : opt_set) {
      Timer_Bank &time_bank = timings[opts];
      time_bank.synchronize();

      const std::vector<float>& ts = time_bank.get_times();
      float mean = 
        std::accumulate(ts.cbegin(), ts.cend(), 0.0)/ts.size();

      if (mean < best_time) {
        best_opts = opts;
        best_time = mean;
      }
    }
    converged_plans[key] = best_opts;
    return converged_plans[key];
  }

  void execute(Params params, Opts opts, Workspace space, Stream s) {
    if (space.size() < executor.calculate_workspace(params, opts)) {
      opts = degrade_plan(params, opts, space);
    }

    auto sync = Device_Timer::ASYNCHRONOUS;
    if (executor.get_timings(params)[opts].size() < tests_until_converge)
      sync = Device_Timer::SEMI_SYNCHRONOUS;

    executor.execute(params, opts, space, s, sync);
  }

  size_t calculate_workspace(Params params, Opts opts) {
    return executor.calculate_workspace(params, opts);
  }
};


struct GEMM_Inputs {
  cublasHandle_t handle;
  cublasOperation_t transa; cublasOperation_t transb;
  const Matrix A;
  const Matrix B;
        Matrix C;
  const double alpha; const double beta;

  GEMM_Inputs(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
              const Matrix A, const Matrix B, Matrix C, double alpha, double beta)
        : handle(handle), transa(transa), transb(transb), A(A), B(B), C(C), 
          alpha(alpha), beta(beta) {}

  size_t m() {return C.dims().m;}
  size_t n() {return C.dims().n;}
  size_t k() {return (transa == CUBLAS_OP_N) ? A.dims().n : A.dims().m;}
};

struct GEMM_Key {
  cublasOperation_t transa; cublasOperation_t transb;
  int m; int n; int k;

  GEMM_Key(GEMM_Inputs i) : transa(i.transa), transb(i.transb), 
                            m(i.m()), n(i.n()), k(i.k()) {}
  GEMM_Key(cublasOperation_t transa, cublasOperation_t transb,
           int m, int k, int n) : transa(transa), transb(transb), 
                                  m(m), n(n), k(k) {}

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
      os << dt.op_str()
         << " m=" << dt.m
         << " n=" << dt.n
         << " k=" << dt.k;
      return os;
  }

  std::string op_str() const {
     return std::string(transa == CUBLAS_OP_N ? "N" : "T")
         +  std::string(transb == CUBLAS_OP_N ? "N" : "T");
  }
};

inline cublasOperation_t switch_op(cublasOperation_t op) {
  switch (op) {
    case CUBLAS_OP_N: return CUBLAS_OP_T;
    case CUBLAS_OP_T: return CUBLAS_OP_N;
    default: 
      std::cout << "Invalid blas op" << std::endl;
      throw;
  }
}


inline Predicate<std::pair<GEMM_Options, GEMM_Key>>
    exclude_option(cublasOperation_t opA, cublasOperation_t opB) {
  return [opA, opB](std::pair<GEMM_Options, GEMM_Key> p) -> bool {
    auto &opts = p.first;
    auto &params = p.second;
    auto opA_ = params.transa;
    auto opB_ = params.transb;

    if (opts.transa() == TRANS) opA_ = switch_op(opA_);
    if (opts.transb() == TRANS) opB_ = switch_op(opB_);

    return !(opA == opA_ && opB == opB_);
  };
}

// Predicate which succeeds only for problems which match the given
// options and transposes
inline Predicate<std::pair<GEMM_Options, GEMM_Key>>
    permit_option(GEMM_Options opts, cublasOperation_t opa, cublasOperation_t opb) {
  return [opts, opa, opb](std::pair<GEMM_Options, GEMM_Key> p) -> bool {
    return (p.first == opts) && 
           (opa == p.second.transa) && 
           (opb == p.second.transb);
  };
}

class GEMM_Executor : public Executor<GEMM_Inputs, GEMM_Key, GEMM_Options> {
public:
  size_t calculate_workspace(GEMM_Inputs params, GEMM_Options opts) override {
    auto mult = form_operation(params, opts);
    return mult->workspace_req();
  }

protected:

  void warmup(GEMM_Inputs params, [[maybe_unused]] GEMM_Options opts,
              [[maybe_unused]] Stream s) override {
    size_t n = 8;
    double *A, *B, *C;
    gpuAssert(cudaMalloc(&A, n*n*sizeof(double)));
    gpuAssert(cudaMalloc(&B, n*n*sizeof(double)));
    gpuAssert(cudaMalloc(&C, n*n*sizeof(double)));

    std::vector<cublasOperation_t> ops = {CUBLAS_OP_N, CUBLAS_OP_T};

    for (auto &opA : ops) {
      for (auto &opB : ops) {
        double alpha = 1.0;
        double beta = 0.0;
        cublasDgemm(params.handle, opA, opB, n,n,n, &alpha, A,n,B,n, &beta, C,n);
        cublasDgeam(params.handle, opA, opB, n,n, &alpha, A, n, &beta, B, n, C, n);
      }
    }
    gpuAssert(cudaDeviceSynchronize());
  }

  void internal_execute(GEMM_Inputs params, GEMM_Options opts, Workspace space,
                        [[maybe_unused]] Stream s) override {
    auto mult = form_operation(params, opts);
    if (mult->workspace_req() > space.size()) {
      throw "GEMM internal_execute: Insufficient workspace";
    }
    mult->execute(params.handle, Workspace(), space);
  }

private:
  std::unique_ptr<MatrixOp> form_operation(GEMM_Inputs params, GEMM_Options opts) {

    std::unique_ptr<MatrixOp> A = std::make_unique<NoOp>(params.A);
    std::unique_ptr<MatrixOp> B = std::make_unique<NoOp>(params.B);
    std::unique_ptr<MatrixOp> C = std::make_unique<NoOp>(params.C);

    bool transa = opts.transa() == TRANS;
    bool transb = opts.transb() == TRANS;
    bool transc = opts.transc() == TRANS;
    bool pada = opts.pada() == PAD;
    bool padb = opts.padb() == PAD;
    bool padc = opts.padc() == PAD;

    if (transa) 
      params.transa = switch_op(params.transa);
    if (transa || pada)
      A = std::make_unique<MatrixMove>(std::move(A), 1.0, transa, pada ? 32 : 1);

    if (transb)
      params.transb = switch_op(params.transb);
    if (transb || padb)
      B = std::make_unique<MatrixMove>(std::move(B), 1.0, transb, padb ? 32 : 1);

    if (transc) {
      auto scratch = std::make_unique<MatrixMultAlloc>(std::move(B), std::move(A), 
                                                       params.transb != CUBLAS_OP_T, 
                                                       params.transa != CUBLAS_OP_T, 
                                                       params.alpha, padc ? 32 : 1);

      return std::make_unique<MatrixAccumulate>(std::move(scratch), std::move(C), 
                                                1.0, params.beta, true);
    } else if (padc) {
      auto scratch = std::make_unique<MatrixMultAlloc>(std::move(A), std::move(B),
                                                       params.transa == CUBLAS_OP_T, 
                                                       params.transb == CUBLAS_OP_T, 
                                                       params.alpha, 32);

      return std::make_unique<MatrixAccumulate>(std::move(scratch), std::move(C), 
                                                1.0, params.beta, false);
    } else {
      return std::make_unique<MatrixMult>(std::move(A), std::move(B), std::move(C), 
                      params.transa == CUBLAS_OP_T, params.transb == CUBLAS_OP_T,
                      params.alpha, params.beta);
    }
  }
};

template class Planning_System<GEMM_Executor>;
using GEMM_Planner = Planning_System<GEMM_Executor>;

// class GEMM_Planner : public Planning_System<GEMM_Executor> {
//   unsigned int tests_until_converge = 1;
// public:
//   //std::istream& load_predicates(std::istream &is) {
//   //  std::vector<Predicate<std::pair<GEMM_Options, GEMM_Key>>> preds;
//   //  std::string s;
// 
//   //  while (std::getline(is,s)) {
//   //    std::stringstream ss(s);
// 
//   //    std::string ops;
//   //    ss >> ops;
// 
//   //    GEMM_Options opts;
//   //    ss >> opts;
//   //    if (ops.size() != 2 || ss.fail()) {
//   //      std::cout << "Parse failure on line: " << s << ", size " << s.size() << std::endl;
//   //      throw;
//   //    }
//   //    cublasOperation_t opA = ops[0] == 'T' ? CUBLAS_OP_T : CUBLAS_OP_N;
//   //    cublasOperation_t opB = ops[1] == 'T' ? CUBLAS_OP_T : CUBLAS_OP_N;
// 
//   //    preds.push_back(permit_option(opts, opA, opB));
//   //  } 
//   //  predicates = {disjunction(preds)};
// 
//   //  {
//   //    std::vector<cublasOperation_t> ops = {CUBLAS_OP_N, CUBLAS_OP_T};
// 
//   //    for (auto opts : GEMM_Options::enumerate()) {
//   //      for (auto &opA : ops) {
//   //        for (auto &opB : ops) {
//   //          GEMM_Key k(opA,opB,1,1,1);
// 
//   //          bool good = true;
//   //          for (auto &pred : predicates) 
//   //            good = good && pred(std::make_pair(opts,k));
// 
//   //          if (good) {
//   //            std::cout << "Permitted: " << ((opA == CUBLAS_OP_N) ? "N" : "T")
//   //                                       << ((opB == CUBLAS_OP_N) ? "N" : "T") << " "
//   //                                       << op_to_char(std::get<0>(opts)) 
//   //                                       << op_to_char(std::get<2>(opts)) 
//   //                                       << op_to_char(std::get<4>(opts)) 
//   //                                       << op_to_char(std::get<1>(opts)) 
//   //                                       << op_to_char(std::get<3>(opts)) 
//   //                                       << op_to_char(std::get<5>(opts)) << std::endl;
//   //          }
//   //        }
//   //      }
//   //    }
//   //  }
//   //  return is;
//   //}
// 
//   GEMM_Options create_plan(GEMM_Inputs params) override {
//     // TODO actually come up with a method to determine a plan
//     auto &an = get_analytics(params);
// 
//     //GEMM_Options plan(NOTRANS, NOTRANS);
//     GEMM_Options plan = an.performance_data.begin()->first;
//     unsigned int min_count = 10000;
//     for (auto &[opts, rec] : an.performance_data)
//       if (rec.count() < min_count) min_count = rec.count();
// 
//     if (min_count < tests_until_converge) 
//       for (auto &[opts, rec] : an.performance_data)
//         if (rec.count() == min_count) return opts;
// 
//     float ms = std::numeric_limits<float>::infinity();
//     for (auto &[opts, rec] : an.performance_data) {
//       float t = rec.get_time();
//       rec.synchronous = false;
//       if (t < ms) {
//         plan = opts;
//         ms = t;
//       }
//     }
// 
//     return plan;
//   }
// 
//   GEMM_Options degrade_plan(GEMM_Inputs params) {
//     // TODO: Plan degradation should interact somehow with plan selection.
//     // Should maybe call create_plan with a reduced set of options?
//     // Make the set of options a parameter for create_plan? Could be good
//     Analytics &an = get_analytics(params);
//     std::vector<std::pair<GEMM_Options, float>> data;
//     for (auto &[opts, rec] : an.performance_data)
//       data.emplace_back(opts, rec.get_time());
// 
//     std::sort(data.begin(), data.end(), [](const std::pair<GEMM_Options, float> a,
//                                            const std::pair<GEMM_Options, float> b) {
//       return a.second < b.second;
//     });
// 
//     for (auto &x : data) {
//       if (calculate_workspace(x.first, params) <= params.space.size())
//         return x.first;
//     }
//     std::cout << "No valid plan found" << std::endl;
//     throw;
//   }
// 
//   double get_floprate(GEMM_Options opts, GEMM_Inputs params) {
//     Analytics &an = get_analytics(params);
// 
//     double secs = (double)(an.performance_data[opts].get_time())/1000.0;
//     double tflops = 2*params.m()*params.k()*params.n()/1e12;
// 
//     return tflops/secs;
//   }
// 
//   double get_floprate(GEMM_Inputs params) {
//     Analytics &an = get_analytics(params);
// 
//     size_t count = 0;
//     float total = 0.0;
//     for (auto &[opts,rec] : an.performance_data) {
//       if (rec.count() > 0) {
//         rec.flush();
//         count += rec.count();
//         total += rec.count()*rec.get_time();
//       }
//     }
//     std::cout << "Count = " << count << ", Total = " << total << std::endl;
//     double secs = (double)(total/count)/1000.0;
//     double tflops = 2*params.m()*params.k()*params.n()/1e12;
//     std::cout << "Rate = " << tflops/secs << std::endl;
// 
//     return tflops/secs;
//   }
// };

}
