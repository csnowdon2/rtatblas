#pragma once

#include <map>
#include <iostream>

#include <gpu-api.h>
#include <timer_bank.h>
#include <numeric>

#include "matrix_ops/matrixop.h"
//#include "plan.h"
#include "methods/gemm.h"
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


class GEMM_Executor : public Executor<GEMM_Inputs, GEMM_Key, GEMM_Options> {
public:
  size_t calculate_workspace(GEMM_Inputs params, GEMM_Options opts) override {
    auto mult = opts.form_operation(params);
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
    auto mult = opts.form_operation(params);
    if (mult->workspace_req() > space.size()) {
      throw "GEMM internal_execute: Insufficient workspace";
    }
    mult->execute(params.handle, Workspace(), space);
  }
};

template class Planning_System<GEMM_Executor>;
using GEMM_Planner = Planning_System<GEMM_Executor>;

}
