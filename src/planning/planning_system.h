#pragma once

#include <map>
#include <gpu-api.h>
#include <numeric>

#include <gemm.h>
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



template class Planning_System<GEMM_Executor>;
using GEMM_Planner = Planning_System<GEMM_Executor>;

}

