#include <gemm.h>
#include <planning_system.h>


namespace rtat {

template<typename T>
class Lazy {
  std::unique_ptr<T> val;
public:
  operator T&() {
    if (!val) val = std::make_unique<T>();
    return *val;
  }
};

// Should contain planning systems for all operations, so 
// we can call e.g. rtat.gemm_planner().create_plan(...) 
class rtat {
  Lazy<Planning_System<GEMM_Executor<double>>> dgemm_planner;
  Lazy<Planning_System<GEMM_Executor<float>>> sgemm_planner;
public:
  template<typename T>
  Planning_System<GEMM_Executor<T>>& gemm_planner();

  template<>
  Planning_System<GEMM_Executor<double>>& gemm_planner() {
    return dgemm_planner;
  }

  template<>
  Planning_System<GEMM_Executor<float>>& gemm_planner() {
    return sgemm_planner;
  }
};

}
