#include <gemm.h>
#include <syrk.h>
#include <trsm.h>
#include <planning_system.h>
#include <memory>


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

class rtat {
  Lazy<Planning_System<GEMM_Executor<double>>> dgemm_planner;
  Lazy<Planning_System<GEMM_Executor<float>>> sgemm_planner;
  Lazy<Planning_System<TRSM_Executor<double>>> dtrsm_planner;
  Lazy<Planning_System<TRSM_Executor<float>>> strsm_planner;
  Lazy<Planning_System<SYRK_Executor<double>>> dsyrk_planner;
  Lazy<Planning_System<SYRK_Executor<float>>> ssyrk_planner;
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

  template<typename T>
  Planning_System<TRSM_Executor<T>>& trsm_planner();
  template<>
  Planning_System<TRSM_Executor<double>>& trsm_planner() {
    return dtrsm_planner;
  }
  template<>
  Planning_System<TRSM_Executor<float>>& trsm_planner() {
    return strsm_planner;
  }

  template<typename T>
  Planning_System<SYRK_Executor<T>>& syrk_planner();
  template<>
  Planning_System<SYRK_Executor<double>>& syrk_planner() {
    return dsyrk_planner;
  }
  template<>
  Planning_System<SYRK_Executor<float>>& syrk_planner() {
    return ssyrk_planner;
  }
};

}
