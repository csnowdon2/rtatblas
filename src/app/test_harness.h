#pragma once
#include <nlohmann/json.hpp>
#include <json_encoding.h>
#include <stdexcept>
#include <vector>
#include <gpu-api.h>
#include <gemm.h>
#include <syrk.h>
#include <trsm.h>

namespace rtat {


class Run_Type {
public:
  enum _Run_Type {
    EXHAUSTIVE,
    AUTOTUNE
  };
  _Run_Type val;

  Run_Type(std::string rt) {
    if (rt == "exhaustive") {
      val = EXHAUSTIVE;
    } else if (rt == "autotune") {
      val = AUTOTUNE;
    } else {
      throw std::runtime_error("Invalid run_type: "+rt);
    }
  }

  operator std::string() {
    switch (val) {
      case EXHAUSTIVE:
        return "exhaustive";
      case AUTOTUNE:
        return "autotune";
    }
  }
};

class Method {
public:
  enum _Method {
    GEMM,
    TRSM,
    SYRK
  };
  _Method val;

  Method(std::string m) {
    if (m == "gemm") {
      val = GEMM;
    } else if (m == "syrk") {
      val = SYRK;
    } else if (m == "trsm") {
      val = TRSM;
    }else {
      throw std::runtime_error("Invalid method: "+m);
    }
  }

  operator std::string() {
    switch (val) {
      case GEMM:
        return "gemm";
      case SYRK:
        return "syrk";
      case TRSM:
        return "trsm";
    }
  }
};

class Data_Type {
public:
  enum _Data_Type {
    FLOAT,
    DOUBLE
  };
  _Data_Type val;

  Data_Type(std::string dt) {
    if (dt == "double") {
      val = DOUBLE;
    } else if (dt == "float") {
      val = FLOAT;
    } else {
      throw std::runtime_error("Invalid data_type: "+dt);
    }
  }

  operator std::string() {
    switch (val) {
      case DOUBLE:
        return "double";
      case FLOAT:
        return "float";
    }
  }
};

template<typename Key>
class Problem_Set {
  std::vector<Key> problems;
public:
  Problem_Set(const nlohmann::json& json) {
    for (auto &key_json : json) 
      problems.push_back(from_json<Key>(key_json));
  }

  const std::vector<Key>& get_problems() {
    return problems;
  }
};

struct Input_File {
  //const Problem_Set<Key> problems;
  const nlohmann::json problem_json;
  const Run_Type run_type;
  const Method method;
  const Data_Type data_type;
  const int repetitions;

  Input_File(const nlohmann::json& input_json) 
    : problem_json(input_json["problems"]),
      run_type(input_json["run_type"].get<std::string>()),
      method(input_json["method"].get<std::string>()),
      data_type(input_json["data_type"].get<std::string>()),
      repetitions(input_json["repetitions"].get<int>()) {}
};

struct Device_Resources {
private:
  ManagedWorkspace space;
  Device_RNG rng;
public:
  Stream s;
  cublasHandle_t handle;
  ManagedWorkspace scratch_space;

  Device_Resources() : space(1024), scratch_space(1024) {
    cublasCreate(&handle);
    cublasSetStream(handle,s);
  }

  ~Device_Resources() {
    cublasDestroy(handle);
  }

  void sync() {s.synchronize();}

  template<typename T>
  std::vector<Matrix<T>> allocate_matrices(
      const std::vector<MatrixDims> &dim_vector) {
    std::vector<size_t> sizes;
    std::vector<Matrix<T>> ret;
    size_t required_size = 0;

    for (auto &dims : dim_vector) {
      size_t s = ((dims.footprint()*sizeof(T)+511)/512)*512/sizeof(T);
      sizes.push_back(s);
      required_size += s;
    }
    space.grow_to_fit<T>(required_size);

    size_t offset = 0;
    for (size_t i=0; i<dim_vector.size(); i++) {
      Workspace ws(space, offset, sizes[i]);
      rng.uniform<T>((T*)ws, ws.size<T>());

      ret.emplace_back(ws, dim_vector[i]);
      offset += sizes[i]*sizeof(T);
    }

    return ret;
  }
};

template<typename Planner_Type>
class Test_Harness {
  using Params = typename Planner_Type::Params;
  using Key    = typename Planner_Type::Key;
  using Opts   = typename Planner_Type::Opts;
  using Scalar = typename Params::Scalar;

  Problem_Set<Key> problems;
  Device_Resources resources;
public:
  Test_Harness(nlohmann::json problem_json) 
    : problems(problem_json) {}

  nlohmann::json run_exhaustive(int repetitions) {
    Planner_Type planner;
    for (auto &problem : problems.get_problems()) {
      Params input = form_input<Scalar>(problem);
      for (auto &opts : Opts::enumerate()) {
        for (int i=0; i<repetitions; i++) {
          resources.scratch_space.grow_to_fit<char>(
              planner.calculate_workspace(input,opts));
          planner.execute(
              input, opts, resources.scratch_space, resources.s);
          resources.sync();
        }
      }
    }
    return planner.make_statistics().json();
  }

  nlohmann::json run_autotune([[maybe_unused]] int repetitions) {
    Planner_Type planner;
    for (auto &problem : problems.get_problems()) {
      for (int i=0; i<repetitions; i++) {
        Params input = form_input<Scalar>(problem);
        auto opts = planner.create_plan(problem);
        resources.scratch_space.grow_to_fit<char>(
            planner.calculate_workspace(input,opts));
        planner.execute(
            input, opts, resources.scratch_space, resources.s);
        resources.sync();
      }
    }
    return planner.make_statistics().json();
  }

  template<typename T>
  GEMM_Inputs<T> form_input(GEMM_Key key) {
    MatrixDims Adims(key.transa == CUBLAS_OP_N ? key.m : key.k,
                     key.transa == CUBLAS_OP_N ? key.k : key.m,
                     key.transa == CUBLAS_OP_N ? key.m : key.k);
    MatrixDims Bdims(key.transb == CUBLAS_OP_N ? key.k : key.n,
                     key.transb == CUBLAS_OP_N ? key.n : key.k,
                     key.transb == CUBLAS_OP_N ? key.k : key.n);
    MatrixDims Cdims(key.m, key.n, key.m);

    auto matrices = 
      resources.allocate_matrices<T>({Adims,Bdims,Cdims});

    return GEMM_Inputs<T>(resources.handle, 
        key.transa, key.transb, matrices[0], matrices[1],
        matrices[2], 1.0, 0.0);
  }

  template<typename T>
  TRSM_Inputs<T> form_input(TRSM_Key key) {
    size_t mA = key.side == CUBLAS_SIDE_LEFT ? key.m : key.n;
    MatrixDims Adims(mA,mA,mA);
    MatrixDims Bdims(key.m, key.n, key.m);

    auto matrices = 
      resources.allocate_matrices<T>({Adims,Bdims});

    return TRSM_Inputs<T>(resources.handle, 
        key.side, key.uplo, key.trans, key.diag, 
        matrices[0], matrices[1], 1.0);
  }

  template<typename T>
  SYRK_Inputs<T> form_input(SYRK_Key key) {
    MatrixDims Adims(key.trans == CUBLAS_OP_N ? key.n : key.k,
                     key.trans == CUBLAS_OP_N ? key.k : key.n,
                     key.trans == CUBLAS_OP_N ? key.n : key.k);
    MatrixDims Cdims(key.n, key.n, key.n);

    auto matrices = 
      resources.allocate_matrices<T>({Adims,Cdims});

    return SYRK_Inputs<T>(resources.handle, 
        key.uplo, key.trans,  
        matrices[0], matrices[1], 1.0, 0.0);
  }
};

}
