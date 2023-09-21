#include <fstream>
#include <planning_system.h>
#include <nlohmann/json.hpp>

nlohmann::json jsonify(GEMM_Key key) {
  nlohmann::json ret;
  ret["m"] = key.m;
  ret["k"] = key.k;
  ret["n"] = key.n;
  ret["opA"] = key.transa == CUBLAS_OP_N ? "N" : "T";
  ret["opB"] = key.transb == CUBLAS_OP_N ? "N" : "T";
  return ret;
}

template<typename Planner>
class JSON_Planner : public Planner {
public:
  using Planner::Planner;
  
  void json_output(std::ostream &os) {
    nlohmann::json json;
    
    for (auto &[key,an] : this->analytics) {
      nlohmann::json j = jsonify(key);

      for (auto &[opts, rec] : an.performance_data) {
        auto op_str = std::string(opts);
        j["results"][op_str]["mean"] = rec.get_time();
        j["results"][op_str]["data"] = rec.data();
      }

      json.push_back(j);
    }

    os << std::setw(2) << json;
  }

  void json_output(std::string filename) {
    std::string ofname;
    int i = 1;
    do {
      {
        ofname = filename + std::to_string(i++);
        std::ifstream is(ofname);
        if (is.good()) {
          continue;
        }
      }

      std::ofstream os(ofname);
      json_output(os);
      break;
    } while (true);
  }
};
