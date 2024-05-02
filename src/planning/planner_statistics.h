#pragma once
#include <vector>
#include <map>
#include <numeric>
#include <algorithm>
#include <json_encoding.h>

template<typename Key, typename Opts>
class Planner_Statistics {
  std::map<Key, std::map<Opts, std::vector<float>>> times; 
  std::map<Key, std::map<Opts, std::vector<float>>> floprates; 
  std::map<Key, std::map<Opts, float>> means; 
  std::map<Key, std::map<Opts, size_t>> counts; 
public:
  Planner_Statistics(
      std::map<Key, std::map<Opts, std::vector<float>>> times) 
    : times(times) {

    for (auto &[key, opt_map] : times) {
      for (auto &[opt, times] : opt_map) {
        counts[key][opt] = times.size();
        means [key][opt] = 
          std::accumulate(times.cbegin(),times.cend(),0.0)/times.size();

      }
    }
  }

  const std::map<Key, std::map<Opts, std::vector<float>>>& get_times() {return times;}
  const std::map<Key, std::map<Opts, float>>& get_means() {return means;} 
  const std::map<Key, std::map<Opts, size_t>>& get_counts() {return counts;} 

  // FLOP rates are constructed separately for SFINAE reasons
  // Don't want to force Opts to have a flopcount necessarily
  const std::map<Key, std::map<Opts, size_t>>& get_floprates() {
    if (floprates.size() > 0) return floprates;

    for (auto &[key, opt_map] : times) {
      for (auto &[opt, times] : opt_map) {
        auto k = key;
        std::transform(times.begin(), times.end(), 
                       std::back_inserter(floprates[key][opt]),
                       [&k](float x) {return k.flopcount()/x;});
      }
    }

    return floprates;
  }

  nlohmann::json json() {
    nlohmann::json json;
    for (auto &[key, opt_map] : times) {
      nlohmann::json key_json = to_json(key);

      key_json["options"] = nlohmann::json();
      for (auto &[opt, ts] : opt_map) {
        nlohmann::json opt_json;
        opt_json["option"] = to_json(opt);
        opt_json["times"] = ts;
        key_json["options"].push_back(opt_json);
      }

      json.push_back(key_json);
    }
    return json;
  }
};
