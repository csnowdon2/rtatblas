#pragma once
#include <vector>
#include <functional>

namespace rtat {

template<typename T>
using Predicate = std::function<bool(T)>;

template<typename T>
Predicate<T> disjunction(std::vector<Predicate<T>> preds) {
  return [preds](T val) {
    for (auto &pred : preds) {
      if (pred(val)) 
        return true;
    }
    return false;
  };
}

template<typename T>
Predicate<T> conjunction(std::vector<Predicate<T>> preds) {
  return [preds](T val) {
    for (auto &pred : preds) {
      if (!pred(val)) 
        return false;
    }
    return true;
  };
}

}
