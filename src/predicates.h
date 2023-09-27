#pragma once
#include <vector>
#include <functional>
#include <numeric>

template<typename T>
using Predicate = std::function<bool(T)>;

template<typename T>
Predicate<T> disjunction(std::vector<Predicate<T>> preds) {
  return [preds](T val) {
    std::vector<bool> results;
    std::transform(preds.cbegin(), preds.cend(),
                   std::back_inserter(results),
                   [&val](Predicate<T> pred) {return pred(val);});
    return std::accumulate(results.cbegin(), results.cend(), false,
                           [](bool a, bool b) { return a || b; });
  };
}

