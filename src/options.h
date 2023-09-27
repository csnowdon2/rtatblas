#pragma once
#include <vector>
#include <tuple>

template<typename... Args>
struct Enumerate;

template<typename T, typename S, typename... Rest>
struct Enumerate<T, S, Rest...> {
  static constexpr std::vector<std::tuple<T, S, Rest...>> get() {
    std::vector<std::tuple<T, S, Rest...>> ret;
  
    auto rest = Enumerate<S, Rest...>::get();

    for (auto &r : rest) {
      for (auto &op : T::vals) {
        auto t = std::tuple_cat(std::make_tuple(op), r);
        ret.push_back(t);
      }
    }
    return ret;
  }
};

template<typename T>
struct Enumerate<T> {
  static constexpr std::vector<std::tuple<T>> get() {
    std::vector<std::tuple<T>> ret;
    for (auto &op : T::vals) {
      ret.push_back(std::make_tuple(op));
    }
    return ret;
  }
};


template<typename T, T... VALS>
class Option {
public:
  static constexpr std::array<T, sizeof...(VALS)> vals = { { VALS... } };
  static constexpr int N = sizeof...(VALS); 

  Option() {}
  Option(T val) : val(val) {}

  operator T() {return val;}

  T val;

  bool operator==(const Option &other) const {return val == other.val;}
  bool operator==(const T &other) const {return val == other;}

  friend bool operator<(const Option &l, const Option &r) { 
    return l.val < r.val;
  }
};

template<typename... Ops>
struct Options : public std::tuple<Ops...> {

  using std::tuple<Ops...>::tuple;

  Options(std::tuple<Ops...> tup) : std::tuple<Ops...>(tup) {}

  static constexpr std::vector<Options> enumerate() {
    auto ops = Enumerate<Ops...>::get();
    return std::vector<Options>(ops.begin(), ops.end());
  }
};
