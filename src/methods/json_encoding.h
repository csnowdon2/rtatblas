#pragma once
#include <type_traits>
#include <nlohmann/json.hpp>
#include "gemm.h"

namespace rtat {

template<typename A, typename B, typename C, typename D, typename E>
constexpr bool verify_GEMM_Key_components() {
  return std::is_same_v<A, cublasOperation_t>
      && std::is_same_v<B, cublasOperation_t>
      && std::is_same_v<C, int>
      && std::is_same_v<D, int>
      && std::is_same_v<E, int>;
}

inline nlohmann::json to_json(GEMM_Key key) {
  nlohmann::json json;
  auto &[opA, opB, m, n, k] = key;
  static_assert(verify_GEMM_Key_components<decltype(opA),
      decltype(opB),decltype(m),decltype(n),decltype(k)>());

  json["opA"] = (opA == CUBLAS_OP_N) ? "N" : "T";
  json["opB"] = (opA == CUBLAS_OP_N) ? "N" : "T";
  json["m"] = m;
  json["n"] = n;
  json["k"] = k;
  return json;
}

template<typename A, typename B, typename C, typename D, typename E, typename F>
constexpr bool verify_GEMM_Options_components() {
  return std::is_same_v<A, BLAS_Op>
      && std::is_same_v<B, Pad_Op>
      && std::is_same_v<C, BLAS_Op>
      && std::is_same_v<D, Pad_Op>
      && std::is_same_v<E, BLAS_Op>
      && std::is_same_v<F, Pad_Op>;
}

inline nlohmann::json to_json(GEMM_Options opts) {
  nlohmann::json json;
  auto &[ta, pa, tb, pb, tc, pc] = opts;
  static_assert(verify_GEMM_Options_components<decltype(ta),decltype(pa),
      decltype(tb),decltype(pb),decltype(tc),decltype(pc)>());

  json["transA"] = std::string(ta);
  json["transB"] = std::string(tb);
  json["transC"] = std::string(tc);
  json["padA"] = std::string(pa);
  json["padB"] = std::string(pb);
  json["padC"] = std::string(pc);
  return json;
}

}
