#pragma once
#include <type_traits>
#include <nlohmann/json.hpp>
#include "gpu-api.h"
#include "gemm.h"
//#include "syrk.h"
//#include "trsm.h"

namespace rtat {

template<typename A, typename B, typename C, typename D, typename E>
constexpr bool verify_GEMM_Key_components() {
  return std::is_same_v<A, BLAS_Operation>
      && std::is_same_v<B, BLAS_Operation>
      && std::is_same_v<C, int>
      && std::is_same_v<D, int>
      && std::is_same_v<E, int>;
}

inline nlohmann::json to_json(GEMM_Key key) {
  nlohmann::json json;
  auto &[opA, opB, m, n, k] = key;
  static_assert(verify_GEMM_Key_components<decltype(opA),
      decltype(opB),decltype(m),decltype(n),decltype(k)>());

  json["opA"] = std::string(opA);
  json["opB"] = std::string(opB);
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

//template<typename A, typename B, typename C, typename D>
//constexpr bool verify_SYRK_Key_components() {
//  return std::is_same_v<A, cublasFillMode_t>
//      && std::is_same_v<B, cublasOperation_t>
//      && std::is_same_v<C, int>
//      && std::is_same_v<D, int>;
//}
//
//inline nlohmann::json to_json(SYRK_Key key) {
//  nlohmann::json json;
//  auto &[uplo, trans, n, k] = key;
//  static_assert(verify_SYRK_Key_components<decltype(uplo),
//      decltype(trans),decltype(n),decltype(k)>());
//
//  json["uplo"] = (uplo == CUBLAS_FILL_MODE_LOWER) ? "L" : "U";
//  json["trans"] = (trans == CUBLAS_OP_N) ? "N" : "T";
//  json["n"] = n;
//  json["k"] = k;
//  return json;
//}
//
//template<typename A, typename B>
//constexpr bool verify_SYRK_Options_components() {
//  return std::is_same_v<A, Bool_Op>
//      && std::is_same_v<B, Bool_Op>;
//}
//
//inline nlohmann::json to_json(SYRK_Options opts) {
//  nlohmann::json json;
//  auto &[tA, tC] = opts;
//  static_assert(verify_SYRK_Options_components<decltype(tA),decltype(tC)>());
//
//  json["transA"] = std::string(tA);
//  json["transC"] = std::string(tC);
//  return json;
//}
//
//template<typename A, typename B, typename C, typename D, 
//  typename E, typename F>
//constexpr bool verify_TRSM_Key_components() {
//  return std::is_same_v<A, cublasSideMode_t>
//      && std::is_same_v<B, cublasFillMode_t>
//      && std::is_same_v<C, cublasOperation_t>
//      && std::is_same_v<D, cublasDiagType_t>
//      && std::is_same_v<E, int>
//      && std::is_same_v<F, int>;
//}
//
//inline nlohmann::json to_json(TRSM_Key key) {
//  nlohmann::json json;
//  auto &[side, uplo, trans, diag, m, n] = key;
//  static_assert(verify_TRSM_Key_components<decltype(side), 
//      decltype(uplo), decltype(trans), decltype(diag),
//      decltype(m), decltype(n)>());
//
//  json["side"] = (side == CUBLAS_SIDE_LEFT) ? "Left" : "Right";
//  json["uplo"] = (uplo == CUBLAS_FILL_MODE_LOWER) ? "Lower" : "Upper";
//  json["trans"] = (trans == CUBLAS_OP_N) ? "N" : "T";
//  json["diag"] = (diag == CUBLAS_DIAG_UNIT) ? "U" : "N";
//  json["m"] = m;
//  json["n"] = n;
//  return json;
//}
//
//template<typename A, typename B>
//constexpr bool verify_TRSM_Options_components() {
//  return std::is_same_v<A, Bool_Op>
//      && std::is_same_v<B, Bool_Op>;
//}
//
//inline nlohmann::json to_json(TRSM_Options opts) {
//  nlohmann::json json;
//  auto &[swap_side, tA] = opts;
//  static_assert(verify_TRSM_Options_components<decltype(swap_side),decltype(tA)>());
//
//  json["swap_side"] = std::string(swap_side);
//  json["transA"] = std::string(tA);
//  return json;
//}

}
