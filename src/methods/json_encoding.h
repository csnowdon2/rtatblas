#pragma once
#include <type_traits>
#include <nlohmann/json.hpp>
#include "gpu-api.h"
#include "gemm.h"
#include "syrk.h"
#include "trsm.h"

namespace rtat {

template<typename T>
T from_json(const nlohmann::json);

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

  json["transA"] = std::string(opA);
  json["transB"] = std::string(opB);
  json["m"] = m;
  json["n"] = n;
  json["k"] = k;
  return json;
}

template<>
inline GEMM_Key from_json(const nlohmann::json json) {
  return GEMM_Key(
        BLAS_Operation(json["transA"].get<std::string>()),
        BLAS_Operation(json["transB"].get<std::string>()),
        json["m"].get<int>(),
        json["k"].get<int>(),
        json["n"].get<int>());
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

template<>
inline GEMM_Options from_json(const nlohmann::json json) {
  return GEMM_Options(
      BLAS_Op(json["transA"].get<std::string>()),
      Pad_Op(json["padA"].get<std::string>()),
      BLAS_Op(json["transB"].get<std::string>()),
      Pad_Op(json["padB"].get<std::string>()),
      BLAS_Op(json["transC"].get<std::string>()),
      Pad_Op(json["padC"].get<std::string>()));
}

template<typename A, typename B, typename C, typename D>
constexpr bool verify_SYRK_Key_components() {
  return std::is_same_v<A, BLAS_Fill_Mode>
      && std::is_same_v<B, BLAS_Operation>
      && std::is_same_v<C, int>
      && std::is_same_v<D, int>;
}

inline nlohmann::json to_json(SYRK_Key key) {
  nlohmann::json json;
  auto &[uplo, trans, n, k] = key;
  static_assert(verify_SYRK_Key_components<decltype(uplo),
      decltype(trans),decltype(n),decltype(k)>());

  json["uplo"] = std::string(uplo);
  json["trans"] = std::string(trans);
  json["n"] = n;
  json["k"] = k;
  return json;
}

template<>
inline SYRK_Key from_json(const nlohmann::json json) {
  return SYRK_Key(
        BLAS_Fill_Mode(json["uplo"].get<std::string>()),
        BLAS_Operation(json["trans"].get<std::string>()),
        json["n"].get<int>(),
        json["k"].get<int>());
}

template<typename A, typename B>
constexpr bool verify_SYRK_Options_components() {
  return std::is_same_v<A, Bool_Op>
      && std::is_same_v<B, Bool_Op>;
}

inline nlohmann::json to_json(SYRK_Options opts) {
  nlohmann::json json;
  auto &[tA, tC] = opts;
  static_assert(verify_SYRK_Options_components<decltype(tA),decltype(tC)>());

  json["transA"] = std::string(tA);
  json["transC"] = std::string(tC);
  return json;
}

template<>
inline SYRK_Options from_json(const nlohmann::json json) {
  return SYRK_Options(
      Bool_Op(json["transA"].get<std::string>()),
      Bool_Op(json["transC"].get<std::string>()));
}

template<typename A, typename B, typename C, typename D, 
  typename E, typename F>
constexpr bool verify_TRSM_Key_components() {
  return std::is_same_v<A, BLAS_Side>
      && std::is_same_v<B, BLAS_Fill_Mode>
      && std::is_same_v<C, BLAS_Operation>
      && std::is_same_v<D, BLAS_Diag>
      && std::is_same_v<E, int>
      && std::is_same_v<F, int>;
}

inline nlohmann::json to_json(TRSM_Key key) {
  nlohmann::json json;
  auto &[side, uplo, trans, diag, m, n] = key;
  static_assert(verify_TRSM_Key_components<decltype(side), 
      decltype(uplo), decltype(trans), decltype(diag),
      decltype(m), decltype(n)>());

  json["side"] = std::string(side);
  json["uplo"] = std::string(uplo);
  json["trans"] = std::string(trans);
  json["diag"] = std::string(diag);
  json["m"] = m;
  json["n"] = n;
  return json;
}

template<>
inline TRSM_Key from_json(const nlohmann::json json) {
  return TRSM_Key(
      BLAS_Side(json["side"].get<std::string>()),
      BLAS_Fill_Mode(json["uplo"].get<std::string>()),
      BLAS_Operation(json["trans"].get<std::string>()),
      BLAS_Diag(json["diag"].get<std::string>()),
      json["m"].get<int>(), 
      json["n"].get<int>());
}

template<typename A, typename B>
constexpr bool verify_TRSM_Options_components() {
  return std::is_same_v<A, Bool_Op>
      && std::is_same_v<B, Bool_Op>;
}

inline nlohmann::json to_json(TRSM_Options opts) {
  nlohmann::json json;
  auto &[swap_side, tA] = opts;
  static_assert(verify_TRSM_Options_components<decltype(swap_side),decltype(tA)>());

  json["swap_side"] = std::string(swap_side);
  json["transA"] = std::string(tA);
  return json;
}

template<>
inline TRSM_Options from_json(const nlohmann::json json) {
  return TRSM_Options(
      Bool_Op(json["swap_side"].get<std::string>()),
      Bool_Op(json["transA"].get<std::string>()));
}

}
