#include <gtest/gtest.h>
#include <json_encoding.h>
using namespace rtat;

template<typename T>
void json_compare(nlohmann::json j1, nlohmann::json j2, std::string key) {
  ASSERT_EQ(j1[key].get<T>(),
      j2[key].get<T>());
}

TEST(JSON_Test, SYRK_Key) {
  const int n = 6741;
  const int k = 2463;
  for (auto &uplo : {"Lower", "Upper"}) {
    for (auto &trans : {"N","T"}) {
      nlohmann::json key_json;
      key_json["uplo"] = uplo;
      key_json["trans"] = trans;
      key_json["n"] = n;
      key_json["k"] = k;

      SYRK_Key key = from_json<SYRK_Key>(key_json);
      nlohmann::json test_json = to_json(key);

      ASSERT_EQ(test_json, key_json);
    }
  }
  for (auto &uplo : {"Lower", "Upper"}) {
    for (auto &trans : {"N","T"}) {
      SYRK_Key key(BLAS_Fill_Mode(uplo), BLAS_Operation(trans), n, k);
      nlohmann::json json = to_json(key);
      SYRK_Key test_key = from_json<SYRK_Key>(json);

      ASSERT_TRUE(!(test_key < key) && !(key < test_key));
    }
  }
}

TEST(JSON_Test, SYRK_Options) {
  for (auto &transA : {"T","F"}) {
    for (auto &transC : {"T","F"}) {
      nlohmann::json opts_json;
      opts_json["transA"] = transA;
      opts_json["transC"] = transC;

      SYRK_Options opts = from_json<SYRK_Options>(opts_json);
      nlohmann::json test_json = to_json(opts);

      ASSERT_EQ(test_json, opts_json);
    }
  }
  for (auto &transA : {false,true}) {
    for (auto &transC : {false,true}) {
      SYRK_Options opts(transA, transC);
      nlohmann::json json = to_json(opts);
      SYRK_Options test_opts = from_json<SYRK_Options>(json);

      ASSERT_TRUE(!(test_opts < opts) && !(opts < test_opts));
    }
  }
}

TEST(JSON_Test, TRSM_Key) {
  const int n = 4001;
  const int m = 10025;
  for (auto &side : {"Left","Right"}) {
    for (auto &uplo : {"Lower", "Upper"}) {
      for (auto &trans : {"N","T"}) {
        for (auto &diag : {"Unit", "Non-Unit"}) {
          nlohmann::json key_json;
          key_json["side"] = side;
          key_json["uplo"] = uplo;
          key_json["trans"] = trans;
          key_json["diag"] = diag;
          key_json["m"] = m;
          key_json["n"] = n;

          TRSM_Key key = from_json<TRSM_Key>(key_json);
          nlohmann::json test_json = to_json(key);

          ASSERT_EQ(test_json, key_json);
        }
      }
    }
  }
  for (auto &side : {"Left","Right"}) {
    for (auto &uplo : {"Lower", "Upper"}) {
      for (auto &trans : {"N","T"}) {
        for (auto &diag : {"Unit", "Non-Unit"}) {
          TRSM_Key key(
              BLAS_Side(side),
              BLAS_Fill_Mode(uplo), 
              BLAS_Operation(trans), 
              BLAS_Diag(diag),
              m, n);
          nlohmann::json json = to_json(key);
          TRSM_Key test_key = from_json<TRSM_Key>(json);

          ASSERT_TRUE(!(test_key < key) && !(key < test_key));
        }
      }
    }
  }
}

TEST(JSON_Test, TRSM_Options) {
  for (auto &transA : {"T","F"}) {
    for (auto &swap_side : {"T","F"}) {
      nlohmann::json opts_json;
      opts_json["swap_side"] = swap_side;
      opts_json["transA"] = transA;

      TRSM_Options opts = from_json<TRSM_Options>(opts_json);
      nlohmann::json test_json = to_json(opts);

      ASSERT_EQ(test_json, opts_json);
    }
  }
  for (auto &transA : {false,true}) {
    for (auto &swap_side : {false,true}) {
      TRSM_Options opts(swap_side, transA);
      nlohmann::json json = to_json(opts);
      TRSM_Options test_opts = from_json<TRSM_Options>(json);

      ASSERT_TRUE(!(test_opts < opts) && !(opts < test_opts));
    }
  }
}
