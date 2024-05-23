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

TEST(JSON_Test, GEMM_Key) {
  const int m = 500;
  const int n = 200;
  const int k = 900;
  for (auto &transA : {"N","T"}) {
    for (auto &transB : {"N","T"}) {
       nlohmann::json key_json;
       key_json["transA"] = transA;
       key_json["transB"] = transB;
       key_json["m"] = m;
       key_json["n"] = n;
       key_json["k"] = k;

       GEMM_Key key = from_json<GEMM_Key>(key_json);
       nlohmann::json test_json = to_json(key);

       ASSERT_EQ(test_json, key_json);
    }
  }
  for (auto &transA : {"N","T"}) {
    for (auto &transB : {"N","T"}) {
      GEMM_Key key(
          BLAS_Operation(transA), 
          BLAS_Operation(transB), 
          m, k, n);
      nlohmann::json json = to_json(key);
      GEMM_Key test_key = from_json<GEMM_Key>(json);

      ASSERT_TRUE(!(test_key < key) && !(key < test_key));
    }
  }
  
}

TEST(JSON_Test, GEMM_Options) {
  for (auto &transA : {"N","T"}) {
    for (auto &transB : {"N","T"}) {
      for (auto &transC : {"N","T"}) {
        for (auto &padA : {"N","P"}) {
          for (auto &padB : {"N","P"}) {
            for (auto &padC : {"N","P"}) {
              nlohmann::json opts_json;
              opts_json["transA"] = transA;
              opts_json["transB"] = transB;
              opts_json["transC"] = transC;
              opts_json["padA"] = padA;
              opts_json["padB"] = padB;
              opts_json["padC"] = padC;

              GEMM_Options opts = from_json<GEMM_Options>(opts_json);
              nlohmann::json test_json = to_json(opts);

              ASSERT_EQ(test_json, opts_json);
            }
          }
        }
      }
    }
  }
  for (auto &transA : {"N","T"}) {
    for (auto &transB : {"N","T"}) {
      for (auto &transC : {"N","T"}) {
        for (auto &padA : {"N","P"}) {
          for (auto &padB : {"N","P"}) {
            for (auto &padC : {"N","P"}) {
              GEMM_Options opts(
                  (BLAS_Op(transA)), Pad_Op(padA), 
                  (BLAS_Op(transB)), Pad_Op(padB), 
                  (BLAS_Op(transC)), Pad_Op(padC));
              nlohmann::json json = to_json(opts);
              GEMM_Options test_opts = from_json<GEMM_Options>(json);

              ASSERT_TRUE(!(test_opts < opts) && !(opts < test_opts));
            }
          }
        }
      }
    }
  }
}
