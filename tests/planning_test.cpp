#include <gtest/gtest.h>
#include <planning_system.h>
#include "common.h"

class Planning_Test : public BLAS_Test {};

class Dummy_Op : public MatrixOp {
public:
  Dummy_Op() : MatrixOp({}) {}
  Matrix execute(cublasHandle_t, Workspace, Workspace) override {
    return Matrix();
  }

  size_t output_space_req()  const override {return 0;}
  MatrixDims dims() const override {return MatrixDims();}
};


struct Dummy_Params {
  cublasHandle_t handle;
  int i;
  Dummy_Params(cublasHandle_t handle, int i) 
    : handle(handle), i(i) {}
};


struct Dummy_Key {
  int i;
  Dummy_Key(Dummy_Params p) : i(p.i) {}
  bool operator<(const Dummy_Key& o) const {return i < o.i;}
};


struct Dummy_Opts {
  int i;
  Dummy_Opts(int i) : i(i) {}
  Dummy_Opts() : i(0) {}

  operator std::string() const {return std::to_string(i);}
  friend std::ostream& operator<<(std::ostream& os, 
                                  const Dummy_Opts opts) {
    os << opts.i;
    return os;
  }

  static std::vector<Dummy_Opts> enumerate() {
    return {Dummy_Opts(1), Dummy_Opts(2), Dummy_Opts(3)};
  }

  bool operator<(const Dummy_Opts& o) const {return i < o.i;}
  // friend std::istream& operator>>(std::istream&, GEMM_Options&); 

  static Dummy_Opts default_opts() {return Dummy_Opts();}

  std::unique_ptr<MatrixOp> form_operation(Dummy_Params) {return std::make_unique<Dummy_Op>();}
};


class Dummy_Executor : public Executor<Dummy_Params, Dummy_Key, Dummy_Opts> {
  void warmup(Dummy_Params, Dummy_Opts, Stream) override {};
};


TEST_F(Planning_Test, Dummy_Planner) {
  Planning_System<Dummy_Executor> planner;  

  const int N = 4;
  for (int i=0; i<N; i++) {
    Dummy_Params params(handle, i);
    for (auto opts : Dummy_Opts::enumerate()) {
      for (int j=0; j<opts.i; j++)
        planner.execute(params, opts, Workspace(), s);
    }
  }

  auto stats = planner.make_statistics();
  EXPECT_EQ(stats.get_counts().size(), N);
  for (auto &[key, opt_map] : stats.get_counts()) {
    EXPECT_EQ(opt_map.size(), Dummy_Opts::enumerate().size());
    for (auto &[opt, count] : opt_map)
      EXPECT_EQ(opt.i, count);
  }
}

TEST_F(Planning_Test, GEMM_Correctness) {
  GEMM_Planner planner;

  int m = 23;
  int n = 16;
  int k = 35;

  TestMatrix A(m,k,m);
  TestMatrix B(k,n,k);
  TestMatrix C(m,n,m);

  double alpha = 1.0;

  GEMM_Inputs inputs(handle, CUBLAS_OP_N, CUBLAS_OP_N, A, B, C, alpha, 0.0);

  for (int i=0; i<10; i++) {
    for (auto &plan : GEMM_Options::enumerate()) {
      size_t ws = planner.calculate_workspace(inputs, plan);
      ManagedWorkspace space(ws);

      planner.execute(inputs, plan, space, s);

      C.download();
      test_gemm(A, B, C, -alpha, 1.0, false, false);

      EXPECT_TRUE(C.is_zero());
    }
  }
}

TEST_F(Planning_Test, Hello) {
  // This isn't really testing anything?
  GEMM_Planner planner;

  size_t m = 423;
  size_t n = 125;
  size_t k = 318;

  TestMatrix A(m,k,m);
  TestMatrix B(k,n,k);
  TestMatrix C(m,n,m);

  double alpha = 1.0;

  GEMM_Inputs inputs(handle, CUBLAS_OP_N, CUBLAS_OP_N, A, B, C, alpha, 0.0);

  ManagedWorkspace space(1024);
  for (int j=0; j<2; j++) {
    size_t reps = 100;
    for (size_t i=0; i<reps; i++) {
      GEMM_Options plan = planner.create_plan(inputs);

      size_t req = planner.calculate_workspace(inputs, plan)*sizeof(double);
      space.grow_to_fit(req);

      planner.execute(inputs, plan, space, s);
    }
    gpuAssert(cudaDeviceSynchronize());
    //planner.dump_analytics();
  }
}

// Check that every plan can run without workspace
TEST_F(Planning_Test, Plan_Degradation) {
  GEMM_Planner planner;

  size_t m = 69;
  size_t n = 123;
  size_t k = 42;

  TestMatrix A(m,k,m);
  TestMatrix B(k,n,k);
  TestMatrix C(m,n,m);

  double alpha = 1.0;
  GEMM_Inputs inputs(handle, CUBLAS_OP_N, CUBLAS_OP_N, A, B, C, alpha, 0.0);

  for (auto &plan : GEMM_Options::enumerate()) {
    planner.execute(inputs, plan, Workspace(), s);

    C.download();
    test_gemm(A, B, C, -alpha, 1.0, false, false);

    ASSERT_TRUE(C.is_zero());
  }
}
