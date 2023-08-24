#include "gpu-api.h"
#include <cstdlib>
#include <cublas_v2.h>
#include <queue>
#include <vector>
#include <map>

struct Dgemm_Params {
  int m,k,n;

  bool operator <(const Dgemm_Params &rhs) {
    if (this->m < rhs.m) return true;
    if (this->m > rhs.m) return false;
    if (this->k < rhs.k) return true;
    if (this->k > rhs.k) return false;
    if (this->n < rhs.n) return true;
    if (this->n > rhs.n) return false;
    return false;
  }
};

enum Dgemm_Types {
  NN, NT, TN, TT
};

// Need to separate params into m,k,n and ops separately. 
// Have to record data for different ops within the record.
class Dgemm_Record {
private:
  const int m,k,n;
  const size_t flop_count;

  std::queue<TimeSig> times;
  bool converged;

public:

};

class Workspace {
  char *buffer;
  size_t size;
};

// TIME SIGNATURES
class TimeSig {
  Event start;
  Event end;
};

// END TIME SIGNATURES

static std::vector<Workspace> workspaces;

// Should return ERR type specific to implementation?
cublasStatus_t rtatblasDgemm (cublasHandle_t handle,
                              cublasOperation_t transa, cublasOperation_t transb,
                              int m, int n, int k,
                              const double *alpha,
                              const double *A, int lda,
                              const double *B, int ldb,
                              const double *beta,
                              double       *C, int ldc) 
{
  cudaStream_t raw_stream;
  cublasGetStream(handle, &raw_stream);
  Stream stream(raw_stream);

  TimeSig times;

  // Predict execution plan
  // If converged, (serialize) and measure
  // Execute plan

  // Plan -> a description of the execution flow. Comes with a method to execute.
  //         Accepts workspace as a parameter. Provides workspace reqs.
  // Where do the workspaces live?
  // Planning_System -> generates plans and analyzes results to generate better plans.
  //                    Maintained directly by runtime. 
  // Runtime -> Planning system
  //            Optional workspace maintenance
  //            Fall back when not enough workspace available
  //            Should be agnostic of device, handle that at lower level (workspace, timing)


  times.start.record(stream);
  auto err = cublasDgemm(handle, transa, transb, m, n, k, alpha,
                         A, lda, B, ldb, beta, C, ldc);
  times.end.record(stream);

  return err;
}
