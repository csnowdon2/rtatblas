#pragma once
#ifdef CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#else
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#endif
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hiprand/hiprand.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#define cudaMemGetInfo hipMemGetInfo
#define cudaEventCreate hipEventCreate
#define cudaEventDestroy hipEventDestroy
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventQuery hipEventQuery
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEvent_t hipEvent_t
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStream_t hipStream_t
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaSuccess hipSuccess
#define cudaSetDevice hipSetDevice
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaErrorInvalidResourceHandle hipErrorInvalidResourceHandle
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemset hipMemset
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define cublasDgeam hipblasDgeam
#define cublasDgemm hipblasDgemm
#define cublasDtrsm hipblasDtrsm
#define cublasDsyrk hipblasDsyrk
#define cublasDgemmBatched hipblasDgemmBatched
#define cublasSgeam hipblasSgeam
#define cublasSgemm hipblasSgemm
#define cublasStrsm hipblasStrsm
#define cublasSsyrk hipblasSsyrk
#define cublasSgemmBatched hipblasSgemmBatched
#define cublasGetStream hipblasGetStream
#define cublasSetStream hipblasSetStream
#define cublasHandle_t hipblasHandle_t
#define cublasOperation_t hipblasOperation_t
#define cublasStatus_t hipblasStatus_t
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString 
#define cudaError_t hipError_t 
#define curandGenerator_t hiprandGenerator_t
#define curandSetStream hiprandSetStream
#define curandCreateGenerator hiprandCreateGenerator
#define curandDestroyGenerator hiprandDestroyGenerator
#define curandGenerateUniformDouble hiprandGenerateUniformDouble
#define curandGenerateUniform hiprandGenerateUniform
#define CURAND_RNG_PSEUDO_DEFAULT HIPRAND_RNG_PSEUDO_DEFAULT
#define cublasSideMode_t hipblasSideMode_t
#define CUBLAS_SIDE_LEFT HIPBLAS_SIDE_LEFT
#define CUBLAS_SIDE_RIGHT HIPBLAS_SIDE_RIGHT
#define cublasFillMode_t hipblasFillMode_t
#define CUBLAS_DIAG_NON_UNIT HIPBLAS_DIAG_NON_UNIT
#define CUBLAS_DIAG_UNIT HIPBLAS_DIAG_UNIT
#define cublasDiagType_t hipblasDiagType_t
#define CUBLAS_FILL_MODE_LOWER HIPBLAS_FILL_MODE_LOWER
#define CUBLAS_FILL_MODE_UPPER HIPBLAS_FILL_MODE_UPPER
#endif
#include <memory>
#include <iostream>
#include <map>

namespace rtat {

#define gpuAssert(ans)                          \
  {                                             \
    gpu_error_check((ans), __FILE__, __LINE__); \
  }

inline void gpu_error_check(cudaError_t code, const char* file, int line)
{
  if (code != cudaSuccess)
    std::cerr << "GPU Error: " << cudaGetErrorString(code) 
              << " " << file << " " << line << std::endl;
}

// Stream and Event wrappers, intended to mimic the semantics of 
// the native API types but with automatic resource management.
class Stream;
class Event;


class Raw_Stream {
public:
  friend class Stream;
  virtual ~Raw_Stream() = default;
  operator cudaStream_t();
protected:
  Raw_Stream() {}
  cudaStream_t stream;
};


class Stream {
public:
  Stream();
  Stream(cudaStream_t stream);

  Stream(const Stream& other);
  Stream& operator=(const Stream& other);

  operator cudaStream_t();

  void wait_event(Event e);
  void synchronize();
private:
  std::shared_ptr<Raw_Stream> raw_stream;
};


class Raw_Event {
public:
  friend class Event;
  virtual ~Raw_Event() = default;
  operator cudaEvent_t();
protected:
  Raw_Event() {}
  cudaEvent_t event;
};

class Event {
public:
  Event();
  Event(cudaEvent_t event);

  Event(const Event& other);
  Event& operator=(const Event& other);

  operator cudaEvent_t(); 

  void record(Stream s);
  void synchronize();
  bool query();

  static float elapsed_time(Event start, Event end);
private:
  std::shared_ptr<Raw_Event> raw_event;
};

class Raw_Device_RNG {
public:
  friend class Device_RNG;
  virtual ~Raw_Device_RNG() = default;
protected:
  Raw_Device_RNG() = default;
  curandGenerator_t rng;
};

class Device_RNG {
public:
  Device_RNG();
  Device_RNG(Stream s) : Device_RNG() { set_stream(s); }

  Device_RNG(const Device_RNG& other);
  Device_RNG& operator=(const Device_RNG& other);

  operator curandGenerator_t();

  void set_stream(Stream s) {
    curandSetStream(raw_rng->rng, s);
  }

  template<typename T, typename IGNORE = void>
  void uniform(T*, size_t);

  template<typename IGNORE>
  void uniform(double *A, size_t len) {
    curandGenerateUniformDouble(raw_rng->rng, A, len);
  }

  template<typename IGNORE>
  void uniform(float *A, size_t len) {
    curandGenerateUniform(raw_rng->rng, A, len);
  }

private:
  std::shared_ptr<Raw_Device_RNG> raw_rng;
};




template<class str_map>
class String_Rep {
  using T = typename decltype(str_map::map())::key_type;
  T val;
public:
  String_Rep(T val) : val(val) {}
  operator T() const {return val;}

  String_Rep(std::string str) {
    for (auto &[k,v] : str_map::map()) {
      if (v == str) {
        val = k;
        return;
      }
    }
    throw std::runtime_error("Invalid string " + str + " passed to string rep");
  }

  operator std::string() const {
    auto map = str_map::map();
    if (auto search = map.find(val); search != map.end()) {
      return search->second;
    }
    throw std::runtime_error("Invalid string rep value");
  }

  String_Rep operator!() const {
    if (str_map::map().size() != 2) {
      throw std::runtime_error(
          "operator! applied to non-binary String_Rep");
    }

    for (auto &[k,v] : str_map::map()) {
      if (k != val) {
        return String_Rep(k);
      }
    }
    __builtin_unreachable();
  }

  bool operator==(T o) const {return val == o;}

  friend std::ostream& operator<<(std::ostream& os, 
      const String_Rep& r) {
    os << std::string(r);
    return os;
  }
};

struct BLAS_Operation_Str_Map {
  static std::map<cublasOperation_t, std::string> map() {
    return {{CUBLAS_OP_N, "N"}, 
            {CUBLAS_OP_T, "T"}};
  }
};
using BLAS_Operation = String_Rep<BLAS_Operation_Str_Map>;

struct BLAS_Fill_Mode_Str_Map {
  static std::map<cublasFillMode_t, std::string> map() {
    return {{CUBLAS_FILL_MODE_LOWER, "Lower"}, 
            {CUBLAS_FILL_MODE_UPPER, "Upper"}};
  }
};
using BLAS_Fill_Mode = String_Rep<BLAS_Fill_Mode_Str_Map>;

struct BLAS_Side_Str_Map {
  static std::map<cublasSideMode_t, std::string> map() {
    return {{CUBLAS_SIDE_LEFT,  "Left"},
            {CUBLAS_SIDE_RIGHT, "Right"}};
  }
};
using BLAS_Side = String_Rep<BLAS_Side_Str_Map>;

struct BLAS_Diag_Str_Map {
  static std::map<cublasDiagType_t, std::string> map() {
    return {{CUBLAS_DIAG_UNIT,     "Unit"},
            {CUBLAS_DIAG_NON_UNIT, "Non-Unit"}};
  }
};
using BLAS_Diag = String_Rep<BLAS_Diag_Str_Map>;

}
