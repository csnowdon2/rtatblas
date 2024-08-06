#pragma once
#if defined(_RTAT_CUDA)
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#elif defined(_RTAT_HIP)
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hiprand/hiprand.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop  
#endif
#endif
#include <memory>
#include <iostream>
#include <map>

namespace rtat {

namespace gpu {
#if defined (_RTAT_CUDA)
#define _RTAT_GPU(x) cuda##x
#define _RTAT_GPU_BLAS(x) cublas##x
#define _RTAT_GPU_ENUM(x) CU##x
#elif defined(_RTAT_HIP)
#define _RTAT_GPU(x) hip##x
#define _RTAT_GPU_BLAS(x) hipblas##x
#define _RTAT_GPU_ENUM(x) HIP##x
#else 
  static_assert(false, "Compiler must define either _RTAT_CUDA or _RTAT_HIP");
#endif
  constexpr auto Success = _RTAT_GPU(Success);
  constexpr auto SetDevice = _RTAT_GPU(SetDevice);
  constexpr auto GetDevice = _RTAT_GPU(GetDevice);
  constexpr auto GetDeviceCount = _RTAT_GPU(GetDeviceCount);
  constexpr auto DeviceSynchronize = _RTAT_GPU(DeviceSynchronize);
  constexpr auto MemGetInfo = _RTAT_GPU(MemGetInfo);

  using Error_t = _RTAT_GPU(Error_t);
  constexpr auto GetErrorString = _RTAT_GPU(GetErrorString);
  constexpr auto ErrorInvalidResourceHandle = _RTAT_GPU(ErrorInvalidResourceHandle);
  constexpr auto Free = _RTAT_GPU(Free);

  // The constexpr auto trick seems not to work with Malloc
  template<typename T>
  constexpr Error_t Malloc(T** ptr, size_t size) {
    return _RTAT_GPU(Malloc)(ptr,size);
  }

  using  Stream_t = _RTAT_GPU(Stream_t);
  constexpr auto StreamCreate = _RTAT_GPU(StreamCreate);
  constexpr auto StreamDestroy =  _RTAT_GPU(StreamDestroy);
  constexpr auto StreamSynchronize = _RTAT_GPU(StreamSynchronize);
  constexpr auto StreamWaitEvent = _RTAT_GPU(StreamWaitEvent);

  using Event_t = _RTAT_GPU(Event_t);
  constexpr auto EventCreate = _RTAT_GPU(EventCreate);
  constexpr auto EventDestroy = _RTAT_GPU(EventDestroy);
  constexpr auto EventRecord = _RTAT_GPU(EventRecord);
  constexpr auto EventSynchronize = _RTAT_GPU(EventSynchronize);
  constexpr auto EventElapsedTime = _RTAT_GPU(EventElapsedTime);
  constexpr auto EventQuery = _RTAT_GPU(EventQuery);

  constexpr auto Memcpy = _RTAT_GPU(Memcpy);
  constexpr auto Memset = _RTAT_GPU(Memset);
  constexpr auto MemcpyAsync = _RTAT_GPU(MemcpyAsync);
  constexpr auto MemcpyDeviceToDevice = _RTAT_GPU(MemcpyDeviceToDevice);
  constexpr auto MemcpyDeviceToHost = _RTAT_GPU(MemcpyDeviceToHost);
  constexpr auto MemcpyHostToDevice = _RTAT_GPU(MemcpyHostToDevice);

  using blasHandle_t = _RTAT_GPU_BLAS(Handle_t);
  using blasOperation_t = _RTAT_GPU_BLAS(Operation_t);
  using blasStatus_t = _RTAT_GPU_BLAS(Status_t);
  constexpr auto BLAS_STATUS_SUCCESS = _RTAT_GPU_ENUM(BLAS_STATUS_SUCCESS);
  constexpr auto blasCreate = _RTAT_GPU_BLAS(Create);
  constexpr auto blasDestroy = _RTAT_GPU_BLAS(Destroy);
  constexpr auto blasDgeam = _RTAT_GPU_BLAS(Dgeam);
  constexpr auto blasDgemm = _RTAT_GPU_BLAS(Dgemm);
  constexpr auto blasDtrsm = _RTAT_GPU_BLAS(Dtrsm);
  constexpr auto blasDsyrk = _RTAT_GPU_BLAS(Dsyrk);
  constexpr auto blasSgeam = _RTAT_GPU_BLAS(Sgeam);
  constexpr auto blasSgemm = _RTAT_GPU_BLAS(Sgemm);
  constexpr auto blasStrsm = _RTAT_GPU_BLAS(Strsm);
  constexpr auto blasSsyrk = _RTAT_GPU_BLAS(Ssyrk);
  constexpr auto blasGetStream = _RTAT_GPU_BLAS(GetStream);
  constexpr auto blasSetStream = _RTAT_GPU_BLAS(SetStream);
  using blasSideMode_t = _RTAT_GPU_BLAS(SideMode_t);
  using blasDiagType_t = _RTAT_GPU_BLAS(DiagType_t);
  using blasFillMode_t = _RTAT_GPU_BLAS(FillMode_t);
  constexpr auto BLAS_SIDE_LEFT = _RTAT_GPU_ENUM(BLAS_SIDE_LEFT);
  constexpr auto BLAS_SIDE_RIGHT = _RTAT_GPU_ENUM(BLAS_SIDE_RIGHT);
  constexpr auto BLAS_DIAG_NON_UNIT = _RTAT_GPU_ENUM(BLAS_DIAG_NON_UNIT);
  constexpr auto BLAS_DIAG_UNIT = _RTAT_GPU_ENUM(BLAS_DIAG_UNIT);
  constexpr auto BLAS_FILL_MODE_LOWER = _RTAT_GPU_ENUM(BLAS_FILL_MODE_LOWER);
  constexpr auto BLAS_FILL_MODE_UPPER = _RTAT_GPU_ENUM(BLAS_FILL_MODE_UPPER);
  constexpr auto BLAS_OP_N = _RTAT_GPU_ENUM(BLAS_OP_N);
  constexpr auto BLAS_OP_T = _RTAT_GPU_ENUM(BLAS_OP_T);

  using randGenerator_t = hiprandGenerator_t;
  constexpr auto randSetStream = _RTAT_GPU(randSetStream);
  constexpr auto randCreateGenerator = _RTAT_GPU(randCreateGenerator);
  constexpr auto randDestroyGenerator = _RTAT_GPU(randDestroyGenerator);
  constexpr auto randGenerateUniformDouble = _RTAT_GPU(randGenerateUniformDouble);
  constexpr auto randGenerateUniform = _RTAT_GPU(randGenerateUniform);
  constexpr auto RAND_RNG_PSEUDO_DEFAULT = _RTAT_GPU_ENUM(RAND_RNG_PSEUDO_DEFAULT);
#undef _RTAT_GPU
#undef _RTAT_GPU_BLAS
#undef _RTAT_GPU_ENUM
}

#define gpuAssert(ans)                          \
  {                                             \
    gpu_error_check((ans), __FILE__, __LINE__); \
  }

inline void gpu_error_check(gpu::Error_t code, const char* file, int line)
{
  if (code != gpu::Success)
    std::cerr << "GPU Error: " << gpu::GetErrorString(code) 
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
  operator gpu::Stream_t();
protected:
  Raw_Stream() {}
  gpu::Stream_t stream;
};


class Stream {
public:
  Stream();
  Stream(gpu::Stream_t stream);

  Stream(const Stream& other);
  Stream& operator=(const Stream& other);

  operator gpu::Stream_t();

  void wait_event(Event e);
  void synchronize();
private:
  std::shared_ptr<Raw_Stream> raw_stream;
};


class Raw_Event {
public:
  friend class Event;
  virtual ~Raw_Event() = default;
  operator gpu::Event_t();
protected:
  Raw_Event() {}
  gpu::Event_t event;
};

class Event {
public:
  Event();
  Event(gpu::Event_t event);

  Event(const Event& other);
  Event& operator=(const Event& other);

  operator gpu::Event_t(); 

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
  gpu::randGenerator_t rng;
};

class Device_RNG {
public:
  Device_RNG();
  Device_RNG(Stream s) : Device_RNG() { set_stream(s); }

  Device_RNG(const Device_RNG& other);
  Device_RNG& operator=(const Device_RNG& other);

  operator gpu::randGenerator_t();

  void set_stream(Stream s) {
    gpu::randSetStream(raw_rng->rng, s);
  }

  template<typename T, typename IGNORE = void>
  void uniform(T*, size_t);

  template<typename IGNORE>
  void uniform(double *A, size_t len) {
    gpu::randGenerateUniformDouble(raw_rng->rng, A, len);
  }

  template<typename IGNORE>
  void uniform(float *A, size_t len) {
    gpu::randGenerateUniform(raw_rng->rng, A, len);
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
  static std::map<gpu::blasOperation_t, std::string> map() {
    return {{gpu::BLAS_OP_N, "N"}, 
            {gpu::BLAS_OP_T, "T"}};
  }
};
using BLAS_Operation = String_Rep<BLAS_Operation_Str_Map>;

struct BLAS_Fill_Mode_Str_Map {
  static std::map<gpu::blasFillMode_t, std::string> map() {
    return {{gpu::BLAS_FILL_MODE_LOWER, "Lower"}, 
            {gpu::BLAS_FILL_MODE_UPPER, "Upper"}};
  }
};
using BLAS_Fill_Mode = String_Rep<BLAS_Fill_Mode_Str_Map>;

struct BLAS_Side_Str_Map {
  static std::map<gpu::blasSideMode_t, std::string> map() {
    return {{gpu::BLAS_SIDE_LEFT,  "Left"},
            {gpu::BLAS_SIDE_RIGHT, "Right"}};
  }
};
using BLAS_Side = String_Rep<BLAS_Side_Str_Map>;

struct BLAS_Diag_Str_Map {
  static std::map<gpu::blasDiagType_t, std::string> map() {
    return {{gpu::BLAS_DIAG_UNIT,     "Unit"},
            {gpu::BLAS_DIAG_NON_UNIT, "Non-Unit"}};
  }
};
using BLAS_Diag = String_Rep<BLAS_Diag_Str_Map>;

}
