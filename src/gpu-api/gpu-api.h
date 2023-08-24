#pragma once
#include <cuda_runtime.h>
#include <memory>
#include <math.h>

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
