#include "gpu-api.h"

namespace rtat {

class Owning_RNG : public Raw_Device_RNG {
public:
  Owning_RNG() { curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT); }
  ~Owning_RNG() { curandDestroyGenerator(rng); }
};

Device_RNG::Device_RNG() : raw_rng(std::make_shared<Owning_RNG>()) {}
Device_RNG::Device_RNG(const Device_RNG& other) : raw_rng(other.raw_rng) {}

Device_RNG& Device_RNG::operator=(const Device_RNG& other) { raw_rng = other.raw_rng; return *this; }

Device_RNG::operator curandGenerator_t() { return raw_rng->rng; }


class Non_Owning_Stream : public Raw_Stream {
public:
  Non_Owning_Stream(cudaStream_t stream_) { stream = stream_; }
  ~Non_Owning_Stream() = default;
};

class Owning_Stream : public Raw_Stream {
public:
  Owning_Stream() { gpuAssert(cudaStreamCreate(&stream)); }
  ~Owning_Stream() { gpuAssert(cudaStreamDestroy(stream)); }
};

class Non_Owning_Event : public Raw_Event {
public:
  Non_Owning_Event(cudaEvent_t event_) { event = event_; }
  ~Non_Owning_Event() = default;
};

class Owning_Event : public Raw_Event {
public:
  Owning_Event() { gpuAssert(cudaEventCreate(&event)); }
  ~Owning_Event() { gpuAssert(cudaEventDestroy(event)); }
};


Stream::Stream() : raw_stream(std::make_shared<Owning_Stream>()) {}
Stream::Stream(cudaStream_t stream) : raw_stream(std::make_shared<Non_Owning_Stream>(stream)) {}

Stream::Stream(const Stream& other) : raw_stream(other.raw_stream) {}
Stream& Stream::operator=(const Stream& other)  { raw_stream = other.raw_stream; return *this;}

Stream::operator cudaStream_t() { return raw_stream->stream; }

void Stream::wait_event(Event e) { gpuAssert(cudaStreamWaitEvent(*this, e, 0)); }
void Stream::synchronize() { gpuAssert(cudaStreamSynchronize(*this)); }



Event::Event() : raw_event(std::make_shared<Owning_Event>()) {}
Event::Event(cudaEvent_t event) : raw_event(std::make_shared<Non_Owning_Event>(event)) {}

Event::Event(const Event& other) : raw_event(other.raw_event) {}
Event& Event::operator=(const Event& other) { raw_event = other.raw_event; return *this;}

Event::operator cudaEvent_t() { return raw_event->event; }

void Event::record(Stream s) { gpuAssert(cudaEventRecord(*this, s)); }
void Event::synchronize() { gpuAssert(cudaEventSynchronize(*this)); }
bool Event::query() { return cudaEventQuery(*this) == cudaSuccess; }

float Event::elapsed_time(Event start, Event end) {
  float ms;
  auto err = cudaEventElapsedTime(&ms, start, end);
  if (err != cudaSuccess)
    ms = NAN;
  return ms;
}
}
