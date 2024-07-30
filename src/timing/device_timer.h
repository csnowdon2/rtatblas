#pragma once
#include <gpu-api.h>
#include <optional>

namespace rtat {


class Device_Timer {
public:
  enum Mode {
    ASYNCHRONOUS = 0,
    SEMI_SYNCHRONOUS = 1,
    SYNCHRONOUS = 2
  };

public:
  template<typename Func>
  Device_Timer(Func f, Stream s, Mode mode = ASYNCHRONOUS) :
      start(std::make_unique<Event>()),
      end(std::make_unique<Event>())
  {
    if (mode == SEMI_SYNCHRONOUS || mode == SYNCHRONOUS)
      gpuAssert(cudaDeviceSynchronize());

    start->record(s);
    f(s);
    end->record(s);

    if (mode == SYNCHRONOUS)
      end->synchronize();
  }

  std::optional<float> query_time();
  float time();

private:
  std::unique_ptr<Event> start, end;
  float t = -1.0;
};

}
