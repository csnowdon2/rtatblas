#pragma once
#include <gpu-api.h>
#include <optional>

namespace rtat {

class Device_Timer {
  std::unique_ptr<Event> start, end;
  float t = -1.0;

public:
  template<typename Func>
  Device_Timer(Func f, Stream s) :
      start(std::make_unique<Event>()),
      end(std::make_unique<Event>())
  {
    start->record(s);
    f(s);
    end->record(s);
  }

  std::optional<float> query_time();
  float time();
};

}
