#include "device_timer.h"

namespace rtat {

std::optional<float> Device_Timer::query_time() {
  if (!start->query() || !end->query())
    return {};

  if (t < 0.0) {
    t = Event::elapsed_time(*start,*end);
    start.reset();
    end.reset();
  }

  return t;
}

float Device_Timer::time() {
  start->synchronize();
  end->synchronize();
  
  return *query_time();
}


}
