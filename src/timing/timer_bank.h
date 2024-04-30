#pragma once
#include <queue>

#include "device_timer.h"

namespace rtat {

class Timer_Bank {
  std::queue<Device_Timer> timers;
  std::vector<float> times;

  void update();
public:
  void append(Device_Timer &timer);
  const std::vector<float>& get_times();
  void synchronize();

  size_t size();
  size_t completed();
};

}
