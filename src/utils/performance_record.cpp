#include "performance_record.h"
#include <iostream>

Performance_Record::Performance_Record(bool synchronous) : synchronous(synchronous) {
  int devs;
  cudaGetDeviceCount(&devs);
  for (int i = 0; i < devs; i++) {
    event_timers.emplace_back(i);
  }
}

