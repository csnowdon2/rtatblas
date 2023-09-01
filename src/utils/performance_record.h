#include "event_timer_buffer.h"
#include <iostream>
#include "rolling_average.h"
#include <vector>

class Performance_Record {
public:
  Performance_Record(bool synchronous = false);
  
  template<typename Func>
  void measure(Func f, Stream s) {
    int dev;
    cudaGetDevice(&dev);
    Event e1, e2;

    if (synchronous) cudaDeviceSynchronize();
    e1.record(s);
    f(s);
    e2.record(s);
    //if (synchronous) cudaDeviceSynchronize();

    event_timers[dev].add_interval(e1,e2);
  }

  void flush() {
    float ms;
    for (auto &timer : event_timers)
      while (timer.extract_time(ms)) {
        avg.add_value(ms);
      }
  }

  size_t count() {
    size_t n = avg.count();
    for (auto &timer : event_timers)
      n += timer.count();
    return n;
  }

  float get_time() {
    flush();
    return avg.get_average();
  }

  float get_std() {
    flush();
    return avg.get_std();
  }

  void print() {avg.print();}

  bool synchronous;
private:
  std::vector<Event_Timer_Buffer> event_timers;
  Detailed_Average avg;
};
