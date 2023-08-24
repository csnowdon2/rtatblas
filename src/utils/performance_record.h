#include "event_timer_buffer.h"
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

  float get_time() {
    float ms;
    for (auto &timer : event_timers)
      while (timer.extract_time(ms))
        avg.add_value(ms);

    return avg.get_average();
  }

private:
  bool synchronous;
  std::vector<Event_Timer_Buffer> event_timers;
  Rolling_Average avg;
};
