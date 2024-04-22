#include "timer_bank.h"

namespace rtat {

void Timer_Bank::update() {
  if (timers.empty())
    return;

  while (auto t = timers.front().query_time()) {
    times.push_back(*t);
    timers.pop();
    if (timers.empty()) break;
  }
}

void Timer_Bank::append(Device_Timer &timer) {
  timers.push(std::move(timer));
}

size_t Timer_Bank::size() {
  return timers.size() + times.size();
}

size_t Timer_Bank::completed() {
  update();
  return times.size();
}

const std::vector<float>& Timer_Bank::get_times() {
  update();
  return times;
}

}
