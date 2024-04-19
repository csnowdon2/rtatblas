#include <iostream>
#include <gtest/gtest.h>
#include <timer_bank.h>
#include <device_timer.h>

using namespace rtat;

TEST(Device_Timer_Test, Time) {
  int interval = 50;
  Stream s;
  for (int i=0; i<3; i++) {
    Device_Timer timer([&]([[maybe_unused]] Stream s) {
      std::this_thread::sleep_for(std::chrono::milliseconds(interval));
    },s);

    ASSERT_NEAR(timer.wait_for_time(), interval, 1);
  }
}

TEST(Timer_Bank_Test, Times) {
  int interval = 76;
  Timer_Bank timers;
  Stream s;
  for (int i=0; i<3; i++) {
    Device_Timer timer([&]([[maybe_unused]] Stream s) {
      std::this_thread::sleep_for(std::chrono::milliseconds(interval));
    },s);
    timers.append(timer);
  }

  ASSERT_EQ(timers.size(), 3);
  ASSERT_EQ(timers.completed(), 3);
  std::cout << "hi" << std::endl;
  
  auto times = timers.get_times();
  for (auto &t : times) 
    ASSERT_NEAR(t, interval, 1);
}
