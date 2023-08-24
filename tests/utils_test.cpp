#include <iostream>
#include <thread>
#include <chrono>
#include <gtest/gtest.h>
#include <numeric>
#include <rolling_average.h>
#include <event_timer_buffer.h>
#include <vector>

TEST(Rolling_Average_Test, Calculation) {
  std::vector<double> xs = {2.4, 14.235146, 137.724, -12.15, -162.15};
  std::vector<double> ys = {14.235146, 137.724, -12.15, -162.15};
  double xmean = std::reduce(xs.begin(), xs.end())/xs.size();
  double ymean = std::reduce(ys.begin(), ys.end())/ys.size();
  
  Rolling_Average avg;
  for (auto &x : xs) avg.add_value(x);
  ASSERT_DOUBLE_EQ(avg.get_average(), xmean);

  avg.reset();
  for (auto &y : ys) avg.add_value(y);
  ASSERT_DOUBLE_EQ(avg.get_average(), ymean);

  SUCCEED();
}

TEST(Event_Timer_Test, Three_Intervals) {
  int interval = 50;
  Event_Timer_Buffer timer_buffer(0);
  for (int i=0; i<3; i++) {
    Stream s;
    Event e1, e2;
    e1.record(s);
    std::this_thread::sleep_for(std::chrono::milliseconds(interval));
    e2.record(s);
    timer_buffer.add_interval(e1,e2);
  }

  for (int i=0; i<3; i++) {
    float ms;
    timer_buffer.extract_time(ms);
    EXPECT_NEAR(ms, (float)interval, 1.0);
  }
}

TEST(Event_Timer_Test, Multi_GPU) {
  int ndevices;
  cudaGetDeviceCount(&ndevices);
  int interval = 15;

  std::vector<Event_Timer_Buffer> timer_buffers;
  for (int i = 0; i < ndevices; i++) 
    timer_buffers.emplace_back(i);

  int reps = 3;
  for (int i = 0; i < reps; i++) {
    for (int idev = 0; idev < ndevices; idev++) {
      Change_Device dev(idev);

      Stream s;
      Event e1,e2;
      e1.record(s);
      std::this_thread::sleep_for(std::chrono::milliseconds(interval));
      e2.record(s);
      timer_buffers[idev].add_interval(e1,e2);
    }
  }

  int results = 0;
  for (int idev = 0; idev < ndevices; idev++) {
    float ms;
    while (timer_buffers[idev].extract_time(ms)) {
      results++;
      EXPECT_NEAR(ms, (float)interval, 1.0);
    }
  }
  ASSERT_EQ(results, reps*ndevices);
}
