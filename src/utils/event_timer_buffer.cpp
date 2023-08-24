#include "event_timer_buffer.h"

Change_Device::Change_Device(int device) : current_device(device) {
  cudaGetDevice(&old_device);
  if (old_device != current_device) cudaSetDevice(current_device);
}

Change_Device::~Change_Device() {
  if (old_device != current_device) cudaSetDevice(old_device);
}

Event_Timer_Buffer::Event_Timer_Buffer(int device_id) : device_id(device_id) {}

void Event_Timer_Buffer::add_interval(Event e1, Event e2) {
  Change_Device dev(device_id);

  float ms;
  if (cudaEventElapsedTime(&ms, e1, e2) == cudaErrorInvalidResourceHandle) return;

  timer_queue.emplace(e1, e2);
}

bool Event_Timer_Buffer::extract_time(float &ms) {
  if (count() == 0) return false;
  Change_Device dev(device_id);

  auto events = timer_queue.front();
  ms = Event::elapsed_time(events.first, events.second);

  if (ms != NAN) timer_queue.pop();

  return (ms != NAN);
}

size_t Event_Timer_Buffer::count() {
  return timer_queue.size();
}

