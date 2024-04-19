#include "event_timer_buffer.h"

namespace rtat {

Change_Device::Change_Device(int device) : current_device(device) {
  gpuAssert(cudaGetDevice(&old_device));
  if (old_device != current_device) gpuAssert(cudaSetDevice(current_device));
}

Change_Device::~Change_Device() {
  if (old_device != current_device) gpuAssert(cudaSetDevice(old_device));
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

  if (!std::isnan(ms)) timer_queue.pop();

  return (!std::isnan(ms));
}

size_t Event_Timer_Buffer::count() {
  return timer_queue.size();
}

}
