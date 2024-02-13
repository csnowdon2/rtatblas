#include <gpu-api.h>
#include <queue>

namespace rtat {

class Change_Device {
public:
  Change_Device(int device);
  ~Change_Device();

  Change_Device(const Change_Device &other) = delete;
  Change_Device(Change_Device &&other) = delete;
private:
  int old_device;
  int current_device;
};

class Event_Timer_Buffer {
public:
  Event_Timer_Buffer(int device_id);

  void add_interval(Event e1, Event e2);
  bool extract_time(float &ms);
  size_t count();
private:
  int device_id;
  std::queue<std::pair<Event, Event>> timer_queue;
};
}
