#pragma once
#include <gpu-api.h>

class Rolling_Average {
public:
  void add_value(double value);
  double get_average();

  void reset();
  
private:
  double total = 0;
  size_t count = 0;
};
