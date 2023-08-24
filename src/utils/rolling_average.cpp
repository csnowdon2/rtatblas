#include "rolling_average.h"


void Rolling_Average::add_value(double value) {
  total += value; 
  count++;
}

double Rolling_Average::get_average() {
  return total/count;
}

void Rolling_Average::reset() {
  total = 0;
  count = 0;
}

