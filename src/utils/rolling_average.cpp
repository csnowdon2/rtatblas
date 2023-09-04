#include "rolling_average.h"
#include <numeric>


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


Detailed_Average::Detailed_Average() : Detailed_Average(100) {}
Detailed_Average::Detailed_Average(size_t limit) : limit(limit), vals() {}

void Detailed_Average::add_value(double value) {
  if (vals.size() < limit) {
    vals.push_back(value);
  } else {
    vals[place] = value;
    place = (place + 1) % limit;
  }
}


double Detailed_Average::get_average() {
  return std::accumulate(vals.begin(), vals.end(), 0.0)/vals.size();
}

double Detailed_Average::get_std() {
  double avg = get_average();
  double std = 0.0;
  for (auto &x : vals)
    std += (x-avg)*(x-avg);
  return std::sqrt(std/vals.size());
}

void Detailed_Average::reset() {
  place = 0;
  vals.resize(0);
}

size_t Detailed_Average::count() {
  return vals.size();
}
