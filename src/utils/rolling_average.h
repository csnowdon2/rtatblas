#pragma once
#include <gpu-api.h>
#include <iostream>
#include <vector>

class Rolling_Average {
public:
  void add_value(double value);
  double get_average() const;

  void reset();
  
private:
  double total = 0;
  size_t count = 0;
};

class Detailed_Average {
public:
  Detailed_Average();
  Detailed_Average(size_t limit);

  void add_value(double value);
  double get_average() const;
  double get_std();
  size_t count();

  void reset();
  void print() {
    for (auto &x : vals) std::cout << x << " ";
    std::cout << std::endl;
  }

  const std::vector<double> &data() const {return vals;}
  
private:
  size_t limit;
  size_t place = 0;
  std::vector<double> vals;
};
