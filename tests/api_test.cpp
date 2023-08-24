#include <gtest/gtest.h>
#include <iostream>
#include "gpu-api.h"

TEST(API_Test, Creation) {
  Stream s;
  Event e;
}

TEST(API_Test, Elapsed_Time) {
  cudaStream_t s;
  Event e1, e2;
  double *x,*y;
  cudaMalloc(&x, 512);
  cudaMalloc(&y, 512);
  {
    Stream nos;
    s = nos;

    e1.record(nos);
    if (cudaMemcpyAsync(y, x, 512, cudaMemcpyDeviceToDevice, nos)) std::cout << "fuc" << std::endl;
    e2.record(nos);
    cudaDeviceSynchronize();
    std::cout << "Elapsed time " << Event::elapsed_time(e1,e2) << std::endl;
  }
  if (cudaEventRecord(e1, s)) std::cout << "fuc" << std::endl;
  if (cudaMemcpyAsync(y, x, 512, cudaMemcpyDeviceToDevice, s)) std::cout << "fuc" << std::endl;
  if (cudaEventRecord(e2, s)) std::cout << "fuc" << std::endl;
  cudaDeviceSynchronize();
  std::cout << "Elapsed time " << Event::elapsed_time(e1,e2) << std::endl;
  SUCCEED();
}
