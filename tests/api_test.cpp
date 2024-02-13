#include <gtest/gtest.h>
#include <iostream>
#include "gpu-api.h"

using namespace rtat;

TEST(API_Test, Creation) {
  Stream s;
  Event e;
}

TEST(API_Test, Elapsed_Time) {
  Event e1, e2;
  Stream nos;
  double *x,*y;
  gpuAssert(cudaMalloc(&x, 512));
  gpuAssert(cudaMalloc(&y, 512));

  e1.record(nos);
  gpuAssert(cudaMemcpyAsync(y, x, 512, cudaMemcpyDeviceToDevice, nos));
  e2.record(nos);
  gpuAssert(cudaDeviceSynchronize());
  std::cout << "Elapsed time " << Event::elapsed_time(e1,e2) << std::endl;

  SUCCEED();
}
