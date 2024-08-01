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
  gpuAssert(gpu::Malloc(&x, 512));
  gpuAssert(gpu::Malloc(&y, 512));

  e1.record(nos);
  gpuAssert(gpu::MemcpyAsync(y, x, 512, gpu::MemcpyDeviceToDevice, nos));
  e2.record(nos);
  gpuAssert(gpu::DeviceSynchronize());
  std::cout << "Elapsed time " << Event::elapsed_time(e1,e2) << std::endl;

  SUCCEED();
}
