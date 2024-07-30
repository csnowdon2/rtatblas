#include "gtest/gtest.h"
#include <workspace.h>
#include <gtest/gtest.h>
#include <vector>

using namespace rtat;

TEST(Workspace_Test, Default_Constructor) {
  Workspace();
}

TEST(Workspace_Test, Pointer_Constructor) {
  size_t N = 2048;
  std::vector<char> bytes(N);

  Workspace space(bytes.data(), N);
}

TEST(Workspace_Test, Size) {
  size_t N = 8192;
  std::vector<char> bytes(N);
  Workspace space(bytes.data(), N);

  ASSERT_EQ(space.size<char>(), N);
  ASSERT_EQ(space.size<double>(), N/sizeof(double));
  ASSERT_EQ(space.size<int>(), N/sizeof(int));
  ASSERT_EQ(space.size<float>(), N/sizeof(float));
}

TEST(Workspace_Test, Offset_Constructor) {
  size_t N = 1024;
  std::vector<char> bytes(N);
  Workspace space(bytes.data(), N);

  { // Normal case
    size_t offset = 301;
    size_t size = 123;
    Workspace offset_space(space, offset, size);

    // Correct pointer
    ASSERT_EQ((char*)offset_space, &(((char*)space)[offset]));

    // Correct size
    ASSERT_EQ(offset_space.size<char>(), size);
  }

  { // Death case 
    ASSERT_DEATH(Workspace(space,N-2,10),".*");
  }
}

TEST(Workspace_Test, Peel) {
  size_t N = 1024;
  std::vector<char> bytes(N);
  Workspace space(bytes.data(), N);

  size_t s1 = 18;
  size_t s2 = 27;
  size_t s3 = 48;

  Workspace peel1 = space.peel<double>(s1);
  Workspace peel2 = space.peel<float>(s2);
  Workspace peel3 = space.peel<char>(s3);

  size_t offset = peel1.size<char>() 
                + peel2.size<char>() 
                + peel3.size<char>();

  ASSERT_EQ(peel1.size<char>(), s1*sizeof(double));
  ASSERT_EQ(((char*)peel1), &bytes[0]);

  ASSERT_EQ(peel2.size<char>(), s2*sizeof(float));
  ASSERT_EQ(((char*)peel2), &bytes[s1*sizeof(double)]);

  ASSERT_EQ(peel3.size<char>(), s3*sizeof(char));
  ASSERT_EQ(((char*)peel3), &bytes[s1*sizeof(double)+s2*sizeof(float)]);

  ASSERT_EQ(space.size<char>(), N-offset);
  ASSERT_EQ(((char*)space), &bytes[offset]);
}
