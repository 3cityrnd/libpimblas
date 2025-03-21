#include <iomanip>
#include <limits>

#include "common.hpp"
#include "test_helper.hpp"

std::vector<float> create_sample_data(size_t size) {
  std::vector<float> buffer(size, 0);
  for (size_t i = 0; i < size; i++) {
    buffer[i] = (float)i - ((float)size / 2);
  }
  return buffer;
}

void host_relu(float *input, float *output, size_t size) {
  for (size_t i = 0; i < size; i++) {
    output[i] = input[i] > 0 ? input[i] : 0;
  }
}

int main(int argc, char **argv) {
  auto sample_data = create_sample_data(512000000);
  float *result = new float[sample_data.size()];
  relu_f(sample_data.data(), result, sample_data.size());

  float *result_host = new float[sample_data.size()];
  host_relu(sample_data.data(), result_host, sample_data.size());
  for (size_t i = 0; i < sample_data.size(); i++) {
    if (abs(result[i] - result_host[i]) > 0.01) {
      printf("TEST FAILED\n");
      RET_TEST_FAIL;
    }
  }
  printf("TEST OK\n");
  RET_TEST_OK;
}
