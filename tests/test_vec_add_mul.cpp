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

void host_add(float *input_a, float *input_b, float *output, size_t size) {
  for (size_t i = 0; i < size; i++) {
    output[i] = input_a[i] + input_b[i];
  }
}

void host_mul(float *input_a, float *input_b, float *output, size_t size) {
  for (size_t i = 0; i < size; i++) {
    output[i] = input_a[i] * input_b[i];
  }
}

int single_test(size_t size, int num_dpus) {
  auto sample_data_a = create_sample_data(size);
  auto sample_data_b = create_sample_data(size);

  float *result = new float[size];
  vec_add_f(sample_data_a.data(), sample_data_b.data(), result, size, num_dpus);

  float *result_host = new float[size];
  host_add(sample_data_a.data(), sample_data_b.data(), result_host, size);

  for (size_t i = 0; i < size; i++) {
    if (abs(result[i] - result_host[i]) > 0.01) {
      return -1;
    }
  }
  return 0;
}

int main(int argc, char **argv) {
  if (single_test(10000, 64) != 0) {
    printf("TEST FAILED\n");
    RET_TEST_FAIL;
  }
  printf("TEST OK\n");
  RET_TEST_OK;
}
