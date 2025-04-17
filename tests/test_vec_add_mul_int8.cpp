#include <iomanip>
#include <limits>

#include "common.hpp"
#include "test_helper.hpp"

std::vector<int8_t> create_sample_data(size_t size) {
  std::vector<int8_t> buffer(size, 0);
  for (size_t i = 0; i < size; i++) {
    buffer[i] = (int8_t)i - ((int8_t)size / 2);
  }
  return buffer;
}

void host_add(int8_t *input_a, int8_t *input_b, int8_t *output, size_t size) {
  for (size_t i = 0; i < size; i++) {
    output[i] = input_a[i] + input_b[i];
  }
}

void host_mul(int8_t *input_a, int8_t *input_b, int8_t *output, size_t size) {
  for (size_t i = 0; i < size; i++) {
    output[i] = input_a[i] * input_b[i];
  }
}

void host_sub(int8_t *input_a, int8_t *input_b, int8_t *output, size_t size) {
  for (size_t i = 0; i < size; i++) {
    output[i] = input_a[i] - input_b[i];
  }
}

int single_test_add(size_t size) {
  auto sample_data_a = create_sample_data(size);
  auto sample_data_b = create_sample_data(size);
  for (size_t i = 0; i < size; i++) {
    printf("sample_data_a[%zu] = %d, sample_data_b[%zu] = %d\n", i, sample_data_a[i], i, sample_data_b[i]);
  }
  int8_t *result = new int8_t[size];
  vec_add_int8(sample_data_a.data(), sample_data_b.data(), result, size);

  int8_t *result_host = new int8_t[size];
  host_add(sample_data_a.data(), sample_data_b.data(), result_host, size);

  for (size_t i = 0; i < size; i++) {
    if (result[i] != result_host[i]) {
      printf("result[%zu] = %d, result_host[%zu] = %d\n", i, result[i], i, result_host[i]);
      return -1;
    }
  }
  return 0;
}

int single_test_mul(size_t size) {
  auto sample_data_a = create_sample_data(size);
  auto sample_data_b = create_sample_data(size);

  int8_t *result = new int8_t[size];
  vec_mul_int8(sample_data_a.data(), sample_data_b.data(), result, size);

  int8_t *result_host = new int8_t[size];
  host_mul(sample_data_a.data(), sample_data_b.data(), result_host, size);

  for (size_t i = 0; i < size; i++) {
    if (abs(result[i] - result_host[i]) > 0.01) {
      return -1;
    }
  }
  return 0;
}

int single_test_sub(size_t size) {
  auto sample_data_a = create_sample_data(size);
  auto sample_data_b = create_sample_data(size);

  int8_t *result = new int8_t[size];
  vec_sub_int8(sample_data_a.data(), sample_data_b.data(), result, size);

  int8_t *result_host = new int8_t[size];
  host_sub(sample_data_a.data(), sample_data_b.data(), result_host, size);

  for (size_t i = 0; i < size; i++) {
    if (abs(result[i] - result_host[i]) > 0.01) {
      return -1;
    }
  }
  return 0;
}

int main(int argc, char **argv) {
  for (int i = 0; i < 10; i++) {
    if (single_test_add(100 + i) != 0) {
      printf("ADD TEST FAILED\n");
      RET_TEST_FAIL;
    }
  }
  for (int i = 0; i < 10; i++) {
    if (single_test_mul(100 + i) != 0) {
      printf("MUL TEST FAILED\n");
      RET_TEST_FAIL;
    }
  }
  for (int i = 0; i < 10; i++) {
    if (single_test_sub(100 + i) != 0) {
      printf("SUB TEST FAILED\n");
      RET_TEST_FAIL;
    }
  }
  printf("TEST OK\n");
  RET_TEST_OK;
}
