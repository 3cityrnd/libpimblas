#include "common.hpp"
#include "test_helper.hpp"

int host_gemv_int8(uint32_t m, uint32_t n, const int8_t *mat, const int8_t *vec, int *y, int alpha, int beta) {
  for (int i = 0; i < m; ++i) {
    int dot = 0;
    for (int j = 0; j < n; ++j) {
      dot += vec[j] * mat[i * n + j];
    }
    y[i] = alpha * dot + beta * y[i];
  }
  return 0;
}

int main(int argc, char **argv) {
  const int M = 1024;
  const int N = 15000;
  auto mat = generateRandomIntegers<int8_t>(M * N, 1, 10);
  auto vec = generateRandomIntegers<int8_t>(N, 1, 10);
  auto y = generateRandomIntegers(M, 1, 10);
  auto y_host = pimblas::vector<int>(y.begin(), y.end());
  int alpha = 1;
  int beta = 0;

  int ret = 0;
  if ((ret = gemv_int8(M, N, mat.data(), vec.data(), y.data(), &alpha, &beta)) != 0) {
    RET_TEST_FAIL;
  }

  if ((ret = host_gemv_int8(M, N, mat.data(), vec.data(), y_host.data(), alpha, beta)) != 0) {
    RET_TEST_FAIL;
  }

  bool same = same_vectors(y, y_host);
  if (same) {
    std::cout << "SUCCESS " << std::endl;
    RET_TEST_OK;
  }

  RET_TEST_FAIL;
}
