#include "common.hpp"
#include "test_helper.hpp"

int host_gemv_int8(uint32_t m, uint32_t n, const int8_t *mat, const int8_t *vec, int *y, int alpha, int beta) {
  for (size_t row = 0; row < m; ++row) {
    int mul_res = 0;
    for (size_t col = 0; col < n; ++col) {
      mul_res += static_cast<int>(vec[col]) * static_cast<int>(mat[row * n + col]);
    }
    y[row] = alpha * mul_res + y[row] * beta;
  }

  return 0;
}

int main(int argc, char **argv) {
  const int M = 1331;
  const int N = 1427;
  auto mat =
      generateRandomIntegral<int8_t>(M * N, std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
  auto vec = generateRandomIntegral<int8_t>(N, std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
  auto y = generateRandomIntegers(M, -100, 100);
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
  if (!same) {
    std::cout << "fail\n";
    RET_TEST_FAIL;
  }

  std::cout << "SUCCESS " << std::endl;
  RET_TEST_OK;
}
