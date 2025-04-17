#include "common.hpp"
#include "test_helper.hpp"

int host_gemv_int32(uint32_t m, uint32_t n, const int *mat, const int *vec, int *y, int alpha, int beta) {
  for (size_t row = 0; row < m; ++row) {
    int mul_res = 0;
    for (size_t col = 0; col < n; ++col) {
      mul_res += vec[col] * mat[row * n + col];
    }
    y[row] = alpha * mul_res + y[row] * beta;
  }

  return 0;
}

int main(int argc, char **argv) {
  const int M = 1541;
  const int N = 1317;
  auto mat = generateRandomIntegers(M * N, -1000, 1000);
  auto vec = generateRandomIntegers(N, -1000, 1000);
  auto y = generateRandomIntegers(M, 1, 10);
  auto y_host = pimblas::vector<int>(y.begin(), y.end());
  int alpha = 1;
  int beta = 0;

  int ret = 0;
  if ((ret = gemv_int32(M, N, mat.data(), vec.data(), y.data(), &alpha, &beta)) != 0) {
    RET_TEST_FAIL;
  }

  if ((ret = host_gemv_int32(M, N, mat.data(), vec.data(), y_host.data(), alpha, beta)) != 0) {
    RET_TEST_FAIL;
  }

  if (false == same(y.data(), y_host.data(), y.size())) {
    RET_TEST_FAIL;
  }

  std::cout << "SUCCESS " << std::endl;
  RET_TEST_OK;
}