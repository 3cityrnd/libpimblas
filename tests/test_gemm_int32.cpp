#include <chrono>

#include "common.hpp"
#include "test_helper.hpp"

int host_gemm_row_major_int32(const int32_t *A, const int32_t *B, int32_t *C, int alpha, int beta, uint32_t M,
                              uint32_t N, uint32_t K) {
  // Loop over rows of C
  for (size_t row = 0; row < M; row++) {
    // Loop over columns of C
    for (size_t col = 0; col < N; col++) {
      int sum = 0;
      // Compute the dot product of the row of A and the column of B
      for (size_t i = 0; i < K; i++) {
        sum += A[row * K + i] * B[i * N + col];
      }
      // Update C with alpha * AB + beta * C
      C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
  }
  return 0;
}

bool test_gemm_row_maj_int32(int M, int N, int K, int alpha, int beta) {
  auto A =
      generateRandomIntegral<int32_t>(M * K, std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
  auto B =
      generateRandomIntegral<int32_t>(K * N, std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
  auto C = generateRandomIntegral<int32_t>(M * N, -1000, 1000);
  auto C_host = pimblas::vector<int32_t>(C.begin(), C.end());

  gemm_row_maj_int32(&M, &N, &K, &alpha, A.data(), B.data(), &beta, C.data());
  host_gemm_row_major_int32(A.data(), B.data(), C_host.data(), alpha, beta, M, N, K);

  if (false == same(C.data(), C_host.data(), C.size())) {
    std::cout << "FAIL\n";
    return false;
  }
  return true;
}

int main(int argc, char **argv) {
  const int M = 1341;
  const int N = 101;
  const int K = 1473;
  const int alpha = 1;
  const int beta = 0;

  if (false == test_gemm_row_maj_int32(M, N, K, alpha, beta)) {
    RET_TEST_FAIL;
  }

  std::cout << "SUCCESS" << std::endl;
  RET_TEST_OK;
}