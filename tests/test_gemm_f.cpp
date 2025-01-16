#include "common.hpp"
#include "test_helper.hpp"

int host_gemm_f(uint32_t M, uint32_t N, uint32_t K, const float *A, const float *B, float *C, float alpha, float beta) {
  pimblas::vector<float> temp(M * K, 0.0f);

  // temp = (A x B)
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < K; j++) {
      for (size_t k = 0; k < N; k++) {
        C[i * K + j] += A[i * N + k] * B[k * K + j];
      }
    }
  }

  // C = alpha * temp + beta * C
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < K; j++) {
      C[i * K + j] = alpha * temp[i * K + j] + beta * C[i * K + j];
    }
  }

  return 0;
}

template <typename T>
void printMatrix(uint32_t rows, uint32_t cols, const T *data) {
  for (auto i = 0; i < rows; i++) {
    for (auto j = 0; j < cols; j++) {
      std::cout << data[i * cols + j] << " ";
    }
    std::cout << std::endl;
  }
}

int main(int argc, char **argv) {
  const int M = 131;
  const int N = 23;
  const int K = 19;
  auto A = generateRandomFloats(M * N, 1.0f, 10.0f);
  auto B = generateRandomFloats(N * K, 1.0f, 10.0f);
  auto C = generateRandomFloats(M * K, 1.0f, 10.0f);
  auto C_host = pimblas::vector<float>(C.begin(), C.end());
  float alpha = 1.0f;
  float beta = 1.0f;

  sgemm_wrapper(nullptr, nullptr, &M, &N, &K, &alpha, A.data(), nullptr, B.data(), nullptr, &beta, C.data(), nullptr);

  host_gemm_f(M, N, K, A.data(), B.data(), C_host.data(), alpha, beta);

  // 0.01 percent difference at most
  bool same = mostly_same(C.data(), C_host.data(), M * K, 1e-4f);
  if (same) {
    std::cout << "SUCCESS " << std::endl;
    RET_TEST_OK;
  }

  RET_TEST_FAIL;
}