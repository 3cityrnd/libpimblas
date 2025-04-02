#include <chrono>

#include "common.hpp"
#include "test_helper.hpp"

int host_sgemm_column_major(const float *A, const float *B, float *C, float alpha, float beta, uint32_t M, uint32_t N,
                            uint32_t K) {
  // Loop over columns of C
  for (size_t col = 0; col < N; col++) {
    // Loop over rows of C
    for (size_t row = 0; row < M; row++) {
      float sum = 0.0f;
      for (size_t i = 0; i < K; i++) {
        sum += A[row + i * M] * B[i + col * K];
      }
      C[row + col * M] = alpha * sum + beta * C[row + col * M];
    }
  }
  return 0;
}

int host_sgemm_row_major(const float *A, const float *B, float *C, float alpha, float beta, uint32_t M, uint32_t N,
                         uint32_t K) {
  // Loop over rows of C
  for (size_t row = 0; row < M; row++) {
    // Loop over columns of C
    for (size_t col = 0; col < N; col++) {
      float sum = 0.0f;
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

bool test_sgemm_wrapper() {
  const int M = 1024;
  const int N = 1024;
  const int K = 1024;
  // All matricies are in col major order
  auto A = generateRandomFloats(M * K, 1.0f, 10.0f);
  auto B = generateRandomFloats(K * N, 1.0f, 10.0f);
  auto C = generateRandomFloats(M * N, 1.0f, 10.0f);
  auto C_host = pimblas::vector<float>(C.begin(), C.end());
  float alpha = 1.0f;
  float beta = 1.0f;

  char transa = 'n';
  char transb = 'n';
  sgemm_wrapper(&transa, &transb, &M, &N, &K, &alpha, A.data(), &M, B.data(), &K, &beta, C.data(), &M);

  host_sgemm_column_major(A.data(), B.data(), C_host.data(), alpha, beta, M, N, K);

  // 0.01 percent difference at most
  return mostly_same_rel(C.data(), C_host.data(), M * N, 1e-4f);
}

bool test_gemm_row_maj_f() {
  const int M = 1111;
  const int N = 143;
  const int K = 13;
  auto A = generateRandomFloats(M * K, 1.0f, 10.0f);
  auto B = generateRandomFloats(K * N, 1.0f, 10.0f);
  auto C = generateRandomFloats(M * N, 1.0f, 10.0f);
  auto C_host = pimblas::vector<float>(C.begin(), C.end());
  float alpha = 1.0f;
  float beta = 0.0f;

  gemm_row_maj_f(&M, &N, &K, &alpha, A.data(), B.data(), &beta, C.data());
  host_sgemm_row_major(A.data(), B.data(), C_host.data(), alpha, beta, M, N, K);

  if (false == mostly_same_rel(C.data(), C_host.data(), M * N, 1e-4f)) {
    return false;
  }
  return true;
}

int main(int argc, char **argv) {
  if (false == test_sgemm_wrapper()) {
    RET_TEST_FAIL;
  }
  if (false == test_gemm_row_maj_f()) {
    RET_TEST_FAIL;
  }

  std::cout << "SUCCESS" << std::endl;
  RET_TEST_OK;
}