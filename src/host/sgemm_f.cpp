#include <cassert>
#include <chrono>

#include "dpu_transfer_helper.hpp"
#include "gemvf_kernel.hpp"
#include "matrix_transpose.hpp"

template <typename Kernel>
Kernel &get_free_kernel(std::vector<Kernel> &kernels, size_t &cur_kernel) {
  if (cur_kernel >= kernels.size()) {
    cur_kernel = 0;
  }
  auto &kernel = kernels[cur_kernel];
  cur_kernel++;
  return kernel;
}

// Assumption A is in row order, B and C are in column order
// A is of size rowsA x rowsB
// B rowsB x colsB
// C rowsA x colsB
void sgemm_f(uint32_t rowsA, uint32_t rowsB, uint32_t colsB, const float *A, const float *B, float *C,
             const float *alpha, const float *beta) {
  uint32_t nr_dpus = 512;
  uint32_t rows_per_dpu = 0;
  gemv_launch_statistics<float>(rowsA, rowsB, nr_dpus, rows_per_dpu);

  auto nr_kernels = colsB;

  if (*beta == 0.0f) {
    std::vector<GEMVF_Kernel> kernels(nr_kernels);
    size_t kernel_it = 0;
    for (kernel_it = 0; kernel_it < kernels.size(); kernel_it++) {
      auto &kernel = kernels[kernel_it];
      if (kernel.init(rowsA, rowsB, nr_dpus, rows_per_dpu) == false) {
        break;
      }

      kernel.set_params(alpha, false);
      kernel.set_A(A, true);
    }
    kernels.resize(kernel_it);

    show_trace("Running {} kernels. Each kernel with {} DPUs.\n", kernels.size(), nr_dpus);

    size_t cur_kernel = 0;
    for (uint32_t i = 0; i < colsB; i++) {
      auto &kernel = get_free_kernel(kernels, cur_kernel);
      kernel.set_x(B + rowsB * i, true);
      kernel.launch(true);
      kernel.get_y(C + rowsA * i, true);
    }

    for (auto &kernel : kernels) {
      kernel.sync();
      kernel.free_dpus();
    }
  } else {
    std::vector<GEMVF_Kernel_Beta> kernels(nr_kernels);
    size_t kernel_it = 0;
    for (kernel_it = 0; kernel_it < kernels.size(); kernel_it++) {
      auto &kernel = kernels[kernel_it];
      if (kernel.init(rowsA, rowsB, nr_dpus, rows_per_dpu) == false) {
        break;
      }

      kernel.set_params(alpha, beta, false);
      kernel.set_A(A, true);
    }
    kernels.resize(kernel_it);

    show_trace("Running {} kernels. Each kernel with {} DPUs.\n", kernels.size(), nr_dpus);

    size_t cur_kernel = 0;
    for (uint32_t i = 0; i < colsB; i++) {
      auto &kernel = get_free_kernel(kernels, cur_kernel);
      kernel.set_x(B + rowsB * i, true);
      kernel.set_y(C + rowsA * i, true);
      kernel.launch(true);
      kernel.get_y(C + rowsA * i, true);
    }

    for (auto &kernel : kernels) {
      kernel.sync();
      kernel.free_dpus();
    }
  }
}

bool is_transpose(char trans) {
  if (trans == 'N' || trans == 'n') {
    return false;
  } else if (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c') {
    return true;
  }
  // Unknown trans
  return false;
}

// Performs: C = alpha * op(A) * op(B) + beta * C
// a, b, c are always in column major order
// trans - if 'n' op(a) = A, if 't' or 'c' op (A) = A**T
// ld - leading dimension as declared in the calling program.
// op(A) is an m by k matrix
// op(B) is a k by n matrix
// C is an m by n matrix
void sgemm_wrapper(const char *transa, const char *transb, const int *m, const int *n, const int *k, const float *alpha,
                   const float *a, const int *lda, const float *b, const int *ldb, const float *beta, float *c,
                   const int *ldc) {
  const float *a_buffer = nullptr;
  float *a_tmp_buffer = nullptr;

  if (false == is_transpose(*transa)) {
    // Matrix is in colum major order
    // And is treated as is that means
    // we need to change it into row major in order to use
    // it with our algorithm.
    assert(*lda == *m && "Unexpected padding in matrix A - UNHANDLED");
    a_tmp_buffer = reinterpret_cast<float *>(malloc(alignUp(*m * *k * sizeof(float), 16)));
    transpose_matrix_column_major(a, a_tmp_buffer, *m, *k);
    a_buffer = a_tmp_buffer;
  } else {
    // Matrix is in column major order
    // And is treated as transposed
    // that means we can treat it as is already
    // assuming there's no padding
    // Just switch the dimensions
    assert(*lda == *k && "Unexpected padding in matrix A - UNHANDLED");
    a_buffer = a;
  }

  const float *b_buffer = nullptr;
  float *b_tmp_buffer = nullptr;
  if (false == is_transpose(*transb)) {
    // No need to do anything
    assert(*ldb == *k && "Unexpected padding in matrix B - UNHANDLED");
    b_buffer = b;
  } else {
    // Matrix is in col major ordering and is treated as transposed
    // That means if we convert it then it's all good
    // we got B (LDB/n, k) and we need it to be B**T (k, n)
    assert(*ldb == *n && "Unexpected padding in matrix B - UNHANDLED");
    b_tmp_buffer = reinterpret_cast<float *>(malloc(alignUp(*n * *k * sizeof(float), 16)));
    transpose_matrix_column_major(b, b_tmp_buffer, *n, *k);
    b_buffer = b_tmp_buffer;
  }

  // C is already in column major order no need to do anything
  assert(*ldc == *m && "Unexpected padding in matrix C - UNHANDLED");

  sgemm_f(*m, *k, *n, a_buffer, b_buffer, c, alpha, beta);

  free(a_tmp_buffer);
  free(b_tmp_buffer);
}

/*
A is an m by k matrix
B is an k by n matrix
C is an m by n matrix
All matricies are in row major format. Memory is continuous
*/
void gemm_row_maj_f(const int *m, const int *n, const int *k, const float *alpha, const float *a, const float *b,
                    const float *beta, float *c) {
  show_trace(
      "gemm_row_maj_f m=[{}] n=[{}] k=[{}] alpha=[{}] a=[{:#018x}] b=[{:#018x}] "
      "beta=[{}] c=[{:#018x}]",
      *m, *n, *k, *alpha, reinterpret_cast<const uintptr_t>(a), reinterpret_cast<const uintptr_t>(b), *beta,
      reinterpret_cast<const uintptr_t>(c));

  // Get B to column major format
  float *tmp_b = reinterpret_cast<float *>(malloc(alignUp(*k * *n * sizeof(float), 16)));
  transpose_matrix_row_major(b, tmp_b, *k, *n);

  // If Beta is not zero we need to change C to column major format
  float *tmp_c = reinterpret_cast<float *>(malloc(alignUp(*m * *n * sizeof(float), 16)));
  if (*beta != 0.0f) {
    transpose_matrix_row_major(c, tmp_c, *m, *n);
  }

  sgemm_f(*m, *k, *n, a, tmp_b, tmp_c, alpha, beta);

  // Get C from col major to row major
  transpose_matrix_column_major(tmp_c, c, *m, *n);

  free(tmp_b);
  free(tmp_c);
}