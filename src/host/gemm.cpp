#include <cassert>
#include <chrono>
#include <iomanip>
#include <type_traits>

#include "dpu_transfer_helper.hpp"
#include "gemv_kernel.hpp"
#include "matrix_transpose.hpp"

template <typename T>
void print_matrix_row_major(const T *mat, size_t rows, size_t cols) {
  static_assert(std::is_integral<T>::value, "This function only works with integer types");

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      // For small types (like int8_t), cast to int for proper numeric display
      if constexpr (sizeof(T) < sizeof(int)) {
        std::cout << std::setw(6) << static_cast<int>(mat[i * cols + j]) << " ";
      } else {
        std::cout << std::setw(6) << mat[i * cols + j] << " ";
      }
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

template <typename T>
void print_matrix_col_major(const T *mat, size_t rows, size_t cols) {
  static_assert(std::is_integral<T>::value, "This function only works with integer types");

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      // For small types (like int8_t), cast to int for proper numeric display
      if constexpr (sizeof(T) < sizeof(int)) {
        std::cout << std::setw(6) << static_cast<int>(mat[j * rows + i]) << " ";
      } else {
        std::cout << std::setw(6) << mat[j * rows + i] << " ";
      }
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

template <class Kernel>
struct SCS {  // Single Column Solver
  Kernel kernel;
  int column = -1;
};

template <class Kernel>
class MCS {  // Multi Column Solver
  using SCSVecT = std::vector<SCS<Kernel>>;
  using iter = typename SCSVecT::iterator;

 public:
  MCS(size_t nr_solvers) : solvers(nr_solvers) { it = solvers.begin(); }

  SCSVecT &get_solvers() { return solvers; }

  SCS<Kernel> &get_free_kernel() {
    while (true) {
      if (it->kernel.get_status().done) {
        return *it;
      }
      if (std::next(it) == solvers.end()) {
        it = solvers.begin();
      } else {
        it++;
      }
    }
  }

 private:
  SCSVecT solvers;
  iter it;
};

// Assumption A is in row order, B and C are in column order
// A is of size rowsA x rowsB
// B rowsB x colsB
// C rowsA x colsB
void sgemm_int32(uint32_t rowsA, uint32_t rowsB, uint32_t colsB, const int32_t *A, const int32_t *B, int32_t *C,
                 const int *alpha, const int *beta) {
  uint32_t nr_dpus = 512;
  uint32_t rows_per_dpu = 0;
  gemv_launch_statistics<int32_t>(rowsA, rowsB, nr_dpus, rows_per_dpu);

  auto nr_solvers = std::min(8 * 8 * 2 * 20 / nr_dpus, (colsB + 1) / 2);

  bool has_beta = (*beta != 0);
  MCS<GEMV_INT32_Kernel> mcs(nr_solvers);

  auto &solvers = mcs.get_solvers();
  size_t kernel_it = 0;
  for (kernel_it = 0; kernel_it < nr_solvers; kernel_it++) {
    auto &kernel = solvers[kernel_it].kernel;
    if (kernel.init(rowsA, rowsB, nr_dpus, rows_per_dpu) == false) {
      break;
    }
  }
  solvers.resize(kernel_it);

  for (auto &scs : solvers) {
    auto &kernel = scs.kernel;
    kernel.set_params(alpha, beta, false);
    kernel.set_A(A, true);
  }

  show_trace("Running {} kernels. Each kernel with {} DPUs.\n", solvers.size(), nr_dpus);

  size_t cur_kernel = 0;
  for (uint32_t i = 0; i < colsB; i++) {
    auto &scs = mcs.get_free_kernel();
    auto &kernel = scs.kernel;
    if (scs.column != -1) {
      kernel.get_y_safe(C + rowsA * scs.column);
      scs.column = -1;
    }
    kernel.set_x(B + rowsB * i, true);
    if (has_beta) {
      kernel.set_y(C + rowsA * i, true);
    }
    kernel.launch(true);
    scs.column = i;
  }

  for (auto &scs : solvers) {
    auto &kernel = scs.kernel;
    if (scs.column != -1) {
      kernel.sync();
      kernel.get_y_safe(C + rowsA * scs.column);
      scs.column = -1;
    }

    kernel.free_dpus();
  }
}

// Assumption A is in row order, B and C are in column order
// A is of size rowsA x rowsB
// B rowsB x colsB
// C rowsA x colsB
void sgemm_int8(uint32_t rowsA, uint32_t rowsB, uint32_t colsB, const int8_t *A, const int8_t *B, int32_t *C,
                const int *alpha, const int *beta) {
  uint32_t nr_dpus = 512;
  uint32_t rows_per_dpu = 0;
  gemv_launch_statistics<int8_t>(rowsA, rowsB, nr_dpus, rows_per_dpu);

  auto nr_solvers = std::min(8 * 8 * 2 * 20 / nr_dpus, (colsB + 1) / 2);

  bool has_beta = (*beta != 0);
  MCS<GEMV_INT8_Kernel> mcs(nr_solvers);

  auto &solvers = mcs.get_solvers();
  size_t kernel_it = 0;
  for (kernel_it = 0; kernel_it < nr_solvers; kernel_it++) {
    auto &kernel = solvers[kernel_it].kernel;
    if (kernel.init(rowsA, rowsB, nr_dpus, rows_per_dpu) == false) {
      break;
    }
  }
  solvers.resize(kernel_it);

  for (auto &scs : solvers) {
    auto &kernel = scs.kernel;
    kernel.set_params(alpha, beta, false);
    kernel.set_A(A, true);
  }

  show_trace("Running {} kernels. Each kernel with {} DPUs.\n", solvers.size(), nr_dpus);

  size_t cur_kernel = 0;
  for (uint32_t i = 0; i < colsB; i++) {
    auto &scs = mcs.get_free_kernel();
    auto &kernel = scs.kernel;
    if (scs.column != -1) {
      kernel.get_y_safe(C + rowsA * scs.column);
      scs.column = -1;
    }
    kernel.set_x(B + rowsB * i, true);
    if (has_beta) {
      kernel.set_y(C + rowsA * i, true);
    }
    kernel.launch(true);
    scs.column = i;
  }

  for (auto &scs : solvers) {
    auto &kernel = scs.kernel;
    if (scs.column != -1) {
      kernel.sync();
      kernel.get_y_safe(C + rowsA * scs.column);
      scs.column = -1;
    }

    kernel.free_dpus();
  }
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

  auto nr_solvers = std::min(8 * 8 * 2 * 20 / nr_dpus, (colsB + 1) / 2);

  if (*beta == 0.0f) {
    MCS<GEMVF_Kernel> mcs(nr_solvers);

    auto &solvers = mcs.get_solvers();
    size_t kernel_it = 0;
    for (kernel_it = 0; kernel_it < nr_solvers; kernel_it++) {
      auto &kernel = solvers[kernel_it].kernel;
      if (kernel.init(rowsA, rowsB, nr_dpus, rows_per_dpu) == false) {
        break;
      }
    }
    solvers.resize(kernel_it);

    for (auto &scs : solvers) {
      auto &kernel = scs.kernel;
      kernel.set_params(alpha, false);
      kernel.set_A(A, true);
    }

    show_trace("Running {} kernels. Each kernel with {} DPUs.\n", solvers.size(), nr_dpus);

    size_t cur_kernel = 0;
    for (uint32_t i = 0; i < colsB; i++) {
      auto &scs = mcs.get_free_kernel();
      auto &kernel = scs.kernel;
      if (scs.column != -1) {
        kernel.sync();
        kernel.get_y_safe(C + rowsA * scs.column);
        scs.column = -1;
      }
      kernel.set_x(B + rowsB * i, true);
      kernel.launch(true);
      scs.column = i;
    }

    for (auto &scs : solvers) {
      if (scs.column != -1) {
        scs.kernel.get_y_safe(C + rowsA * scs.column);
      }
      scs.kernel.free_dpus();
    }
  } else {
    MCS<GEMVF_Kernel_Beta> mcs(nr_solvers);
    auto &solvers = mcs.get_solvers();
    size_t kernel_it = 0;
    for (kernel_it = 0; kernel_it < solvers.size(); kernel_it++) {
      auto &kernel = solvers[kernel_it].kernel;
      if (kernel.init(rowsA, rowsB, nr_dpus, rows_per_dpu) == false) {
        break;
      }

      kernel.set_params(alpha, beta, false);
      kernel.set_A(A, true);
    }
    solvers.resize(kernel_it);

    show_trace("Running {} kernels. Each kernel with {} DPUs.\n", solvers.size(), nr_dpus);

    size_t cur_kernel = 0;
    for (uint32_t i = 0; i < colsB; i++) {
      auto &scs = mcs.get_free_kernel();
      auto &kernel = scs.kernel;
      if (scs.column != -1) {
        kernel.get_y_safe(C + rowsA * scs.column);
        scs.column = -1;
      }
      kernel.set_x(B + rowsB * i, true);
      kernel.set_y(C + rowsA * i, true);
      kernel.launch(true);
      scs.column = i;
    }

    for (auto &scs : solvers) {
      auto &kernel = scs.kernel;
      if (scs.column != -1) {
        kernel.sync();
        kernel.get_y_safe(C + rowsA * scs.column);
        scs.column = -1;
      }

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

extern "C" {
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
All matricies are in row major format. Memory is contiguous
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

/*
A is an m by k matrix
B is an k by n matrix
C is an m by n matrix
All matricies are in row major format. Memory is contiguous
*/
void gemm_row_maj_int8(const int *m, const int *n, const int *k, const int *alpha, const int8_t *a, const int8_t *b,
                       const int *beta, int *c) {
  show_trace(
      "gemm_row_int8 m=[{}] n=[{}] k=[{}] alpha=[{}] a=[{:#018x}] b=[{:#018x}] "
      "beta=[{}] c=[{:#018x}]",
      *m, *n, *k, *alpha, reinterpret_cast<const uintptr_t>(a), reinterpret_cast<const uintptr_t>(b), *beta,
      reinterpret_cast<const uintptr_t>(c));

  // Get B to column major format
  int8_t *tmp_b = reinterpret_cast<int8_t *>(malloc(alignUp(*k * *n * sizeof(int8_t), 16)));
  transpose_matrix_row_major(b, tmp_b, *k, *n);

  // If Beta is not zero we need to change C to column major format
  int *tmp_c = reinterpret_cast<int *>(malloc(alignUp(*m * *n * sizeof(int), 16)));
  if (*beta != 0) {
    transpose_matrix_row_major(c, tmp_c, *m, *n);
  }

  sgemm_int8(*m, *k, *n, a, tmp_b, tmp_c, alpha, beta);

  // Get C from col major to row major
  transpose_matrix_column_major(tmp_c, c, *m, *n);

  free(tmp_b);
  free(tmp_c);
}

/*
A is an m by k matrix
B is an k by n matrix
C is an m by n matrix
All matricies are in row major format. Memory is contiguous
*/
void gemm_row_maj_int32(const int *m, const int *n, const int *k, const int *alpha, const int32_t *a, const int32_t *b,
                        const int *beta, int32_t *c) {
  show_trace(
      "gemm_row_int32 m=[{}] n=[{}] k=[{}] alpha=[{}] a=[{:#018x}] b=[{:#018x}] "
      "beta=[{}] c=[{:#018x}]",
      *m, *n, *k, *alpha, reinterpret_cast<const uintptr_t>(a), reinterpret_cast<const uintptr_t>(b), *beta,
      reinterpret_cast<const uintptr_t>(c));

  // Get B to column major format
  int32_t *tmp_b = reinterpret_cast<int32_t *>(malloc(alignUp(*k * *n * sizeof(int32_t), 16)));
  transpose_matrix_row_major(b, tmp_b, *k, *n);

  // If Beta is not zero we need to change C to column major format
  int *tmp_c = reinterpret_cast<int *>(malloc(alignUp(*m * *n * sizeof(int32_t), 16)));
  if (*beta != 0) {
    transpose_matrix_row_major(c, tmp_c, *m, *n);
  }

  sgemm_int32(*m, *k, *n, a, tmp_b, tmp_c, alpha, beta);

  // Get C from col major to row major
  transpose_matrix_column_major(tmp_c, c, *m, *n);

  free(tmp_b);
  free(tmp_c);
}
}
