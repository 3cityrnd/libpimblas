#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
// #include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((constructor)) void pimblas_constructor();

__attribute__((destructor)) void pimblas_destructor();

/* Internal Functions */

const char *pimblas_get_kernel_dir();
char *pimblas_get_kernel_dir_concat_free(const char *name);
const char *pimblas_get_git_version();

/* end of Internal Functions */

int pimblas_test_function(int c);
int pimblas_relu(void *ptr, size_t size);

int gemv(uint32_t m, uint32_t n, const int *mat, const int *vec, int *out);

int gemv_f_basic(uint32_t m, uint32_t n, const float *mat, const float *vec, float *out);

int gemv_f(uint32_t m, uint32_t n, const float *A, const float *x, float *y, const float *alpha, const float *beta);

int gemv_int32(uint32_t m, uint32_t n, const int *A, const int *x, int *y, const int *alpha, const int *beta);
int gemv_int8(uint32_t m, uint32_t n, const int8_t *A, const int8_t *x, int *y, const int *alpha, const int *beta);

int vector_add(const int *a_input_ptr, const int *b_input_ptr, size_t num_elem, int *output);
void sgemm_wrapper(const char *transa, const char *transb, const int *m, const int *n, const int *k, const float *alpha,
                   const float *a, const int *lda, const float *b, const int *ldb, const float *beta, float *c,
                   const int *ldc);

void gemm_row_maj_f(const int *m, const int *n, const int *k, const float *alpha, const float *a, const float *b,
                    const float *beta, float *c);
void gemm_row_maj_int8(const int *m, const int *n, const int *k, const int *alpha, const int8_t *a, const int8_t *b,
                       const int *beta, int *c);
void gemm_row_maj_int32(const int *m, const int *n, const int *k, const int *alpha, const int32_t *a, const int32_t *b,
                        const int *beta, int *c);

int relu_f(const float *input, float *output, size_t num_elem);
int vec_add_f(const float *input_a, const float *input_b, float *output, size_t size);
int vec_mul_f(const float *input_a, const float *input_b, float *output, size_t size);
int vec_sub_f(const float *input_a, const float *input_b, float *output, size_t size);

int softmax(const float *vec_in, float *vec_out, size_t size);

/* CBLAS API */

/* end of CBLAS API */

#ifdef __cplusplus
}
#endif
