#include <cstddef>
#include <cstdint>

void transpose_matrix_column_major(const float *src, float *dst, size_t rows, size_t cols);
void transpose_matrix_row_major(const float *src, float *dst, size_t rows, size_t cols);

void transpose_matrix_column_major(const int8_t *src, int8_t *dst, size_t rows, size_t cols);
void transpose_matrix_row_major(const int8_t *src, int8_t *dst, size_t rows, size_t cols);

void transpose_matrix_column_major(const int32_t *src, int32_t *dst, size_t rows, size_t cols);
void transpose_matrix_row_major(const int32_t *src, int32_t *dst, size_t rows, size_t cols);
