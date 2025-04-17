#include "matrix_transpose.hpp"

#include <immintrin.h>

#include <algorithm>

void transpose8x8_block(const int32_t *src, int32_t *dst, size_t src_stride, size_t dst_stride) {
  // Load 8x8 block (each __m256i holds 8 int32_t elements)
  auto row0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&src[0 * src_stride]));
  auto row1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&src[1 * src_stride]));
  auto row2 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&src[2 * src_stride]));
  auto row3 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&src[3 * src_stride]));
  auto row4 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&src[4 * src_stride]));
  auto row5 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&src[5 * src_stride]));
  auto row6 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&src[6 * src_stride]));
  auto row7 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&src[7 * src_stride]));

  // Transpose 8x8 matrix using AVX2
  __m256i t0 = _mm256_unpacklo_epi32(row0, row1);
  __m256i t1 = _mm256_unpackhi_epi32(row0, row1);
  __m256i t2 = _mm256_unpacklo_epi32(row2, row3);
  __m256i t3 = _mm256_unpackhi_epi32(row2, row3);
  __m256i t4 = _mm256_unpacklo_epi32(row4, row5);
  __m256i t5 = _mm256_unpackhi_epi32(row4, row5);
  __m256i t6 = _mm256_unpacklo_epi32(row6, row7);
  __m256i t7 = _mm256_unpackhi_epi32(row6, row7);

  __m256i tt0 = _mm256_unpacklo_epi64(t0, t2);
  __m256i tt1 = _mm256_unpackhi_epi64(t0, t2);
  __m256i tt2 = _mm256_unpacklo_epi64(t1, t3);
  __m256i tt3 = _mm256_unpackhi_epi64(t1, t3);
  __m256i tt4 = _mm256_unpacklo_epi64(t4, t6);
  __m256i tt5 = _mm256_unpackhi_epi64(t4, t6);
  __m256i tt6 = _mm256_unpacklo_epi64(t5, t7);
  __m256i tt7 = _mm256_unpackhi_epi64(t5, t7);

  __m256i o0 = _mm256_permute2x128_si256(tt0, tt4, 0x20);
  __m256i o1 = _mm256_permute2x128_si256(tt1, tt5, 0x20);
  __m256i o2 = _mm256_permute2x128_si256(tt2, tt6, 0x20);
  __m256i o3 = _mm256_permute2x128_si256(tt3, tt7, 0x20);
  __m256i o4 = _mm256_permute2x128_si256(tt0, tt4, 0x31);
  __m256i o5 = _mm256_permute2x128_si256(tt1, tt5, 0x31);
  __m256i o6 = _mm256_permute2x128_si256(tt2, tt6, 0x31);
  __m256i o7 = _mm256_permute2x128_si256(tt3, tt7, 0x31);

  // Store the transposed matrix
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(&dst[0 * dst_stride]), o0);
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(&dst[1 * dst_stride]), o1);
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(&dst[2 * dst_stride]), o2);
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(&dst[3 * dst_stride]), o3);
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(&dst[4 * dst_stride]), o4);
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(&dst[5 * dst_stride]), o5);
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(&dst[6 * dst_stride]), o6);
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(&dst[7 * dst_stride]), o7);
}

void transpose_matrix_column_major(const int32_t *src, int32_t *dst, size_t rows, size_t cols) {
  constexpr size_t block_size = 8;

  for (size_t i = 0; i < cols; i += block_size) {
    for (size_t j = 0; j < rows; j += block_size) {
      size_t block_rows = std::min(block_size, rows - j);
      size_t block_cols = std::min(block_size, cols - i);

      if (block_rows == 8 && block_cols == 8) {
        transpose8x8_block(&src[i * rows + j], &dst[j * cols + i], rows, cols);
      } else {
        for (size_t ii = 0; ii < block_cols; ii++) {
          for (size_t jj = 0; jj < block_rows; jj++) {
            dst[(j + jj) * cols + (i + ii)] = src[(i + ii) * rows + (j + jj)];
          }
        }
      }
    }
  }
}

void transpose_matrix_row_major(const int32_t *src, int32_t *dst, size_t rows, size_t cols) {
  constexpr size_t block_size = 8;

  for (size_t i = 0; i < rows; i += block_size) {
    for (size_t j = 0; j < cols; j += block_size) {
      size_t block_rows = std::min(block_size, rows - i);
      size_t block_cols = std::min(block_size, cols - j);

      if (block_rows == 8 && block_cols == 8) {
        transpose8x8_block(&src[i * cols + j], &dst[j * rows + i], cols, rows);
      } else {
        for (size_t ii = 0; ii < block_rows; ii++) {
          for (size_t jj = 0; jj < block_cols; jj++) {
            dst[(j + jj) * rows + (i + ii)] = src[(i + ii) * cols + (j + jj)];
          }
        }
      }
    }
  }
}

void transpose8x8_block(const int8_t *src, size_t src_stride, int8_t *dst, size_t dst_stride) {
  __m128i row0 = _mm_loadl_epi64((const __m128i *)(src + 0 * src_stride));
  __m128i row1 = _mm_loadl_epi64((const __m128i *)(src + 1 * src_stride));
  __m128i row2 = _mm_loadl_epi64((const __m128i *)(src + 2 * src_stride));
  __m128i row3 = _mm_loadl_epi64((const __m128i *)(src + 3 * src_stride));
  __m128i row4 = _mm_loadl_epi64((const __m128i *)(src + 4 * src_stride));
  __m128i row5 = _mm_loadl_epi64((const __m128i *)(src + 5 * src_stride));
  __m128i row6 = _mm_loadl_epi64((const __m128i *)(src + 6 * src_stride));
  __m128i row7 = _mm_loadl_epi64((const __m128i *)(src + 7 * src_stride));

  __m128i t0 = _mm_unpacklo_epi8(row0, row1);
  __m128i t1 = _mm_unpacklo_epi8(row2, row3);
  __m128i t2 = _mm_unpacklo_epi8(row4, row5);
  __m128i t3 = _mm_unpacklo_epi8(row6, row7);

  __m128i abcd_lo = _mm_unpacklo_epi16(t0, t2);
  __m128i abcd_hi = _mm_unpackhi_epi16(t0, t2);
  __m128i efgh_lo = _mm_unpacklo_epi16(t1, t3);
  __m128i efgh_hi = _mm_unpackhi_epi16(t1, t3);

  __m128i o0 = _mm_unpacklo_epi32(abcd_lo, efgh_lo);
  __m128i o1 = _mm_unpackhi_epi32(abcd_lo, efgh_lo);
  __m128i o2 = _mm_unpacklo_epi32(abcd_hi, efgh_hi);
  __m128i o3 = _mm_unpackhi_epi32(abcd_hi, efgh_hi);

  // Swap (2 <-> 4 and 3 <-> 5)
  const __m128i shuffle_mask = _mm_setr_epi8(0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15);

  o0 = _mm_shuffle_epi8(o0, shuffle_mask);
  o1 = _mm_shuffle_epi8(o1, shuffle_mask);
  o2 = _mm_shuffle_epi8(o2, shuffle_mask);
  o3 = _mm_shuffle_epi8(o3, shuffle_mask);

  _mm_storel_epi64(reinterpret_cast<__m128i *>(dst + 0 * dst_stride), o0);
  _mm_storel_epi64(reinterpret_cast<__m128i *>(dst + 1 * dst_stride), _mm_srli_si128(o0, 8));
  _mm_storel_epi64(reinterpret_cast<__m128i *>(dst + 2 * dst_stride), o1);
  _mm_storel_epi64(reinterpret_cast<__m128i *>(dst + 3 * dst_stride), _mm_srli_si128(o1, 8));
  _mm_storel_epi64(reinterpret_cast<__m128i *>(dst + 4 * dst_stride), o2);
  _mm_storel_epi64(reinterpret_cast<__m128i *>(dst + 5 * dst_stride), _mm_srli_si128(o2, 8));
  _mm_storel_epi64(reinterpret_cast<__m128i *>(dst + 6 * dst_stride), o3);
  _mm_storel_epi64(reinterpret_cast<__m128i *>(dst + 7 * dst_stride), _mm_srli_si128(o3, 8));
}

void transpose_matrix_column_major(const int8_t *src, int8_t *dst, size_t rows, size_t cols) {
  constexpr size_t block_size = 8;

  for (size_t i = 0; i < cols; i += block_size) {
    for (size_t j = 0; j < rows; j += block_size) {
      size_t block_rows = std::min(block_size, rows - j);
      size_t block_cols = std::min(block_size, cols - i);

      if (block_rows == 8 && block_cols == 8) {
        transpose8x8_block(&src[i * rows + j], rows, &dst[j * cols + i], cols);
      } else {
        for (size_t ii = 0; ii < block_cols; ii++) {
          for (size_t jj = 0; jj < block_rows; jj++) {
            dst[(j + jj) * cols + (i + ii)] = src[(i + ii) * rows + (j + jj)];
          }
        }
      }
    }
  }
}

void transpose_matrix_row_major(const int8_t *src, int8_t *dst, size_t rows, size_t cols) {
  constexpr size_t block_size = 8;

  for (size_t i = 0; i < rows; i += block_size) {
    for (size_t j = 0; j < cols; j += block_size) {
      size_t block_rows = std::min(block_size, rows - i);
      size_t block_cols = std::min(block_size, cols - j);

      if (block_rows == 8 && block_cols == 8) {
        transpose8x8_block(&src[i * cols + j], cols, &dst[j * rows + i], rows);
      } else {
        for (size_t ii = 0; ii < block_rows; ii++) {
          for (size_t jj = 0; jj < block_cols; jj++) {
            dst[(j + jj) * rows + (i + ii)] = src[(i + ii) * cols + (j + jj)];
          }
        }
      }
    }
  }
}

void transpose8x8_block(const float *src, float *dst, size_t src_stride, size_t dst_stride) {
  auto col0 = _mm256_loadu_ps(&src[0 * src_stride]);
  auto col1 = _mm256_loadu_ps(&src[1 * src_stride]);
  auto col2 = _mm256_loadu_ps(&src[2 * src_stride]);
  auto col3 = _mm256_loadu_ps(&src[3 * src_stride]);
  auto col4 = _mm256_loadu_ps(&src[4 * src_stride]);
  auto col5 = _mm256_loadu_ps(&src[5 * src_stride]);
  auto col6 = _mm256_loadu_ps(&src[6 * src_stride]);
  auto col7 = _mm256_loadu_ps(&src[7 * src_stride]);

  auto t0 = _mm256_unpacklo_ps(col0, col1);  // a00, a10, a01, a11 ...
  auto t1 = _mm256_unpackhi_ps(col0, col1);  // a02, a12, a03, a13
  auto t2 = _mm256_unpacklo_ps(col2, col3);
  auto t3 = _mm256_unpackhi_ps(col2, col3);
  auto t4 = _mm256_unpacklo_ps(col4, col5);
  auto t5 = _mm256_unpackhi_ps(col4, col5);
  auto t6 = _mm256_unpacklo_ps(col6, col7);
  auto t7 = _mm256_unpackhi_ps(col6, col7);

  auto r0 = _mm256_shuffle_ps(t0, t2, 0x44);  // a00, a10, a20, a30
  auto r1 = _mm256_shuffle_ps(t0, t2, 0xEE);  // a01, a11, a21, a31
  auto r2 = _mm256_shuffle_ps(t1, t3, 0x44);
  auto r3 = _mm256_shuffle_ps(t1, t3, 0xEE);
  auto r4 = _mm256_shuffle_ps(t4, t6, 0x44);
  auto r5 = _mm256_shuffle_ps(t4, t6, 0xEE);
  auto r6 = _mm256_shuffle_ps(t5, t7, 0x44);
  auto r7 = _mm256_shuffle_ps(t5, t7, 0xEE);

  auto p0 = _mm256_permute2f128_ps(r0, r4, 0x20);  // Low 128 bits of r0 and r4
  auto p1 = _mm256_permute2f128_ps(r1, r5, 0x20);
  auto p2 = _mm256_permute2f128_ps(r2, r6, 0x20);
  auto p3 = _mm256_permute2f128_ps(r3, r7, 0x20);
  auto p4 = _mm256_permute2f128_ps(r0, r4, 0x31);  // High 128 bits of r0 and r4
  auto p5 = _mm256_permute2f128_ps(r1, r5, 0x31);
  auto p6 = _mm256_permute2f128_ps(r2, r6, 0x31);
  auto p7 = _mm256_permute2f128_ps(r3, r7, 0x31);

  _mm256_storeu_ps(&dst[0 * dst_stride], p0);
  _mm256_storeu_ps(&dst[1 * dst_stride], p1);
  _mm256_storeu_ps(&dst[2 * dst_stride], p2);
  _mm256_storeu_ps(&dst[3 * dst_stride], p3);
  _mm256_storeu_ps(&dst[4 * dst_stride], p4);
  _mm256_storeu_ps(&dst[5 * dst_stride], p5);
  _mm256_storeu_ps(&dst[6 * dst_stride], p6);
  _mm256_storeu_ps(&dst[7 * dst_stride], p7);
}

void transpose_matrix_column_major(const float *src, float *dst, size_t rows, size_t cols) {
  constexpr size_t block_size = 8;

  for (size_t i = 0; i < cols; i += block_size) {
    for (size_t j = 0; j < rows; j += block_size) {
      size_t block_rows = std::min(block_size, rows - j);
      size_t block_cols = std::min(block_size, cols - i);

      if (block_rows == 8 && block_cols == 8) {
        transpose8x8_block(&src[i * rows + j], &dst[j * cols + i], rows, cols);
      } else {
        for (size_t ii = 0; ii < block_cols; ii++) {
          for (size_t jj = 0; jj < block_rows; jj++) {
            dst[(j + jj) * cols + (i + ii)] = src[(i + ii) * rows + (j + jj)];
          }
        }
      }
    }
  }
}

void transpose_matrix_row_major(const float *src, float *dst, size_t rows, size_t cols) {
  constexpr size_t block_size = 8;

  for (size_t i = 0; i < rows; i += block_size) {
    for (size_t j = 0; j < cols; j += block_size) {
      size_t block_rows = std::min(block_size, rows - i);
      size_t block_cols = std::min(block_size, cols - j);

      if (block_rows == 8 && block_cols == 8) {
        transpose8x8_block(&src[i * cols + j], &dst[j * rows + i], cols, rows);
      } else {
        for (size_t ii = 0; ii < block_rows; ii++) {
          for (size_t jj = 0; jj < block_cols; jj++) {
            dst[(j + jj) * rows + (i + ii)] = src[(i + ii) * cols + (j + jj)];
          }
        }
      }
    }
  }
}
