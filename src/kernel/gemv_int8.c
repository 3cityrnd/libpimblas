#include <alloc.h>
#include <barrier.h>
#include <built_ins.h>
#include <defs.h>
#include <mram.h>
#include <stdint.h>
#include <string.h>

/*
Basic GEMV kernel performing y = alpha * A * x + beta * y
A is a matrix of size m x n,
x is a vector of size n
y is a vector of size m

Notes:
Part of A is transferred to single DPU - rows_per_dpu rows
Part of y - rows_per_dpu elements

x is same across all DPU's

Computing parameters:
NR_TASKLETS - number of tasklets (threads) running on single DPU
rows_per_dpu - maximum number of rows to be processed by single DPU
row_size - maximum size of single matrix row

*/

// We've got 64KB of WRAM, we are working with 8-bit ints, and need to allocate wram
// for part of A rows, and part of X rows and output(small in comparison), and 16 tasklets.
// That makes 4KB per tasklet, that means we could go to block size 2048 - aka 4KB, but that
// would leave no place for output. So we stick with 1024 for now.
// TODO: Find even more optimal value
#define BLOCK_SIZE 1024

#define MIN(x, y) (((y) < (x)) ? (y) : (x))
#define ROUND_UP(x, s) (((x) + ((s) - 1)) & ~((s) - 1))
#define ROUND_DOWN(x, s) ((x) & ~((s) - 1))

#define DOT_8(x, y, acc)                      \
  do {                                        \
    unsigned long x_dw, y_dw;                 \
    unsigned int x_lo, x_hi, y_lo, y_hi;      \
    int tmp;                                  \
                                              \
    x_dw = *((unsigned long *)(x));           \
    x_lo = x_dw;                              \
    x_hi = x_dw >> 32;                        \
    y_dw = *((unsigned long *)(y));           \
    y_lo = y_dw;                              \
    y_hi = y_dw >> 32;                        \
                                              \
    __builtin_mul_sl_sl_rrr(tmp, x_lo, y_lo); \
    acc += tmp;                               \
    __builtin_mul_sh_sh_rrr(tmp, x_lo, y_lo); \
    acc += tmp;                               \
    x_lo >>= 16;                              \
    y_lo >>= 16;                              \
    __builtin_mul_sl_sl_rrr(tmp, x_lo, y_lo); \
    acc += tmp;                               \
    __builtin_mul_sh_sh_rrr(tmp, x_lo, y_lo); \
                                              \
    acc += tmp;                               \
    __builtin_mul_sl_sl_rrr(tmp, x_hi, y_hi); \
    acc += tmp;                               \
    __builtin_mul_sh_sh_rrr(tmp, x_hi, y_hi); \
    acc += tmp;                               \
    x_hi >>= 16;                              \
    y_hi >>= 16;                              \
    __builtin_mul_sl_sl_rrr(tmp, x_hi, y_hi); \
    acc += tmp;                               \
    __builtin_mul_sh_sh_rrr(tmp, x_hi, y_hi); \
    acc += tmp;                               \
  } while (0)

struct params {
  uint32_t rows_per_dpu;
  uint32_t row_size;
  int alpha;
  int beta;
};

__host struct params args;

BARRIER_INIT(mem_reset_barrier, NR_TASKLETS);

int main() {
  int tasklet_id = me();
  if (tasklet_id == 0) {
    mem_reset();
  }
  barrier_wait(&mem_reset_barrier);

  // Sanity checks: NR_tasklets should be 16, rows_per_dpu should be a multiple of 32, because
  // rows per tasklet should be even
  if (NR_TASKLETS != 16 || args.rows_per_dpu & 31) {
    return 1;
  }
  // Rows per tasklet
  int rows_per_tasklet = args.rows_per_dpu / NR_TASKLETS;

  // Note: All MRAM allocations need to be 8B aligned in order to read from/write to them.

  // Offset of A_mram should be 8B aligned,
  // even if row_size is odd, rows_per_tasklet is always aligned to 2,
  // so it should be fine, because we are operating 4B ints.
  int A_mram_offset = tasklet_id * rows_per_tasklet * args.row_size;
  int8_t *A_mram = (int8_t *)(DPU_MRAM_HEAP_POINTER);
  int mram_offset = ROUND_UP(args.row_size * args.rows_per_dpu, 8);

  int8_t *x_mram = (int8_t *)(DPU_MRAM_HEAP_POINTER + mram_offset);
  mram_offset += ROUND_UP(args.row_size, 8);

  // Should be fine as long as rows_per_tasklet is even
  int *y_mram = (int *)(DPU_MRAM_HEAP_POINTER + mram_offset + tasklet_id * rows_per_tasklet * sizeof(int));

  // TODO: Find better way to share x across all tasklets, because now we
  // have multiple copies of the same values across tasklets.
  // If number of rows to be processed is small enough it should be possible
  // or we could just make a barrier and wait until all tasklets finish until
  // getting another part of x
  int8_t *x_wram = (int8_t *)mem_alloc(BLOCK_SIZE);
  // It's important we allocate more memory for A_wram, because of the hack
  // we later to do to write into it from mram (alignment issues).
  // We add 8B in order to be aligned.
  int8_t *A_wram = (int8_t *)mem_alloc(BLOCK_SIZE + 8);

  // Allocation needs to be aligned to 64B, or we start getting
  // allocations on top of another...
  int Ax_len = rows_per_tasklet * sizeof(int);
  int *Ax_wram = (int *)mem_alloc(Ax_len);

  // zero out the results - it's required when we are running the kernel multiple times.
  memset(Ax_wram, 0, Ax_len);

  int nr_blocks = (args.row_size - 1) / BLOCK_SIZE + 1;
  for (int b = 0; b < nr_blocks; ++b) {
    int b_offset = b * BLOCK_SIZE;
    int b_length = MIN(BLOCK_SIZE, args.row_size - b_offset);
    mram_read((__mram_ptr void *)(x_mram + b_offset), x_wram, BLOCK_SIZE);
    for (int i = 0; i < rows_per_tasklet; ++i) {
      // If offset is not aligned to 8B it will be automatically aligned down to 8 bytes
      // This happens when row_size is an odd value.
      // In our case when we are working on 8-bit ints it means we need to shift
      // 8 int (4B) to get to the values we want. That also means we need to read a bit more
      int A_offset = A_mram_offset + i * args.row_size + b_offset;
      int8_t* A_wram_read = A_wram;
      if (A_offset & 7) {
        mram_read((__mram_ptr void *)(A_mram + ROUND_DOWN(A_offset, 8)), A_wram, BLOCK_SIZE + 8);
        A_wram_read += A_offset & 7;
      } else {
        mram_read((__mram_ptr void *)(A_mram + A_offset), A_wram, BLOCK_SIZE);
      }

      int acc = 0;
      int j = 0;
      for (; j < ROUND_DOWN(b_length, 64); j += 64) {
        DOT_8(&A_wram_read[j],      &x_wram[j],      acc);
        DOT_8(&A_wram_read[j + 8],  &x_wram[j + 8],  acc);
        DOT_8(&A_wram_read[j + 16], &x_wram[j + 16], acc);
        DOT_8(&A_wram_read[j + 24], &x_wram[j + 24], acc);
        DOT_8(&A_wram_read[j + 32], &x_wram[j + 32], acc);
        DOT_8(&A_wram_read[j + 40], &x_wram[j + 40], acc);
        DOT_8(&A_wram_read[j + 48], &x_wram[j + 48], acc);
        DOT_8(&A_wram_read[j + 56], &x_wram[j + 56], acc);
      }

      for (; j < b_length; ++j) {
        acc += A_wram_read[j] * x_wram[j];
      }

      Ax_wram[i] += acc;
    }
  }

  int *y_wram = (int *)mem_alloc(Ax_len);
  mram_read((__mram_ptr void *)y_mram, y_wram, rows_per_tasklet * sizeof(int));

  for (int i = 0; i < rows_per_tasklet; ++i) {
    y_wram[i] = args.alpha * Ax_wram[i] + args.beta * y_wram[i];
  }

  mram_write(y_wram, (__mram_ptr void *)y_mram, rows_per_tasklet * sizeof(int));
  return 0;
}
