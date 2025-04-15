#include <alloc.h>
#include <barrier.h>
#include <built_ins.h>
#include <defs.h>
#include <mram.h>
#include <mram_unaligned.h>
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

#define BLOCK_SIZE 512

struct params {
  uint32_t rows_per_dpu;
  uint32_t row_size;
  int alpha;
  int beta;
};

__host struct params args;

BARRIER_INIT(mem_reset_barrier, NR_TASKLETS);

uint32_t alignUpTo8(uint32_t value) { return (value + 7) & ~7; }

uint32_t alignDownTo8(uint32_t value) { return value & ~7; }

uint32_t alignUpTo64(uint32_t value) { return (value + 63) & ~63; }

uint32_t alignUpTo2(uint32_t value) { return (value + 1) & ~1; }

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

  uint32_t mram_offset_in_bytes = 0;

  int8_t *A_mram = (int8_t *)(DPU_MRAM_HEAP_POINTER + mram_offset_in_bytes +
                              (tasklet_id * args.row_size * rows_per_tasklet) * sizeof(int8_t));
  mram_offset_in_bytes += alignUpTo8(args.row_size * args.rows_per_dpu * sizeof(int8_t));

  int8_t *x_mram = (int8_t *)(DPU_MRAM_HEAP_POINTER + mram_offset_in_bytes);
  mram_offset_in_bytes += alignUpTo8(args.row_size * sizeof(int8_t));

  // Should be fine as long as rows_per_tasklet is even
  int *result_mram =
      (int *)(DPU_MRAM_HEAP_POINTER + mram_offset_in_bytes + (tasklet_id * rows_per_tasklet) * sizeof(int));

  // TODO: Find better way to share x across all tasklets, because now we
  // have multiple copies of the same values across tasklets.
  // If number of rows to be processed is small enough it should be possible
  // or we could just make a barrier and wait until all tasklets finish until
  // getting another part of x
  int8_t *x_wram = (int *)mem_alloc(BLOCK_SIZE * sizeof(int8_t));
  // It's important we allocate more memory for A_wram, because of the hack
  // we later to do to write into it from mram (alignment issues).
  // We add 64B in order to be aligned.
  int8_t *A_wram = (int *)mem_alloc((BLOCK_SIZE) * sizeof(int8_t) + 64);

  uint32_t result_size = alignUpTo64(rows_per_tasklet * sizeof(int));
  int *mul_result_wram = (int *)mem_alloc(result_size);

  // zero out the results - it's required when we are running the kernel multiple times.
  memset(mul_result_wram, 0, result_size);

  int nr_blocks = (args.row_size - 1) / BLOCK_SIZE + 1;
  for (uint32_t block = 0; block < nr_blocks; block++) {
    const int block_offset = block * BLOCK_SIZE;

    int block_length = block_offset + BLOCK_SIZE <= args.row_size ? BLOCK_SIZE : args.row_size - block_offset;
    mram_read((__mram_ptr void *)(x_mram + block_offset), x_wram, BLOCK_SIZE * sizeof(int8_t));
    for (uint32_t i = 0; i < rows_per_tasklet; i++) {
      uint32_t a_offset = (uint32_t)(A_mram + i * args.row_size + block_offset);
      int8_t *A_wram_read = NULL;

      if (a_offset & 7) {
        int offset = (a_offset & 7);
        mram_read((__mram_ptr void *)(alignDownTo8(a_offset)), A_wram, (BLOCK_SIZE + 8) * sizeof(int8_t));
        A_wram_read = (A_wram + offset);
      } else {
        mram_read((__mram_ptr void *)(a_offset), A_wram, BLOCK_SIZE * sizeof(int8_t));
        A_wram_read = A_wram;
      }

      int sum = 0;
#pragma unroll
      for (uint32_t j = 0; j < block_length; ++j) {
        int tmp;
        __builtin_mul_sl_sl_rrr(tmp, A_wram_read[j], x_wram[j]);
        sum += tmp;
      }

      mul_result_wram[i] += sum;
    }
  }

  if (args.beta != 0) {
    int *result_wram = (int *)mem_alloc(result_size);
    mram_read((__mram_ptr void *)(result_mram), result_wram, rows_per_tasklet * sizeof(int));

    if (args.alpha != 1) {
      for (uint32_t i = 0; i < rows_per_tasklet; i++) {
        result_wram[i] = args.alpha * mul_result_wram[i] + args.beta * result_wram[i];
      }
    } else {
      for (uint32_t i = 0; i < rows_per_tasklet; i++) {
        result_wram[i] = mul_result_wram[i] + args.beta * result_wram[i];
      }
    }
    mram_write(result_wram, (__mram_ptr void *)result_mram, rows_per_tasklet * sizeof(int));
  } else {
    if (args.alpha != 1) {
      for (uint32_t i = 0; i < rows_per_tasklet; i++) {
        mul_result_wram[i] = args.alpha * mul_result_wram[i];
      }
    }
    mram_write(mul_result_wram, (__mram_ptr void *)result_mram, rows_per_tasklet * sizeof(int));
  }

  return 0;
}