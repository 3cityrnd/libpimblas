#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <stdint.h>
#include <string.h>

__host uint64_t offset;
__host int64_t result;
__host uint64_t num_elems;

#define PERFCOUNT on

#ifdef PERFCOUNT
#include <perfcounter.h>
__host uint32_t nb_cycles[NR_TASKLETS];
__host uint32_t nb_instructions[NR_TASKLETS];
BARRIER_INIT(perfcount_start_barrier, NR_TASKLETS);
#endif

unsigned int cao_32(uint32_t value) {
  unsigned int result;
  __asm__ volatile("cao %0, %1" : "=r"(result) : "r"(value));
  return result;
}
unsigned int cao_64(uint64_t value) {
  uint32_t lower = (uint32_t)(value & 0xFFFFFFFF);
  uint32_t higher = (uint32_t)(value >> 32);
  unsigned int lower_ones = cao_32(lower);
  unsigned int higher_ones = cao_32(higher);
  return lower_ones + higher_ones;
}

#define PRECISION 4
#define BLOCK_SIZE 8

int64_t clever_dot_product(const uint64_t *arr1, const uint64_t *arr2) {
  int64_t dp = 0;

  int exp = (PRECISION - 1) << 1;

  for (; exp >= PRECISION; --exp) {
    int64_t partSum = 0;

    for (int j = exp - PRECISION + 1; j <= PRECISION - 1; ++j) {
      partSum += cao_64(arr1[j] & arr2[exp - j]);
    }

    dp += partSum << exp;
  }

  for (; exp >= 0; --exp) {
    int64_t partSum = 0;

    for (int j = 0; j <= exp; ++j) {
      partSum += cao_64(arr1[j] & arr2[exp - j]);
    }

    dp += partSum << exp;
  }

  return dp;
}

__dma_aligned uint64_t vec1[NR_TASKLETS][PRECISION * BLOCK_SIZE];
__dma_aligned uint64_t vec2[NR_TASKLETS][PRECISION * BLOCK_SIZE];
__dma_aligned int64_t tmpResults[NR_TASKLETS];

BARRIER_INIT(gather_results_barrier, NR_TASKLETS);

int main() {
  //  __dma_aligned volatile uint64_t wait = wait_value;
  //  while (wait); // loops forever
#ifdef PERFCOUNT
  if (me() == 0) {
    perfcounter_config(COUNT_ENABLE_BOTH, true);
  }
  barrier_wait(&perfcount_start_barrier);
#endif
  int tasklet_id = me();

  int number_blocks = num_elems / PRECISION;
  int number_blocks_per_tasklet = (number_blocks - 1) / NR_TASKLETS + 1;
  int tasklet_block_start = tasklet_id * number_blocks_per_tasklet;
  int blocks_per_tasklet_size = number_blocks_per_tasklet * PRECISION * sizeof(uint64_t);

  uint64_t *vec1_MRAM = (uint64_t *)(DPU_MRAM_HEAP_POINTER + tasklet_id * blocks_per_tasklet_size);
  uint64_t *vec2_MRAM = (uint64_t *)(DPU_MRAM_HEAP_POINTER + offset  // Offset to second vec
                                     + tasklet_id * blocks_per_tasklet_size);

  tmpResults[tasklet_id] = 0;

  for (uint32_t block_id = 0; block_id < number_blocks_per_tasklet; block_id += BLOCK_SIZE) {
    const int block_offset = block_id * PRECISION;

    const int num_blocks_left = number_blocks - tasklet_block_start - block_id;
    if (num_blocks_left <= 0) break;

    int num_blocks = BLOCK_SIZE;
    if (num_blocks > num_blocks_left) {
      num_blocks = num_blocks_left;
    }
    if (num_blocks > number_blocks_per_tasklet - block_id) {
      num_blocks = number_blocks_per_tasklet - block_id;
    }

    mram_read((__mram_ptr void *)(vec1_MRAM + block_offset), vec1[tasklet_id],
              num_blocks * PRECISION * sizeof(uint64_t));
    mram_read((__mram_ptr void *)(vec2_MRAM + block_offset), vec2[tasklet_id],
              num_blocks * PRECISION * sizeof(uint64_t));

    for (int i = 0; i < num_blocks; i++) {
      tmpResults[tasklet_id] += clever_dot_product(vec1[tasklet_id] + PRECISION * i, vec2[tasklet_id] + PRECISION * i);
    }
  }

  barrier_wait(&gather_results_barrier);
  if (tasklet_id == 0) {
    result = 0;
    for (int i = 0; i < NR_TASKLETS; i++) {
      result += tmpResults[i];
    }
  }

#ifdef PERFCOUNT
  perfcounter_pair_t counters = perfcounter_get_both(false);
  nb_cycles[me()] = counters.cycles;
  nb_instructions[me()] = counters.instr;
#endif

  return 0;
}