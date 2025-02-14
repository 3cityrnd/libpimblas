#include <defs.h>
#include <mram.h>
#include <stdio.h>

#define BUFFER_SIZE 512
// 2x 16MB allocation
#define MRAM_ALLOCATION (1024 * 1024 * 4)
__mram_noinit float buffer_a[MRAM_ALLOCATION];
__mram_noinit float buffer_b[MRAM_ALLOCATION];

__mram_noinit int32_t params[4];

#define ACTIONABLE_LENGTH_POS 0
#define OP_TYPE_POS 1
#define VEC_ADD 1
#define VEC_MUL 2

int main() {
  __dma_aligned float local_cache_a[BUFFER_SIZE];
  __dma_aligned float local_cache_b[BUFFER_SIZE];
  int tasklet_id = me();
  if (tasklet_id == 0) {
    mem_reset();
  }
  if (tasklet_id > 0) {
    return 0;
  }

  int actionable_length = params[ACTIONABLE_LENGTH_POS];
  printf("Actionable length: %d\n", actionable_length);
  if ((actionable_length % 2) != 0) {
    printf("Actionable len not aligen to 8\n");
    return 0;
  }

  int full_copies = actionable_length / BUFFER_SIZE;
  int reminder = actionable_length % BUFFER_SIZE;

  for (int i = 0; i < full_copies; i++) {
    mram_read(buffer_a + (i * BUFFER_SIZE), local_cache_a, BUFFER_SIZE * sizeof(float));
    mram_read(buffer_b + (i * BUFFER_SIZE), local_cache_b, BUFFER_SIZE * sizeof(float));

    for (int j = 0; j < BUFFER_SIZE; j++) {
      if (params[OP_TYPE_POS] == VEC_ADD) {
        local_cache_a[j] = local_cache_a[j] + local_cache_b[j];
      } else if (params[OP_TYPE_POS] == VEC_MUL) {
        local_cache_a[j] = local_cache_a[j] * local_cache_b[j];
      }
    }
    mram_write(local_cache_a, buffer_a + (i * BUFFER_SIZE), BUFFER_SIZE * sizeof(float));
  }

  if (reminder > 0) {
    mram_read(buffer_a + (full_copies * BUFFER_SIZE), local_cache_a, reminder * sizeof(float));
    mram_read(buffer_b + (full_copies * BUFFER_SIZE), local_cache_b, reminder * sizeof(float));

    for (int j = 0; j < reminder; j++) {
      if (params[OP_TYPE_POS] == VEC_ADD) {
        local_cache_a[j] = local_cache_a[j] + local_cache_b[j];
      } else if (params[OP_TYPE_POS] == VEC_MUL) {
        local_cache_a[j] = local_cache_a[j] * local_cache_b[j];
      }
    }
    mram_write(local_cache_a, buffer_a + (full_copies * BUFFER_SIZE), reminder * sizeof(float));
  }
}
