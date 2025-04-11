#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <stdint.h>
#include <string.h>

#define PERFCOUNT on

#ifdef PERFCOUNT
#include <perfcounter.h>
__host uint32_t nb_cycles[NR_TASKLETS];
__host uint32_t nb_instructions[NR_TASKLETS];
BARRIER_INIT(perfcount_start_barrier, NR_TASKLETS);
#endif

#define BLOCK_SIZE 512

__dma_aligned uint8_t vec[NR_TASKLETS][BLOCK_SIZE];
__dma_aligned int32_t result_tmp[NR_TASKLETS];

__host int32_t result;
__host uint64_t num_pairs;

BARRIER_INIT(gather_results_barrier, NR_TASKLETS);

int sx4(int x) { return (int8_t)(x << 4) >> 4; }

int32_t sum(uint8_t *vec, uint32_t nr_pairs) {
  for (uint32_t i = 0; i < nr_pairs; i++) {
  }
}

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

  barrier_wait(&gather_results_barrier);
  if (tasklet_id == 0) {
    result = 0;
    for (int i = 0; i < NR_TASKLETS; i++) {
      result += result_tmp[i];
    }
  }

#ifdef PERFCOUNT
  perfcounter_pair_t counters = perfcounter_get_both(false);
  nb_cycles[me()] = counters.cycles;
  nb_instructions[me()] = counters.instr;
#endif

  return 0;
}