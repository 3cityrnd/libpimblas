#include "dpu_transfer_helper.hpp"
#include "kernel.hpp"

extern "C" {
int gemv_int8(uint32_t m, uint32_t n, const int8_t* A, const int8_t* x, int* y, const int* alpha, const int* beta) {
  Kernel kernel;

  struct params {
    uint32_t rowsPerDPU;
    uint32_t row_size;
    int alpha;
    int beta;
  };

  uint32_t numDPUs = 64;
  uint32_t rowsPerDPU;

  gemv_launch_statistics<int>(m, n, numDPUs, rowsPerDPU);
  dpu_set_t dpu_set;
  DPU_ASSERT(dpu_alloc(numDPUs, nullptr, &dpu_set));

  kernel.set_dpu_set(dpu_set, numDPUs);
  kernel.load_program("gemv_int8.kernel");

  params args = {
    .rowsPerDPU  = rowsPerDPU,
    .row_size  = n,
    .alpha = *alpha,
    .beta  = *beta
  };

  kernel.set_arg_broadcast(
    "args",
    0,
    &args,
    sizeof(args),
    false);

  kernel.set_arg_scatter(
    DPU_MRAM_HEAP_POINTER_NAME,
    0,
    A,
    rowsPerDPU * n,
    m * n,
    false);

  size_t x_offset = alignUp(rowsPerDPU * n, 8);
  kernel.set_arg_broadcast(
    DPU_MRAM_HEAP_POINTER_NAME,
    x_offset,
    x,
    n,
    false);

  size_t y_offset = x_offset + alignUp(n, 8);
  kernel.set_arg_scatter(
    DPU_MRAM_HEAP_POINTER_NAME,
    y_offset,
    y,
    rowsPerDPU * sizeof(int32_t),
    m * sizeof(int32_t),
    false);

  kernel.launch(false);

  kernel.get_arg_gather(
    DPU_MRAM_HEAP_POINTER_NAME,
    y_offset,
    y,
    rowsPerDPU * sizeof(int32_t),
    m * sizeof(int32_t),
    false);

  kernel.free_dpus();
  return 0;
}
}
