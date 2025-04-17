#include "dpu_transfer_helper.hpp"

#include <cassert>

#include "common.hpp"

template <typename T>
void gemv_launch_statistics(uint32_t m, uint32_t n, uint32_t &numDPUs, uint32_t &rowsPerDPU) {
  // Assumptions:
  // MRAM size of each DPU is 64MB
  // part of A needs to be copied to each DPU - n * rows_per_dpu
  // x vector needs to be copied to each DPU - n
  // part of y vector needs to be copied to each DPU - rows_per_dpu
  // Total ints per DPU: n * (rows_per_dpu + 1) + rows_per_dpu
  // Threads per DPU: 16
  // At minimum two rows per tasklet when sizeof(T) == 4 (because the output needs to be 8B aligned)
  constexpr size_t minRowsPerDPU = 16 * 8 / sizeof(T);

  rowsPerDPU = alignUp((m - 1) / numDPUs + 1, minRowsPerDPU);
  size_t memory_requirement = (n * (rowsPerDPU + 1) + rowsPerDPU) * sizeof(T);

  // Let's leave 1 MB
  constexpr size_t mem_cap = 63 * 1024 * 1024;
  while (memory_requirement > mem_cap) {
    rowsPerDPU -= minRowsPerDPU;
    memory_requirement = n * (rowsPerDPU + 1) + rowsPerDPU;
  }

  if (rowsPerDPU < minRowsPerDPU) {
    rowsPerDPU = minRowsPerDPU;
  }

  numDPUs = (m - 1) / rowsPerDPU + 1;
}

// Instantiation
template void gemv_launch_statistics<int>(uint32_t m, uint32_t n, uint32_t &numDPUs, uint32_t &rowsPerDPU);
template void gemv_launch_statistics<float>(uint32_t m, uint32_t n, uint32_t &numDPUs, uint32_t &rowsPerDPU);

size_t safe_gather(dpu_set_t set, uint32_t nr_dpus, const char *symbol_name, size_t symbol_offset, uint8_t *data,
                   size_t chunk_size, size_t size) {
  bool has_remainder = (size < chunk_size) || (size % chunk_size != 0);

  dpu_set_t dpu;
  uint32_t dpu_idx;
  DPU_FOREACH(set, dpu, dpu_idx) {
    auto offset = dpu_idx * chunk_size;
    if (false == (has_remainder && dpu_idx + 1 == nr_dpus)) {
      DPU_ASSERT(dpu_prepare_xfer(dpu, (void *)&data[offset]));
    }
  }
  DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, symbol_name, symbol_offset, chunk_size, DPU_XFER_DEFAULT));

  if (has_remainder) {
    auto offset = (nr_dpus - 1) * chunk_size;
    auto remainder = size - offset;
    auto transfer_size = alignDown(remainder, 8);
    auto missing_size = remainder - transfer_size;

    DPU_ASSERT(dpu_prepare_xfer(dpu, (void *)&data[offset]));
    DPU_ASSERT(dpu_push_xfer(dpu, DPU_XFER_FROM_DPU, symbol_name, symbol_offset, transfer_size, DPU_XFER_DEFAULT));

    if (missing_size > 0) {
      uint8_t tmp_buffer[8];
      auto last_read = &data[offset + transfer_size];
      DPU_ASSERT(dpu_copy_from(dpu, symbol_name, symbol_offset + transfer_size, (void *)tmp_buffer, 8));
      memcpy(last_read, tmp_buffer, missing_size);
    }
  }

  return symbol_offset + alignUp(chunk_size, 8);
}
