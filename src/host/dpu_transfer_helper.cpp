#include "dpu_transfer_helper.hpp"

#include "common.hpp"
#include <cassert>

template <typename T>
void transfer_chunks_to_mram(dpu_set_t set, const char *symbol, T *data, size_t chunk_size, size_t size) {
  bool has_remainder = size % chunk_size != 0;

  uint32_t nr_dpus = 0;
  DPU_ASSERT(dpu_get_nr_dpus(set, &nr_dpus));

  dpu_set_t dpu;
  uint32_t dpu_idx;
  DPU_FOREACH(set, dpu, dpu_idx) {
    // Set the offset to transfarable memory for each dpu
    auto offset = dpu_idx * chunk_size;
    if (has_remainder && dpu_idx + 1 == nr_dpus) {  // Handle remainder
      size_t remainder = size - offset;
      DPU_ASSERT(
          dpu_broadcast_to(dpu, symbol, 0, (void *)&data[offset], alignUp(remainder * sizeof(T), 8), DPU_XFER_DEFAULT));
    } else {
      DPU_ASSERT(dpu_prepare_xfer(dpu, (void *)&data[offset]));
    }
  }
  DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, symbol, 0, chunk_size * sizeof(T), DPU_XFER_DEFAULT));
}

template <typename T>
void transfer_full_to_mram(dpu_set_t set, const char *symbol, T *data, size_t size) {
  DPU_ASSERT(dpu_broadcast_to(set, symbol, 0, data, alignUp(size * sizeof(T), 8), DPU_XFER_DEFAULT));
}

template <typename T>
void transfer_chunks_from_mram(dpu_set_t set, const char *symbol, T *data, size_t chunk_size, size_t size) {
  bool has_remainder = size % chunk_size != 0;

  uint32_t nr_dpus = 0;
  DPU_ASSERT(dpu_get_nr_dpus(set, &nr_dpus));

  dpu_set_t dpu;
  uint32_t dpu_idx;
  DPU_FOREACH(set, dpu, dpu_idx) {
    auto offset = dpu_idx * chunk_size;
    if (has_remainder && dpu_idx + 1 == nr_dpus) {
      size_t remainder = size - offset;
      DPU_ASSERT(dpu_copy_from(dpu, symbol, 0, &data[offset], alignUp(remainder * sizeof(T), 8)));
    } else {
      DPU_ASSERT(dpu_prepare_xfer(dpu, data[offset]));
    }
  }
  DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, symbol, 0, chunk_size * sizeof(T), DPU_XFER_DEFAULT));
}

template <typename T>
size_t transfer_chunks_from_mram_directly(dpu_set_t set, uint32_t nrDPUs, size_t offset, T *data, size_t chunkSize,
                                          size_t size) {
  bool has_remainder = size % chunkSize != 0;

  dpu_set_t dpu;
  uint32_t dpu_idx;
  DPU_FOREACH(set, dpu, dpu_idx) {
    if (has_remainder && dpu_idx + 1 == nrDPUs) {
      size_t remainder = size - dpu_idx * chunkSize;
      DPU_ASSERT(dpu_copy_from(dpu, DPU_MRAM_HEAP_POINTER_NAME, offset, data + dpu_idx * chunkSize,
                               alignUp(remainder * sizeof(T), 8)));
    } else {
      DPU_ASSERT(dpu_prepare_xfer(dpu, data + dpu_idx * chunkSize));
    }
  }

  DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, offset, chunkSize * sizeof(T),
                           DPU_XFER_DEFAULT));
  return offset + alignUp(chunkSize * sizeof(T), 8);
}

template <typename T>
size_t transfer_chunks_to_mram_directly(dpu_set_t set, uint32_t nrDPUs, size_t offset, T *data, size_t chunkSize,
                                        size_t size) {
  bool has_remainder = size % chunkSize != 0;

  dpu_set_t dpu;
  uint32_t dpu_idx;
  DPU_FOREACH(set, dpu, dpu_idx) {
    if (has_remainder && dpu_idx + 1 == nrDPUs) {
      size_t remainder = size - dpu_idx * chunkSize;
      DPU_ASSERT(dpu_copy_to(dpu, DPU_MRAM_HEAP_POINTER_NAME, offset, (void *)(data + dpu_idx * chunkSize),
                             alignUp(remainder * sizeof(T), 8)));
    } else {
      DPU_ASSERT(dpu_prepare_xfer(dpu, (void *)(data + dpu_idx * chunkSize)));
    }
  }

  DPU_ASSERT(
      dpu_push_xfer(set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, offset, chunkSize * sizeof(T), DPU_XFER_DEFAULT));
  return offset + alignUp(chunkSize * sizeof(T), 8);
}

template <typename T>
size_t transfer_full_to_mram_directly(dpu_set_t set, uint32_t nrDPUs, size_t offset, T *data, size_t size) {
  size_t copySize = alignUp(size * sizeof(T), 8);
  DPU_ASSERT(dpu_broadcast_to(set, DPU_MRAM_HEAP_POINTER_NAME, offset, data, copySize, DPU_XFER_DEFAULT));
  return offset + copySize;
}

void gemv_launch_statistics(uint32_t m, uint32_t n, uint32_t &numDPUs, uint32_t &rowsPerDPU) {
  // Assumptions:
  // MRAM size of each DPU is 64MB
  // part of A needs to be copied to each DPU - n * rows_per_dpu
  // x vector needs to be copied to each DPU - n
  // part of y vector needs to be copied to each DPU - rows_per_dpu
  // part of results resides on each DPU - rows_per_dpu
  // Total ints per DPU: n * (rows_per_dpu + 1) + 2 * rows_per_dpu
  // Threads per DPU: 16
  // At minimum two rows per tasklet (because the output needs to be 8B aligned)

  rowsPerDPU = alignUp((m - 1) / numDPUs + 1, 32);
  size_t memory_requirement = (n * (rowsPerDPU + 1) + 2 * rowsPerDPU) * sizeof(float);

  // Let's leave 1 MB
  constexpr size_t mem_cap = 63 * 1024 * 1024;
  while (memory_requirement > mem_cap) {
    rowsPerDPU -= 32;
    memory_requirement = n * (rowsPerDPU + 1) + 2 * rowsPerDPU;
  }

  constexpr size_t minRowsPerDPU = 32 * 8 / sizeof(float);
  if (rowsPerDPU < minRowsPerDPU) {
    rowsPerDPU = minRowsPerDPU;
  }

  numDPUs = (m - 1) / rowsPerDPU + 1;

  constexpr uint32_t maxDPUs = 128;
  assert(numDPUs <= maxDPUs);
}

// Instantiation
template size_t transfer_chunks_from_mram_directly<float>(dpu_set_t set, uint32_t nrDPUs, size_t offset, float *data,
                                                          size_t chunkSize, size_t size);

template size_t transfer_chunks_to_mram_directly<float>(dpu_set_t set, uint32_t nrDPUs, size_t offset, float *data,
                                                        size_t chunkSize, size_t size);
template size_t transfer_chunks_to_mram_directly<const float>(dpu_set_t set, uint32_t nrDPUs, size_t offset,
                                                              const float *data, size_t chunkSize, size_t size);

template size_t transfer_full_to_mram_directly<float>(dpu_set_t set, uint32_t nrDPUs, size_t offset, float *data,
                                                      size_t size);
template size_t transfer_full_to_mram_directly<const float>(dpu_set_t set, uint32_t nrDPUs, size_t offset,
                                                            const float *data, size_t size);

template void transfer_full_to_mram<uint32_t>(dpu_set_t set, const char *symbol, uint32_t *data, size_t size);
template void transfer_full_to_mram<uint8_t>(dpu_set_t set, const char *symbol, uint8_t *data, size_t size);