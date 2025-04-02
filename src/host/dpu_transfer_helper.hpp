#pragma once
#include <cstddef>
#include <cstdint>

#include "common.hpp"

template <typename T>
T alignUp(T value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

template <typename T>
T alignDown(T value, size_t alignment) {
  return value & ~(alignment - 1);
}

template <typename T>
size_t transfer_chunks(dpu_set_t set, uint32_t nr_dpus, dpu_xfer_t xfer, dpu_xfer_flags_t flags,
                       const char *symbol_name, size_t sym_offset, T *data, size_t chunk_size, size_t size) {
  bool has_remainder = size % chunk_size != 0;

  dpu_set_t dpu;
  uint32_t dpu_idx;
  DPU_FOREACH(set, dpu, dpu_idx) {
    auto offset = dpu_idx * chunk_size;
    if (false == (has_remainder && dpu_idx + 1 == nr_dpus)) {
      DPU_ASSERT(dpu_prepare_xfer(dpu, (void *)&data[offset]));
    }
  }
  DPU_ASSERT(dpu_push_xfer(set, xfer, symbol_name, sym_offset, chunk_size * sizeof(T), flags));

  if (has_remainder) {
    auto offset = (nr_dpus - 1) * chunk_size;
    auto remainder = size - offset;
    DPU_ASSERT(dpu_prepare_xfer(dpu, (void *)&data[offset]));
    DPU_ASSERT(dpu_push_xfer(dpu, xfer, symbol_name, sym_offset, alignUp(remainder * sizeof(T), 8), flags));
  }

  return sym_offset + alignUp(chunk_size * sizeof(T), 8);
}

size_t safe_gather(dpu_set_t set, uint32_t nr_dpus, const char *symbol_name, size_t symbol_offset, uint8_t *data,
                   size_t chunk_size, size_t size);

template <typename T>
size_t transfer_full(dpu_set_t set, dpu_xfer_flags_t flags, const char *symbol_name, size_t sym_offset, T *data,
                     size_t size) {
  size_t copySize = alignUp(size * sizeof(T), 8);
  DPU_ASSERT(dpu_broadcast_to(set, symbol_name, sym_offset, data, copySize, flags));
  return sym_offset + copySize;
}

template <typename T>
size_t transfer_full_exact(dpu_set_t set, dpu_xfer_flags_t flags, const char *symbol_name, size_t sym_offset, T *data,
                           size_t size) {
  DPU_ASSERT(dpu_broadcast_to(set, symbol_name, sym_offset, data, size, flags));
  return sym_offset + size;
}

template <typename T>
void gemv_launch_statistics(uint32_t m, uint32_t n, uint32_t &numDPUs, uint32_t &rowsPerDPU);