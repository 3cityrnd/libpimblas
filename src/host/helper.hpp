#pragma once
#include <dpu.h>

#include <cstring>  // for memcpy

#include "common.hpp"
#include "dpu_transfer_helper.hpp"

// Forward declarations
void transfer_chunks_to_mram2(dpu_set_t set, const char *symbol, float *data, size_t chunk_size, size_t size);
void transfer_chunks_from_mram2(dpu_set_t set, const char *symbol, float *data, size_t chunk_size, size_t size);

void get_chunk_size2(dpu_set_t set, int vector_len, int &split_size) {
  uint32_t nr_dpus = 0;
  DPU_ASSERT(dpu_get_nr_dpus(set, &nr_dpus));

  // Lets split out memory as evenly as we can between N DPUs, while having each chunk even in size
  split_size = vector_len / nr_dpus;

  // If the vector length is not divisible by number of dpus, we need to add 1 to the split size
  if (vector_len % nr_dpus != 0) {
    split_size++;
  }
  // if the split size in not even, we need to align it to the nearest even number
  if (split_size % 2 != 0) {
    split_size++;
  }
}

void broadcast_mram2(dpu_set_t set, const char *symbol, int *data, size_t size) {
  DPU_ASSERT(dpu_broadcast_to(set, symbol, 0, data, size, DPU_XFER_DEFAULT));
}
void to_mram2(dpu_set_t set, const char *symbol, float *data, size_t len) {
  uint32_t nr_dpus = 0;
  DPU_ASSERT(dpu_get_nr_dpus(set, &nr_dpus));

  int chunk_size = 0;
  get_chunk_size2(set, len, chunk_size);
  transfer_chunks_to_mram2(set, symbol, data, chunk_size, len);
}

void from_mram2(dpu_set_t set, const char *symbol, float *data, size_t len) {
  uint32_t nr_dpus = 0;
  DPU_ASSERT(dpu_get_nr_dpus(set, &nr_dpus));

  int split_size = 0;
  get_chunk_size2(set, len, split_size);
  float *buffer = new float[nr_dpus * split_size];
  transfer_chunks_from_mram2(set, symbol, buffer, split_size, len);
  memcpy(data, buffer, len * sizeof(float));
  delete[] buffer;
}

void transfer_chunks_to_mram2(dpu_set_t set, const char *symbol, float *data, size_t chunk_size, size_t size) {
  bool has_reminder = size % chunk_size != 0;

  uint32_t nr_dpus = 0;
  DPU_ASSERT(dpu_get_nr_dpus(set, &nr_dpus));

  has_reminder = has_reminder && (nr_dpus * chunk_size < size);
  dpu_set_t dpu;
  uint32_t dpu_idx;
  DPU_FOREACH(set, dpu, dpu_idx) {
    auto offset = dpu_idx * chunk_size;
    if (has_reminder && dpu_idx + 1 == nr_dpus) {
      size_t remainder = size - offset;
      DPU_ASSERT(
          dpu_broadcast_to(dpu, symbol, 0, &data[offset], alignUp(remainder * sizeof(float), 8), DPU_XFER_DEFAULT));
    } else {
      DPU_ASSERT(dpu_prepare_xfer(dpu, (void *)&data[offset]));
    }
  }
  DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, symbol, 0, chunk_size * sizeof(float), DPU_XFER_DEFAULT));
}

void transfer_chunks_from_mram2(dpu_set_t set, const char *symbol, float *data, size_t chunk_size, size_t size) {
  bool has_reminder = size % chunk_size != 0;
  uint32_t nr_dpus = 0;
  DPU_ASSERT(dpu_get_nr_dpus(set, &nr_dpus));

  has_reminder = has_reminder && (nr_dpus * chunk_size < size);
  dpu_set_t dpu;
  uint32_t dpu_idx;
  DPU_FOREACH(set, dpu, dpu_idx) {
    auto offset = dpu_idx * chunk_size;
    if (has_reminder && dpu_idx + 1 == nr_dpus) {
      size_t remainder = size - offset;
      DPU_ASSERT(dpu_copy_from(dpu, symbol, 0, &data[offset], alignUp(remainder * sizeof(float), 8)));
    } else {
      DPU_ASSERT(dpu_prepare_xfer(dpu, &data[offset]));
    }
  }
  DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, symbol, 0, chunk_size * sizeof(float), DPU_XFER_DEFAULT));
}