#include "kernel.hpp"

#include "dpu_transfer_helper.hpp"

void Kernel::set_arg_scatter(const char *sym_name, size_t sym_offset, const void *data, size_t chunk_size, size_t size,
                             bool async) {
  if (async) {
    transfer_chunks(dpu_set, nr_dpus, DPU_XFER_TO_DPU, DPU_XFER_ASYNC, sym_name, sym_offset,
                    reinterpret_cast<const uint8_t *>(data), chunk_size, size);
  } else {
    transfer_chunks(dpu_set, nr_dpus, DPU_XFER_TO_DPU, DPU_XFER_DEFAULT, sym_name, sym_offset,
                    reinterpret_cast<const uint8_t *>(data), chunk_size, size);
  }
}

void Kernel::set_arg_broadcast(const char *sym_name, size_t sym_offset, const void *data, size_t size, bool async) {
  if (async) {
    transfer_full(dpu_set, DPU_XFER_ASYNC, sym_name, sym_offset, reinterpret_cast<const uint8_t *>(data), size);
  } else {
    transfer_full(dpu_set, DPU_XFER_DEFAULT, sym_name, sym_offset, reinterpret_cast<const uint8_t *>(data), size);
  }
}

void Kernel::set_arg_broadcast_exact(const char *sym_name, size_t sym_offset, const void *data, size_t size,
                                     bool async) {
  if (async) {
    transfer_full_exact(dpu_set, DPU_XFER_ASYNC, sym_name, sym_offset, reinterpret_cast<const uint8_t *>(data), size);
  } else {
    transfer_full_exact(dpu_set, DPU_XFER_DEFAULT, sym_name, sym_offset, reinterpret_cast<const uint8_t *>(data), size);
  }
}

void Kernel::get_arg_gather(const char *sym_name, size_t sym_offset, void *data, size_t chunk_size, size_t size,
                            bool async) {
  if (async) {
    transfer_chunks(dpu_set, nr_dpus, DPU_XFER_FROM_DPU, DPU_XFER_ASYNC, sym_name, sym_offset,
                    reinterpret_cast<uint8_t *>(data), chunk_size, size);
  } else {
    transfer_chunks(dpu_set, nr_dpus, DPU_XFER_FROM_DPU, DPU_XFER_DEFAULT, sym_name, sym_offset,
                    reinterpret_cast<uint8_t *>(data), chunk_size, size);
  }
}

void Kernel::get_arg_copy_each(const char *sym_name, size_t sym_offset, void *data, size_t size) {
  dpu_set_t dpu;
  uint32_t idx;
  DPU_FOREACH(dpu_set, dpu, idx) {
    DPU_ASSERT(dpu_copy_from(dpu, sym_name, sym_offset, reinterpret_cast<uint8_t *>(data) + idx * size, size));
  }
}

void Kernel::launch(bool async) {
  if (async) {
    DPU_ASSERT(dpu_launch(dpu_set, DPU_ASYNCHRONOUS));
  } else {
    DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
  }
}

void Kernel::load_program(const char *name) {
  char *kernel_path = pimblas_get_kernel_dir_concat_free(name);
  show_debug("kern_path = {}", kernel_path);
  DPU_ASSERT(dpu_load(dpu_set, kernel_path, &program));
  free(kernel_path);
}

void Kernel::load_program(uint8_t *data, size_t size) {
  DPU_ASSERT(dpu_load_from_memory(dpu_set, data, size, &program));
}

void Kernel::set_dpu_set(dpu_set_t dpu_set, uint32_t nr_dpus) {
  this->dpu_set = dpu_set;
  this->nr_dpus = nr_dpus;
}

bool Kernel::allocate_n(uint32_t nr_dpus) {
  if (dpu_alloc(nr_dpus, nullptr, &this->dpu_set) != DPU_OK) {
    return false;
  }

  this->nr_dpus = nr_dpus;
  return true;
}

void Kernel::sync() { DPU_ASSERT(dpu_sync(dpu_set)); }

const KernelStatus &Kernel::get_status() {
  DPU_ASSERT(dpu_status(dpu_set, &status.done, &status.fault));
  return status;
}

void Kernel::read_log(FILE *stream) {
  dpu_set_t dpu;
  DPU_FOREACH(dpu_set, dpu) { dpu_log_read(dpu, stream); }
}

void Kernel::free_dpus() { DPU_ASSERT(dpu_free(dpu_set)); }