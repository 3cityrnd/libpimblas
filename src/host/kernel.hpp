#pragma once

#include <memory>
#include <string>

#include "common.hpp"

struct KernelStatus {
  bool done;
  bool fault;
};

class Kernel {
 public:
  Kernel() = default;
  ~Kernel() = default;

  void set_arg_scatter(const char *sym_name, size_t sym_offset, const void *data, size_t chunk_size, size_t size,
                       bool async);
  void set_arg_broadcast(const char *sym_name, size_t sym_offset, const void *data, size_t size, bool async);
  void set_arg_broadcast_exact(const char *sym_name, size_t sym_offset, const void *data, size_t size, bool async);
  void get_arg_gather(const char *sym_name, size_t sym_offset, void *data, size_t chunk_size, size_t size, bool async);
  void get_arg_copy_each(const char *sym_name, size_t sym_offset, void *data, size_t size);
  void get_arg_gather_safe(const char *sym_name, size_t sym_offset, void *data, size_t chunk_size, size_t size);

  void launch(bool async);

  void load_program(const char *name);
  void load_program(uint8_t *data, size_t size);

  dpu_set_t &get_dpu_set() { return dpu_set; }
  uint32_t get_nr_dpus() { return nr_dpus; }
  void set_dpu_set(dpu_set_t dpu_set, uint32_t nr_dpus);
  bool allocate_n(uint32_t nr_dpus);

  void sync();
  const KernelStatus &get_status();

  void read_log(FILE *stream = stdout);

  void free_dpus();

 protected:
  dpu_set_t dpu_set;
  uint32_t nr_dpus;
  dpu_program_t *program;
  KernelStatus status;
};