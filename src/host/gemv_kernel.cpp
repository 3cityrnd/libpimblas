#include "gemv_kernel.hpp"

#include "dpu_transfer_helper.hpp"

void GEMVF_Kernel_Beta::set_A(const float *data, bool async) {
  set_arg_scatter(DPU_MRAM_HEAP_POINTER_NAME, A_offset, data, rows_per_dpu * n * sizeof(float), m * n * sizeof(float),
                  async);
}

void GEMVF_Kernel_Beta::set_x(const float *data, bool async) {
  set_arg_broadcast(DPU_MRAM_HEAP_POINTER_NAME, x_offset, data, n * sizeof(float), async);
}

void GEMVF_Kernel_Beta::set_y(const float *data, bool async) {
  set_arg_scatter(DPU_MRAM_HEAP_POINTER_NAME, y_offset, data, rows_per_dpu * sizeof(float), m * sizeof(float), async);
}

void GEMVF_Kernel_Beta::get_y(float *data, bool async) {
  get_arg_gather(DPU_MRAM_HEAP_POINTER_NAME, y_offset, data, rows_per_dpu * sizeof(float), m * sizeof(float), async);
}

void GEMVF_Kernel_Beta::get_y_safe(float *data) {
  get_arg_gather_safe(DPU_MRAM_HEAP_POINTER_NAME, y_offset, data, rows_per_dpu * sizeof(float), m * sizeof(float));
}

void GEMVF_Kernel_Beta::set_params(const float *alpha, const float *beta, bool async) {
  params args{.rows_per_dpu = this->rows_per_dpu, .row_size = n, .alpha = *alpha, .beta = *beta};
  this->set_arg_broadcast_exact("args", 0, reinterpret_cast<uint8_t *>(&args), sizeof(params), async);
}

void GEMVF_Kernel_Beta::init(uint32_t m, uint32_t n) {
  this->nr_dpus = 64;
  gemv_launch_statistics<float>(m, n, this->nr_dpus, this->rows_per_dpu);
  this->init(m, n, nr_dpus, rows_per_dpu);
}

bool GEMVF_Kernel_Beta::init(uint32_t m, uint32_t n, uint32_t nr_dpus, uint32_t rows_per_dpu) {
  this->m = m;
  this->n = n;
  this->nr_dpus = nr_dpus;
  this->rows_per_dpu = rows_per_dpu;

  if (dpu_alloc(this->nr_dpus, nullptr, &this->dpu_set) != DPU_OK) {
    return false;
  }

  this->load_program("gemv_f_y.kernel");

  A_offset = 0;
  x_offset = alignUp(rows_per_dpu * n * sizeof(float), 8);
  y_offset = x_offset + alignUp(n * sizeof(float), 8);

  return true;
}

void GEMVF_Kernel::set_A(const float *data, bool async) {
  set_arg_scatter(DPU_MRAM_HEAP_POINTER_NAME, A_offset, data, rows_per_dpu * n * sizeof(float), m * n * sizeof(float),
                  async);
}

void GEMVF_Kernel::set_x(const float *data, bool async) {
  set_arg_broadcast(DPU_MRAM_HEAP_POINTER_NAME, x_offset, data, n * sizeof(float), async);
}

void GEMVF_Kernel::get_y(float *data, bool async) {
  get_arg_gather(DPU_MRAM_HEAP_POINTER_NAME, y_offset, data, rows_per_dpu * sizeof(float), m * sizeof(float), async);
}

void GEMVF_Kernel::get_y_safe(float *data) {
  get_arg_gather_safe(DPU_MRAM_HEAP_POINTER_NAME, y_offset, data, rows_per_dpu * sizeof(float), m * sizeof(float));
}

void GEMVF_Kernel::set_params(const float *alpha, bool async) {
  params args{.rows_per_dpu = this->rows_per_dpu, .row_size = n, .alpha = *alpha};
  this->set_arg_broadcast_exact("args", 0, reinterpret_cast<uint8_t *>(&args), sizeof(params), async);
}

void GEMVF_Kernel::init(uint32_t m, uint32_t n) {
  this->nr_dpus = 64;
  gemv_launch_statistics<float>(m, n, this->nr_dpus, this->rows_per_dpu);
  this->init(m, n, nr_dpus, rows_per_dpu);
}

bool GEMVF_Kernel::init(uint32_t m, uint32_t n, uint32_t nr_dpus, uint32_t rows_per_dpu) {
  this->m = m;
  this->n = n;
  this->nr_dpus = nr_dpus;
  this->rows_per_dpu = rows_per_dpu;

  if (dpu_alloc(this->nr_dpus, nullptr, &this->dpu_set) != DPU_OK) {
    return false;
  }

  this->load_program("gemv_f.kernel");

  A_offset = 0;
  x_offset = alignUp(rows_per_dpu * n * sizeof(float), 8);
  y_offset = x_offset + alignUp(n * sizeof(float), 8);

  return true;
}

void GEMV_INT8_Kernel::set_A(const int8_t *data, bool async) {
  set_arg_scatter(DPU_MRAM_HEAP_POINTER_NAME, A_offset, data, rows_per_dpu * n * sizeof(int8_t), m * n * sizeof(int8_t),
                  async);
}

void GEMV_INT8_Kernel::set_x(const int8_t *data, bool async) {
  set_arg_broadcast(DPU_MRAM_HEAP_POINTER_NAME, x_offset, data, n * sizeof(int8_t), async);
}

void GEMV_INT8_Kernel::set_y(const int32_t *data, bool async) {
  set_arg_scatter(DPU_MRAM_HEAP_POINTER_NAME, y_offset, data, rows_per_dpu * sizeof(int32_t), m * sizeof(int32_t),
                  async);
}

void GEMV_INT8_Kernel::get_y(int32_t *data, bool async) {
  get_arg_gather(DPU_MRAM_HEAP_POINTER_NAME, y_offset, data, rows_per_dpu * sizeof(int32_t), m * sizeof(int32_t),
                 async);
}

void GEMV_INT8_Kernel::get_y_safe(int32_t *data) {
  get_arg_gather_safe(DPU_MRAM_HEAP_POINTER_NAME, y_offset, data, rows_per_dpu * sizeof(int32_t), m * sizeof(int32_t));
}

void GEMV_INT8_Kernel::set_params(const int32_t *alpha, const int32_t *beta, bool async) {
  params args{.rows_per_dpu = this->rows_per_dpu, .row_size = n, .alpha = *alpha, .beta = *beta};
  this->set_arg_broadcast_exact("args", 0, reinterpret_cast<uint8_t *>(&args), sizeof(params), async);
}

void GEMV_INT8_Kernel::init(uint32_t m, uint32_t n) {
  this->nr_dpus = 64;
  gemv_launch_statistics<int32_t>(m, n, this->nr_dpus, this->rows_per_dpu);
  this->init(m, n, nr_dpus, rows_per_dpu);
}

bool GEMV_INT8_Kernel::init(uint32_t m, uint32_t n, uint32_t nr_dpus, uint32_t rows_per_dpu) {
  this->m = m;
  this->n = n;
  this->nr_dpus = nr_dpus;
  this->rows_per_dpu = rows_per_dpu;

  if (dpu_alloc(this->nr_dpus, nullptr, &this->dpu_set) != DPU_OK) {
    return false;
  }

  this->load_program("gemv_int8.kernel");

  A_offset = 0;
  x_offset = alignUp(rows_per_dpu * n * sizeof(int8_t), 8);
  y_offset = x_offset + alignUp(n * sizeof(int8_t), 8);

  return true;
}

void GEMV_INT32_Kernel::set_A(const int32_t *data, bool async) {
  set_arg_scatter(DPU_MRAM_HEAP_POINTER_NAME, A_offset, data, rows_per_dpu * n * sizeof(int32_t),
                  m * n * sizeof(int32_t), async);
}

void GEMV_INT32_Kernel::set_x(const int32_t *data, bool async) {
  set_arg_broadcast(DPU_MRAM_HEAP_POINTER_NAME, x_offset, data, n * sizeof(int32_t), async);
}

void GEMV_INT32_Kernel::set_y(const int32_t *data, bool async) {
  set_arg_scatter(DPU_MRAM_HEAP_POINTER_NAME, y_offset, data, rows_per_dpu * sizeof(int32_t), m * sizeof(int32_t),
                  async);
}

void GEMV_INT32_Kernel::get_y(int32_t *data, bool async) {
  get_arg_gather(DPU_MRAM_HEAP_POINTER_NAME, y_offset, data, rows_per_dpu * sizeof(int32_t), m * sizeof(int32_t),
                 async);
}

void GEMV_INT32_Kernel::get_y_safe(int32_t *data) {
  get_arg_gather_safe(DPU_MRAM_HEAP_POINTER_NAME, y_offset, data, rows_per_dpu * sizeof(int32_t), m * sizeof(int32_t));
}

void GEMV_INT32_Kernel::set_params(const int32_t *alpha, const int32_t *beta, bool async) {
  params args{.rows_per_dpu = this->rows_per_dpu, .row_size = n, .alpha = *alpha, .beta = *beta};
  this->set_arg_broadcast_exact("args", 0, reinterpret_cast<uint8_t *>(&args), sizeof(params), async);
}

void GEMV_INT32_Kernel::init(uint32_t m, uint32_t n) {
  this->nr_dpus = 64;
  gemv_launch_statistics<int32_t>(m, n, this->nr_dpus, this->rows_per_dpu);
  this->init(m, n, nr_dpus, rows_per_dpu);
}

bool GEMV_INT32_Kernel::init(uint32_t m, uint32_t n, uint32_t nr_dpus, uint32_t rows_per_dpu) {
  this->m = m;
  this->n = n;
  this->nr_dpus = nr_dpus;
  this->rows_per_dpu = rows_per_dpu;

  if (dpu_alloc(this->nr_dpus, nullptr, &this->dpu_set) != DPU_OK) {
    return false;
  }

  this->load_program("gemv_int32.kernel");

  A_offset = 0;
  x_offset = alignUp(rows_per_dpu * n * sizeof(int32_t), 8);
  y_offset = x_offset + alignUp(n * sizeof(int32_t), 8);

  return true;
}
