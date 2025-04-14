#include "common.hpp"
#include "dpu_transfer_helper.hpp"
#include "gemv_kernel.hpp"

void print_output(dpu_set_t set) {
  dpu_set_t dpu;
  DPU_FOREACH(set, dpu) { dpu_log_read(dpu, stdout); }
}

extern "C" {
int gemv_f_basic(uint32_t m, uint32_t n, const float *mat, const float *vec, float *out) {
  float alpha = 1.0f;
  GEMVF_Kernel kernel;
  kernel.init(m, n);
  kernel.set_params(&alpha, false);
  kernel.set_A(mat, true);
  kernel.set_x(vec, true);
  kernel.launch(true);
  kernel.get_y(out, true);
  kernel.sync();
  kernel.free_dpus();
  return 0;
}

int gemv_f(uint32_t m, uint32_t n, const float *A, const float *x, float *y, const float *alpha, const float *beta) {
  GEMVF_Kernel_Beta kernel;
  kernel.init(m, n);
  kernel.set_params(alpha, beta, false);
  kernel.set_A(A, true);
  kernel.set_x(x, true);
  kernel.set_y(y, true);
  kernel.launch(true);
  kernel.get_y(y, true);
  kernel.sync();
  kernel.free_dpus();
  return 0;
}
}
