#include "common.hpp"
#include "dpu_transfer_helper.hpp"
#include "gemv_kernel.hpp"
#include "kernel.hpp"

extern "C" {
int gemv_int8(uint32_t m, uint32_t n, const int8_t *A, const int8_t *x, int *y, const int *alpha, const int *beta) {
  GEMV_INT8_Kernel kernel;
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