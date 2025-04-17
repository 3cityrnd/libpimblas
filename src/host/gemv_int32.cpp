#include "dpu_transfer_helper.hpp"
#include "gemv_kernel.hpp"

extern "C" {
int gemv_int32(uint32_t m, uint32_t n, const int *A, const int *x, int *y, const int *alpha, const int *beta) {
  GEMV_INT32_Kernel kernel;
  kernel.init(m, n);
  kernel.set_params(alpha, beta, false);
  kernel.set_A(A, true);
  kernel.set_x(x, true);
  if (beta != 0) {
    kernel.set_y(y, true);
  }
  kernel.launch(true);
  kernel.get_y(y, true);
  kernel.sync();
  kernel.free_dpus();
  return 0;
}
}