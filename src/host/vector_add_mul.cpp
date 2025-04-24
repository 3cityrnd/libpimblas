#include "common.hpp"
#include "dpu_transfer_helper.hpp"
#include "helper.hpp"

#define PARAM_COUNT 4
#define VECTOR_LEN_POS 0
#define OP_TYPE_POS 1
#define VEC_ADD 1
#define VEC_MUL 2
#define VEC_SUB 3

extern "C" {
int vec_add_mul_f(const float *input_a, const float *input_b, float *output, int OP_TYPE, size_t size, int num_dpus);

void set_params_add_mul(dpu_set_t set, uint32_t chunk_len, int op_type) {
  std::vector<int> params(PARAM_COUNT, 0);
  params[VECTOR_LEN_POS] = chunk_len;
  params[OP_TYPE_POS] = op_type;
  broadcast_mram2(set, "params", params.data(), PARAM_COUNT * sizeof(int));
}

int vec_add_f(const float *input_a, const float *input_b, float *output, size_t size) {
  uint32_t num_of_DPUs = 64;
  return vec_add_mul_f(input_a, input_b, output, VEC_ADD, size, num_of_DPUs);
}

int vec_mul_f(const float *input_a, const float *input_b, float *output, size_t size) {
  uint32_t num_of_DPUs = 64;
  return vec_add_mul_f(input_a, input_b, output, VEC_MUL, size, num_of_DPUs);
}

int vec_sub_f(const float *input_a, const float *input_b, float *output, size_t size) {
  uint32_t num_of_DPUs = 64;
  return vec_add_mul_f(input_a, input_b, output, VEC_SUB, size, num_of_DPUs);
}

int vec_add_mul_f(const float *input_a, const float *input_b, float *output, int OP_TYPE, size_t size, int num_dpus) {
  dpu_set_t set;
  DPU_ASSERT(dpu_alloc(num_dpus, nullptr, &set));

  char *kernName = pimblas_get_kernel_dir_concat_free("vector_add_mul.kernel");
  show_debug("kern_path = {} ", kernName);
  DPU_ASSERT(dpu_load(set, kernName, nullptr));
  free(kernName);

  int chunk_size = 0;
  get_chunk_size2(set, size, chunk_size, sizeof(float));
  set_params_add_mul(set, chunk_size, OP_TYPE);

  to_mram2(set, "buffer_a", input_a, size);
  to_mram2(set, "buffer_b", input_b, size);

  DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

  from_mram2(set, "buffer_a", output, size);
  DPU_ASSERT(dpu_free(set));
  return 0;
}
}
