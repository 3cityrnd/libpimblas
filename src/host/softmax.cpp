#include "common.hpp"
#include "kernel.hpp"

void set_vec_size(Kernel &kernel, size_t chunk_size, size_t size) {
  uint32_t vec_size = static_cast<uint32_t>(chunk_size);
  kernel.set_arg_broadcast_exact("vec_size", 0, &vec_size, sizeof(uint32_t), false);
  uint32_t remainder = size % chunk_size;
  if (size % chunk_size != 0) {
    dpu_set_t last_dpu;
    DPU_FOREACH(kernel.get_dpu_set(), last_dpu) {}
    dpu_copy_to(last_dpu, "vec_size", 0, &remainder, sizeof(uint32_t));
  }
}

float get_max_value(Kernel &softmax) {
  std::vector<float> max_values(softmax.get_nr_dpus());
  softmax.get_arg_copy_each("max", 0, max_values.data(), sizeof(float));
  float max_val = std::numeric_limits<float>::min();
  for (auto val : max_values) {
    if (val > max_val) {
      max_val = val;
    }
  }
  return max_val;
}

float get_global_sum(Kernel &softmax) {
  std::vector<float> sums(softmax.get_nr_dpus());
  softmax.get_arg_copy_each("sum", 0, sums.data(), sizeof(float));

  float sum = 0.0f;
  for (auto val : sums) {
    sum += val;
  }
  return sum;
}

int softmax_impl(const float *vec_in, float *vec_out, size_t chunk_size, size_t size) {
  Kernel softmax;

  uint32_t nr_dpus = (size - 1) / chunk_size + 1;
  show_trace("softmax: Allocating nr_dpus=[{}]", nr_dpus);
  if (false == softmax.allocate_n(nr_dpus)) {
    show_error("softmax: Couldn't allocate nr_dpus=[{}]", nr_dpus);
    return -1;
  }

  softmax.load_program("softmax_f.kernel");

  softmax.set_arg_scatter(DPU_MRAM_HEAP_POINTER_NAME, 0, vec_in, chunk_size * sizeof(float), size * sizeof(float),
                          false);
  set_vec_size(softmax, chunk_size, size);

  uint32_t op = 0;
  softmax.set_arg_broadcast_exact("op", 0, &op, sizeof(uint32_t), false);

  softmax.launch(false);
  float max_val = get_max_value(softmax);

  op = 1;
  softmax.set_arg_broadcast_exact("op", 0, &op, sizeof(uint32_t), false);
  softmax.set_arg_broadcast_exact("global_max", 0, &max_val, sizeof(float), false);

  softmax.launch(false);
  float global_sum = get_global_sum(softmax);

  op = 2;
  softmax.set_arg_broadcast_exact("op", 0, &op, sizeof(uint32_t), false);
  softmax.set_arg_broadcast_exact("divisor", 0, &global_sum, sizeof(float), false);

  softmax.launch(false);

  softmax.get_arg_gather(DPU_MRAM_HEAP_POINTER_NAME, 0, reinterpret_cast<void *>(vec_out), chunk_size * sizeof(float),
                         size * sizeof(float), false);

  softmax.free_dpus();

  return 0;
}

extern "C" {
int softmax(const float *vec_in, float *vec_out, size_t size) {
  show_trace("softmax vec_in=[{}] vec_out=[{}] size=[{}]", reinterpret_cast<const uintptr_t>(vec_in),
             reinterpret_cast<const uintptr_t>(vec_out), size);
  size_t chunk_size = 8192;
  return softmax_impl(vec_in, vec_out, chunk_size, size);
}
}