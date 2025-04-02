#pragma once
#include "kernel.hpp"

class GEMVF_Kernel_Beta : public Kernel {
  struct params {
    uint32_t rows_per_dpu;
    uint32_t row_size;
    float alpha;
    float beta;
  };

 public:
  void set_A(const float *data, bool async);

  void set_x(const float *data, bool async);

  void set_y(const float *data, bool async);

  void get_y(float *data, bool async);

  void get_y_safe(float *data);

  void set_params(const float *alpha, const float *beta, bool async);

  void init(uint32_t m, uint32_t n);
  bool init(uint32_t m, uint32_t n, uint32_t nr_dpus, uint32_t rows_per_dpu);

  bool running = false;

 private:
  uint32_t m;
  uint32_t n;
  uint32_t rows_per_dpu;

  size_t A_offset;
  size_t x_offset;
  size_t y_offset;
};

class GEMVF_Kernel : public Kernel {
  struct params {
    uint32_t rows_per_dpu;
    uint32_t row_size;
    float alpha;
  };

 public:
  void set_A(const float *data, bool async);

  void set_x(const float *data, bool async);

  void get_y(float *data, bool async);

  void get_y_safe(float *data);

  void set_params(const float *alpha, bool async);

  void init(uint32_t m, uint32_t n);
  bool init(uint32_t m, uint32_t n, uint32_t nr_dpus, uint32_t rows_per_dpu);

  bool running = false;
  int cur_i = -1;

 private:
  uint32_t m;
  uint32_t n;
  uint32_t rows_per_dpu;

  size_t A_offset;
  size_t x_offset;
  size_t y_offset;
};