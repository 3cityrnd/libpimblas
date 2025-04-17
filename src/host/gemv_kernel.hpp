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

 private:
  uint32_t m;
  uint32_t n;
  uint32_t rows_per_dpu;

  size_t A_offset;
  size_t x_offset;
  size_t y_offset;
};

class GEMV_INT8_Kernel : public Kernel {
  struct params {
    uint32_t rows_per_dpu;
    uint32_t row_size;
    int alpha;
    int beta;
  };

 public:
  void set_A(const int8_t *data, bool async);

  void set_x(const int8_t *data, bool async);

  void set_y(const int32_t *data, bool async);

  void get_y(int32_t *data, bool async);

  void get_y_safe(int32_t *data);

  void set_params(const int32_t *alpha, const int32_t *beta, bool async);

  void init(uint32_t m, uint32_t n);
  bool init(uint32_t m, uint32_t n, uint32_t nr_dpus, uint32_t rows_per_dpu);

 private:
  uint32_t m;
  uint32_t n;
  uint32_t rows_per_dpu;

  size_t A_offset;
  size_t x_offset;
  size_t y_offset;
};

class GEMV_INT32_Kernel : public Kernel {
  struct params {
    uint32_t rows_per_dpu;
    uint32_t row_size;
    int32_t alpha;
    int32_t beta;
  };

 public:
  void set_A(const int32_t *data, bool async);

  void set_x(const int32_t *data, bool async);

  void set_y(const int32_t *data, bool async);

  void get_y(int32_t *data, bool async);

  void get_y_safe(int32_t *data);

  void set_params(const int32_t *alpha, const int32_t *beta, bool async);

  void init(uint32_t m, uint32_t n);
  bool init(uint32_t m, uint32_t n, uint32_t nr_dpus, uint32_t rows_per_dpu);

 private:
  uint32_t m;
  uint32_t n;
  uint32_t rows_per_dpu;

  size_t A_offset;
  size_t x_offset;
  size_t y_offset;
};
