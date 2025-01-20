#include "dpu_transfer_helper.hpp"

#include "common.hpp"


#define PARAM_COUNT 4
#define VECTOR_LEN_POS 0



extern "C" {

int alignUpTo8(int value) {
    return (value + 7) & ~7;
}

int alignAny(int value, int alignment) {
    return value % alignment ? value +(alignment - (value % alignment)) : value;
}

void broadcast_mram(dpu_set_t set, const char *symbol, int *data, size_t size) {
    DPU_ASSERT(dpu_broadcast_to(set, symbol, 0, data, size, DPU_XFER_DEFAULT));
}

void get_chunk_size(dpu_set_t set, int vector_len, int &split_size) {
    uint32_t nr_dpus = 0;
    DPU_ASSERT(dpu_get_nr_dpus(set, &nr_dpus));
    
    // Lets split our memory as evenly as we can between N dpus, while having each chunk even in size
    split_size = vector_len / nr_dpus;

    // If the vector length is not divisible by the number of dpus, we need to add 1 to the split size
    if (vector_len % nr_dpus != 0) {
        split_size += 1;
    }
    // If the split size is not even, we need to align it to the nearest even number
    if (split_size % 2 != 0) {
        split_size += 1;
    }
}

void to_mram(dpu_set_t set, const char *symbol, float *data, size_t len) {
    uint32_t nr_dpus = 0;
    DPU_ASSERT(dpu_get_nr_dpus(set, &nr_dpus));
    int split_size = 0;
    get_chunk_size(set, len, split_size);

    dpu_set_t dpu;
    int dpuid;
    DPU_FOREACH(set, dpu, dpuid) {
        DPU_ASSERT(
            dpu_copy_to(
                dpu,
                symbol,
                0,
                data+(dpuid*split_size),
                split_size*sizeof(uint32_t)
            )
        );
        printf("DPU %d is getting data from: %d to %d\n", dpuid, (int)(dpuid*split_size*sizeof(uint32_t)), (int)(dpuid*split_size*sizeof(uint32_t)));
    }
}

void from_mram(dpu_set_t set, const char *symbol, float *data, size_t len) {
    uint32_t nr_dpus = 0;
    DPU_ASSERT(dpu_get_nr_dpus(set, &nr_dpus));
    int split_size = 0;
    get_chunk_size(set, len, split_size);

    float* buffer = new float[nr_dpus*split_size];

    printf("Split size: %d\n", split_size);
    dpu_set_t dpu;
    int dpuid;
    DPU_FOREACH(set, dpu, dpuid) {
        DPU_ASSERT(
            dpu_copy_from(
                dpu,
                symbol,
                0,
                buffer+(dpuid*split_size),
                split_size*sizeof(uint32_t)
            )
        );
    }
    memcpy(data, buffer, len*sizeof(float));
    delete[] buffer;
}

void set_params(dpu_set_t dpu, uint32_t chunk_len) {
    std::vector<int> params(PARAM_COUNT, 0);
    params[VECTOR_LEN_POS] = chunk_len;
    broadcast_mram(dpu, "params", params.data(), PARAM_COUNT*sizeof(int));
}

int relu_f(float* input, float* output, size_t size){
    uint32_t num_of_DPUs = 2;
    dpu_set_t set;

    /*
    Allocate DPUs and loads kernel
    */
   DPU_ASSERT(dpu_alloc(num_of_DPUs, nullptr, &set));

   char *kernName = pimblas_get_kernel_dir_concat_free("relu_f.kernel");
   show_debug("kern_path = {}", kernName);
   DPU_ASSERT(dpu_load(set, kernName, nullptr));
   free(kernName);


   int chunk_size;
   get_chunk_size(set, size, chunk_size);
   printf("Chunk size: %d\n", chunk_size);
   set_params(set, chunk_size);

   to_mram(set, "buffer", input, size);
   DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

   dpu_set_t dpu;
   DPU_FOREACH(set, dpu) {
    DPU_ASSERT(dpu_log_read(dpu, stdout));
   }

   from_mram(set, "buffer", output, size);

   return 0;

}

}

