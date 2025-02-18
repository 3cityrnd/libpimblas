from dpu import DpuSet
from io import StringIO
from sys import stdout

DPU_PROGRAM = '''
    #include <mram.h>
    #include <stdint.h>
    #include <stdio.h>

    __mram uint64_t my_var; // Initialized by the host application

    int main() {
        uint64_t data = my_var;
        printf("My_Var before = 0x%016lx\\n", data);

        my_var = data + 1;

        return 0;
    }
'''

with DpuSet(nr_dpus=1, c_source=StringIO(DPU_PROGRAM)) as dpu:
    dpu.my_var = bytearray([0, 1, 2, 3, 4, 5, 6, 7])
    dpu.exec(log=stdout)
    value = dpu.my_var.uint64()
    print('My_Var after = 0x%016x' % value)