#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__kernel void min_reduction(
    __global const float* input,
    __global float* output,
    const uint n)
{
    const uint i = get_global_id(0);
    const uint localI = get_sub_group_local_id();

    const float val = (i < n) ? input[i] : FLT_MAX;

    const float mn = sub_group_reduce_min(val);

    if (localI == 0) {
        output[get_sub_group_id() + get_group_id(0) * get_num_sub_groups()] = mn;
    }
}