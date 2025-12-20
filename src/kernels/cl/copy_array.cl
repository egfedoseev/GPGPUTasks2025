#include "helpers/rassert.cl"
#include "../defines.h"

__kernel void copy_array(
    __global const float* a,
    __global       float* b, 
    unsigned int n)
{
    const uint x = get_global_id(0);
    if (x < n) {
        b[x] = a[x];
    }
}