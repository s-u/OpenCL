// No input
__kernel void linear(__global int* output,
                     const unsigned int count)
{
	size_t i = get_global_id(0);
	output[i] = i;
}

// One input array
__kernel void square(__global numeric* output,
                     const unsigned int count,
                     __global const numeric* input)
{
	size_t i = get_global_id(0);
	output[i] = input[i] * input[i];
}

// Additional scalar argument
__kernel void multiply(__global numeric* output,
                       const unsigned int count,
                       __global const numeric* input,
                       const numeric factor)
{
	size_t i = get_global_id(0);
	output[i] = factor * input[i];
}
