//Basic Vector Addition A[n] + B[n] = C[n]
//Alex Chacko

__kernel void vectorAdd(__global const float *A, __global const float *B, __global float *C){
	//index of element
	int i = get_global_id(0);
	C[i] = A[i] + B[i];
}