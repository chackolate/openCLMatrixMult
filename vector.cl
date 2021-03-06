//Basic Vector Addition A[n] + B[n] = C[n]
//Alex Chacko

__kernel void add(__global const double *A, __global const double *B, __global double *C){
	//index of element
	int i = get_global_id(0);
	C[i] = A[i] + B[i];
}

__kernel void mult(__global const double *A, __global const double *B, __global double *C){
	//index of element
	int i = get_global_id(0);
	C[i] = A[i] * B[i];
}