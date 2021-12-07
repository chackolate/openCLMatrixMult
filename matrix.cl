//OpenCL Matrix Multiplication
//Alex Chacko
//CPEG655

__kernel void matrixMultiply(const int N, __global const double* A, __global const double* B, __global double *C){
	//Thread IDs
	const int threadRow = get_global_id(0); //Row ID of C
	const int threadCol = get_global_id(1); //Col ID of C

	//single element
	double accumulator = 0.0;
	for (int k = 0; k < N; k++){
		accumulator += A[k*N+threadRow] * B[threadCol*N +k];
	}
	C[threadCol*N+threadRow]=accumulator;
}