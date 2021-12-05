//OpenCL Matrix Multiplication
//Alex Chacko
//CPEG655

__kernel void matrixMultiply(const int M, const int N, const int K, __global const double* A, __global const double* B, __global double *C){
	//Thread IDs
	const int threadRow = get_global_id(0); //Row ID of C
	const int threadCol = get_global_id(1); //Col ID of C

	//single element
	double accumulator = 0.0;
	for (int k = 0; k < K; k++){
		accumulator += A[k*M+threadRow] * B[threadCol*K +k];
	}
	C[threadCol*M+threadRow]=accumulator;
}