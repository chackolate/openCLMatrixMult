//OpenCL Matrix Multiplication
//Alex Chacko
//CPEG655

__kernel void mult(const int N, const __global float* A, const __global float* B, __global float* C){
	//Thread IDs
	const int threadRow = get_global_id(0); //Row ID of C
	const int threadCol = get_global_id(1); //Col ID of C

	//single element
	float accumulator = 0.0f;
	for (int k = 0; k < N; k++){
		accumulator += A[k*N+threadRow] * B[threadCol*N +k];
	}
	C[threadCol*N+threadRow]=accumulator;
}