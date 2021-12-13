//OpenCL Matrix Multiplication
//Alex Chacko
//CPEG655

__kernel void mult(const int N, const __global double* A, const __global double* B, __global double* C){
	//Thread IDs
	const int globalRow = get_global_id(0); //Row ID of C
	const int globalCol = get_global_id(1); //Col ID of C

	//single element
	double accumulator = 0.0f;
	for (int k = 0; k < N; k++){
		accumulator += A[k*N+globalRow] * B[globalCol*N +k];
	}
	C[globalCol*N+globalRow]=accumulator;
}

__kernel void mult1(const int N, const __global float* A, const __global float * B, __global float* C){
	const int threadSize = 16;
	//thread IDs
	const int row = get_local_id(0);//LOCAL row ID
	const int col = get_local_id(1);//LOCAL col ID
	const int globalRow = threadSize * get_group_id(0)+row; //GLOBAL row ID
	const int globalCol = threadSize*get_group_id(1)+col;//GLOBAL col ID
}