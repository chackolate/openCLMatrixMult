//OpenCL Matrix Multiplication
//Alex Chacko
//CPEG655


//basic implementation, only global memory
__kernel void mult(const int N, const __global double* A, const __global double* B, __global double* C){
	//Thread IDs
	const int globalRow = get_global_id(0); //Row ID of C
	const int globalCol = get_global_id(1); //Col ID of C

	//single element
	double accumulator = 0.0;
	for (int k = 0; k < N; k++){
		accumulator += A[k*N+globalRow] * B[globalCol*N +k];
	}
	C[globalCol*N+globalRow]=accumulator;
}

//use tiled local memory to speed up multiplication
__kernel void mult2(const int N, const __global double* A, const __global double * B, __global double* C){
	const int threadSize = 16;
	//thread IDs
	const int row = get_local_id(0);//LOCAL row ID
	const int col = get_local_id(1);//LOCAL col ID
	const int globalRow = threadSize * get_group_id(0)+row; //GLOBAL row ID
	const int globalCol = threadSize*get_group_id(1)+col;//GLOBAL col ID

	//local memory tiles
	__local double As[threadSize][threadSize];
	__local double Bs[threadSize][threadSize];

	double accumulator = 0.0;
	const int numTiles = N/threadSize;
	for(int i = 0; i < numTiles; i++){
		//load tile into local memory
		const int tileRow = threadSize * i + row;
		const int tileCol = threadSize * i + col;
		As[col][row] = A[tileCol*N+globalRow];
		Bs[col][row] = B[globalCol*N + tileRow];

		//synchronize
		barrier(CLK_LOCAL_MEM_FENCE);

		//single tile
		for(int j = 0; j < threadSize; j++){
			accumulator += As[j][row]*Bs[col][j];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	C[globalCol*N+globalRow] = accumulator;
}