#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>

void mm(float * C, float * A, float * B, int N)
{
	int i,j,k;

	/* Here is the most basic matrix multiplication. */
	/* Replace it with your implementation. */
	for(j=0;j<N;j++)
		for(i=0;i<N;i++)
			for(k=0;k<N;k++)
				C[i*N+j]+=A[i*N+k]*B[k*N+j];
}

int main (int argc, char ** argv)
{

	struct timeval begin, end;
	int N=256;
	float * A, * B, * C;

	int size = N*N;
	A = (float*) malloc(size*sizeof(float));
	B = (float*) malloc(size*sizeof(float));
	C = (float*) malloc(size*sizeof(float));

	gettimeofday(&begin, NULL);
	mm(C, A, B, N);
	gettimeofday(&end, NULL);

	fprintf(stdout, "time = %lf\n", (end.tv_sec-begin.tv_sec) + (end.tv_usec-begin.tv_usec)*1.0/1000000);

	free(A);
	free(B);
	free(C);

	return 0;

}
