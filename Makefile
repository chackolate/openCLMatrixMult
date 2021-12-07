#Makefile
CC=gcc
all:

vec:
	$(CC) vectorMain.c -I./include -L./lib -lOpenCL -o vecOp
mat:
	$(CC) matrixMain.c clHelper.c -I./include -L./lib -lOpenCL -o matrixOp