#Makefile
CC=gcc
all:

vec:
	$(CC) vectorAdd.c -I./include -L./lib -lOpenCL -o vectorAdd
mat:
	$(CC) matrixMult.c clHelper.c -I./include -L./lib -lOpenCL -o matrixMult