#Makefile
CC=gcc
vecAdd:
	$(CC) vectorAdd.c -I./include -L./lib -lOpenCL -o vectorAdd
matMult:
	$(CC) matrixMult.c -I./include -L./lib -lOpenCL -o matrixMult