#Makefile
CC=gcc
LIB=-I./include -L./lib -lOpenCL
O=-o vectorAdd
all:
	$(CC) main.c -I./include -L./lib -lOpenCL -o vectorAdd
exl:
	./vectorAdd
exw:
	.\vectorAdd.exe