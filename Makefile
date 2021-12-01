#Makefile
CC=gcc main.c -lOpenCL -lm
all:
	$(CC) -o vectorAdd
exl:
	./vectorAdd
exw:
	.\vectorAdd.exe