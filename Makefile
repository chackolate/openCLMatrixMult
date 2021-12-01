#Makefile
CC=gcc main.c -I./include/ -L./lib/ -lOpenCL -lm
all:
	$(CC) -o vectorAdd
exl:
	./vectorAdd
exw:
	.\vectorAdd.exe