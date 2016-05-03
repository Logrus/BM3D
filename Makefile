TMP=$(shell which nvcc)
CUDA_INCLUDE=$(shell echo ${TMP} | sed "s/\/bin\/nvcc/\/include\//g")

all: Makefile main.cpp
	echo ${CUDA_INCLUDE}
	g++ -std=c++11 -Wall -I. -I${CUDA_INCLUDE} main.cpp -o main -L/usr/X11R6/lib -lm -lpthread -lX11 -O3

clear:
	rm main

