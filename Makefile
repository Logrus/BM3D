TMP=$(shell which nvcc)
CUDA_INCLUDE=$(shell echo ${TMP} | sed "s/\/bin\/nvcc/\/include\//g")

all: Makefile main.cpp
	g++ -std=c++11 -Wall -I. -I${CUDA_INCLUDE} main.cpp -o main -L/usr/X11R6/lib -lm -lpthread -lX11 `pkg-config --libs opencv` `pkg-config --cflags opencv` -fopenmp

optim: Makefile main.cpp
	g++ -std=c++11 -Wall -I. -I${CUDA_INCLUDE} main.cpp -o main -L/usr/X11R6/lib -lm -lpthread -lX11 -O3 -fopenmp `pkg-config --libs opencv` `pkg-config --cflags opencv`
clear:
	rm main

