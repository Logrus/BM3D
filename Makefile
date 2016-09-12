# Find cuda includes
TMP=$(shell which nvcc)
CUDA_INCLUDE=$(shell echo ${TMP} | sed "s/\/bin\/nvcc/\/include\//g")

all: Makefile main.cpp
	g++ -std=c++11 -Wall -I. -I${CUDA_INCLUDE} main.cpp -o main -L/usr/X11R6/lib -lm -lpthread -lX11 `pkg-config --libs opencv` `pkg-config --cflags opencv` -O3 -fopenmp

debug: Makefile main.cpp
	g++ -std=c++11 -Wall -I. -I${CUDA_INCLUDE} main.cpp -o main -L/usr/X11R6/lib -lm -lpthread -lX11 `pkg-config --libs opencv` `pkg-config --cflags opencv`

clear:
	rm main

