NVCC=nvcc
CC=g++

INCLUDE=./Include/
SRC=./Src/
CUDA=./CUDA/

CFLAGS=-I./Include/ -I/usr/local/cuda/include/ -c -g

objects = Lib/main.o Lib/Common.o Lib/DataReader.o Lib/DataStructs.o \
Lib/FileParams.o Lib/Filter.o Lib/ImageWriter.o Lib/Reconstructor.o \
Lib/ReconstructorAbstractFactory.o Lib/Backprojector.cu.o \
Lib/Filter.cu.o Lib/Rebiner.cu.o Lib/PreWeighting.cu.o

PTXS = PTX/Backprojector.ptx PTX/Filter.ptx PTX/Rebiner.ptx

Bin/LISTCTRecons : $(objects)
	$(NVCC) -Xcompiler -fopenmp -lm -L/usr/local/cuda/lib64 -lcudart -o Bin/LISTCTRecons $(objects)
	
all : Bin/LISTCTRecons $(objects)

Lib/main.o : $(SRC)main.cpp
	$(CC) $(CFLAGS) $^ -o $@

Lib/Common.o : $(SRC)Common.cpp
	$(CC) $(CFLAGS) $^ -o $@

Lib/DataReader.o : $(SRC)DataReader.cpp
	$(CC) $(CFLAGS) $^ -o $@

Lib/DataStructs.o : $(SRC)DataStructs.cpp
	$(CC) $(CFLAGS) $^ -o $@

Lib/FileParams.o : $(SRC)FileParams.cpp
	$(CC) $(CFLAGS) $^ -o $@

Lib/Filter.o : $(SRC)Filter.cpp
	$(CC) $(CFLAGS) -fopenmp $^ -o $@

Lib/ImageWriter.o : $(SRC)ImageWriter.cpp
	$(CC) $(CFLAGS) $^ -o $@

Lib/Reconstructor.o : $(SRC)Reconstructor.cpp
	$(CC) $(CFLAGS) $^ -o $@

Lib/ReconstructorAbstractFactory.o : $(SRC)ReconstructorAbstractFactory.cpp
	$(CC) $(CFLAGS) $^ -o $@

Lib/Backprojector.cu.o : $(CUDA)Backprojector.cu
	$(NVCC) $(CFLAGS) --ptx $^ -o PTX/Backprojector.ptx
	$(NVCC) $(CFLAGS) $^ -o $@

Lib/Filter.cu.o : $(CUDA)Filter.cu
	$(NVCC) $(CFLAGS) --ptx $^ -o PTX/Filter.ptx
	$(NVCC) $(CFLAGS) $^ -o $@

Lib/Rebiner.cu.o : $(CUDA)Rebiner.cu
	$(NVCC) $(CFLAGS) --ptx $^ -o PTX/Rebiner.ptx
	$(NVCC) $(CFLAGS) $^ -o $@
	
Lib/PreWeighting.cu.o : $(CUDA)PreWeighting.cu
	$(NVCC) $(CFLAGS) $^ -o $@

clean:
	rm Bin/LISTCTRecons $(objects) $(PTXS)
