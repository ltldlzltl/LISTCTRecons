NVCC=nvcc
CC=g++

INCLUDE=./Include/
SRC=./Src/
CUDA=./CUDA/

CFLAGS=-I./Include/ -I/usr/local/cuda/include/ -c

objects = Lib/main.o Lib/Common.o Lib/DataReader.o Lib/DataStructs.o \
Lib/FileParams.o Lib/Filter.o Lib/ImageWriter.o Lib/Reconstructor.o \
Lib/ReconstructorAbstractFactory.o Lib/Backprojector.cu.o \
Lib/Filter.cu.o Lib/Rebiner.cu.o

Bin/LISTCTRecons : $(objects)
	$(NVCC) -Xcompiler -fopenmp -lm -L/usr/local/cuda/lib64 -lcudart -o Bin/LISTCTRecons $(objects)

Lib/main.o : $(SRC)main.cpp $(INCLUDE)FileParams.h $(INCLUDE)DataStructs.h \
$(INCLUDE)ReconstructorAbstractFactory.h $(INCLUDE)Reconstructor.h \
$(INCLUDE)ImageWriter.h
	$(CC) $(CFLAGS) $(SRC)main.cpp -o Lib/main.o

Lib/Common.o : $(SRC)Common.cpp $(INCLUDE)Common.h
	$(CC) $(CFLAGS) $(SRC)Common.cpp -o Lib/Common.o

Lib/DataReader.o : $(SRC)DataReader.cpp $(INCLUDE)DataReader.h $(INCLUDE)DataStructs.h
	$(CC) $(CFLAGS) $(SRC)DataReader.cpp -o Lib/DataReader.o

Lib/DataStructs.o : $(SRC)DataStructs.cpp $(INCLUDE)DataStructs.h $(INCLUDE)Common.h \
$(INCLUDE)FileParams.h $(INCLUDE)GPUMemoryCtrl.h
	$(CC) $(CFLAGS) $(SRC)DataStructs.cpp -o Lib/DataStructs.o

Lib/FileParams.o : $(SRC)FileParams.cpp $(INCLUDE)FileParams.h $(INCLUDE)Common.h
	$(CC) $(CFLAGS) $(SRC)FileParams.cpp -o Lib/FileParams.o

Lib/Filter.o : $(SRC)Filter.cpp $(INCLUDE)Filter.h $(INCLUDE)DataStructs.h \
$(INCLUDE)Common.h $(SRC)Filter.cpp
	$(CC) $(CFLAGS) -fopenmp $(SRC)Filter.cpp -o Lib/Filter.o

Lib/ImageWriter.o : $(SRC)ImageWriter.cpp $(INCLUDE)ImageWriter.h $(INCLUDE)DataStructs.h
	$(CC) $(CFLAGS) $(SRC)ImageWriter.cpp -o Lib/ImageWriter.o

Lib/Reconstructor.o : $(SRC)Reconstructor.cpp $(INCLUDE)Reconstructor.h $(INCLUDE)Common.h \
$(INCLUDE)DataStructs.h $(INCLUDE)Rebiner.h $(INCLUDE)Filter.h \
$(INCLUDE)Backprojector.h $(INCLUDE)DataReader.h
	$(CC) $(CFLAGS) $(SRC)Reconstructor.cpp -o Lib/Reconstructor.o

Lib/ReconstructorAbstractFactory.o : $(SRC)ReconstructorAbstractFactory.cpp \
$(INCLUDE)Reconstructor.h $(INCLUDE)ReconstructorAbstractFactory.h
	$(CC) $(CFLAGS) $(SRC)ReconstructorAbstractFactory.cpp -o Lib/ReconstructorAbstractFactory.o

Lib/Backprojector.cu.o : $(CUDA)Backprojector.cu $(INCLUDE)Backprojector.h \
$(INCLUDE)DataStructs.h
	$(NVCC) $(CFLAGS) $(CUDA)Backprojector.cu -o Lib/Backprojector.cu.o

Lib/Filter.cu.o : $(CUDA)Filter.cu $(INCLUDE)Filter.h $(INCLUDE)DataStructs.h \
$(INCLUDE)Common.h
	$(NVCC) $(CFLAGS) $(CUDA)Filter.cu -o Lib/Filter.cu.o

Lib/Rebiner.cu.o : $(CUDA)Rebiner.cu $(INCLUDE)Rebiner.h $(INCLUDE)Common.h \
$(INCLUDE)DataStructs.h
	$(NVCC) $(CFLAGS) $(CUDA)Rebiner.cu -o Lib/Rebiner.cu.o

clean:
	rm Bin/LISTCTRecons $(objects)
