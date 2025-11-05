$(CXX) = mpiicpx

all: exercise01.out

exercise01.out: exercise01.o
	mpiicpx -o exercise01.out -fsycl exercise01.o

exercise01.o: exercise01.cpp
	mpiicpx -c exercise01.cpp -fsycl

make clean:
	rm *.o *.out
