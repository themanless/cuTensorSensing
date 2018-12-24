cc = nvcc
prom = one_step
link = -g -G -lcudart -lcublas -lcusolver
src = ../cirMM.cu ../lsqr.cu initialY.cu one_step.cpp

$(prom): $(src)
	$(cc) $(src) $(link) -o $(prom)
