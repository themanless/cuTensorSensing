cc = nvcc
prom = finaltest
link = -g -G -lcudart -lcublas -lcusolver
src = warmup.cu ../cirMM.cu ../lsqr.cu initialY.cu finaltest.cpp

$(prom): $(src)
	$(cc) $(src) $(link) -o $(prom)
