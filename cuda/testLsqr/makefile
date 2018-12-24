cc = nvcc
prom = lsqr
link = -lcudart -lcublas -lcusolver
src = ../lsqr.cu testlsqr.cpp

$(prom): $(src)
	$(cc) $(src) $(link) -o $(prom)
