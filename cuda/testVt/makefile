cc = nvcc
prom = Vt
link = -lcudart -lcublas -lcusolver
src = ../cirMM.cu ../lsqr.cu initialY.cu testVt.cpp

$(prom): $(src)
	$(cc) $(src) $(link) -o $(prom)
