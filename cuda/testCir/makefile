cc = nvcc
prom = cir
link = -lcudart -lcublas
src = ../cirMM.cu testcirMM.cpp

$(prom): $(src)
	$(cc) $(src) $(link) -o $(prom)
