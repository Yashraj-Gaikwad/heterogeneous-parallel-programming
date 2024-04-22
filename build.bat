cls
del *.exe
del *.exp
del *.lib
nvcc -o MatMul.exe MatMul.cu
MatMul.exe

