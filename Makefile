TARGET := gemm

all: $(TARGET)

# sm_75 : Turing architecture
$(TARGET): gemm.cu
	nvcc -O3 -arch sm_75 -maxrregcount 128 gemm.cu -lopenblas -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean 
