CXX := /usr/local/cuda/bin/nvcc
CXXFLAGS := -O3 -arch=sm_87 -std=c++17 -Xcompiler "-fopenmp -pthread -Wall"
CUQUANTUM_ROOT ?= /mnt/nvme/skim/edgeQuantum/third_party/cuquantum-linux-sbsa-22.03.0.40-archive
INCLUDES := -Isrc -I$(CUQUANTUM_ROOT)/include
LDFLAGS := -Lbuild -L$(CUQUANTUM_ROOT)/lib -Xlinker -rpath -Xlinker $(CUQUANTUM_ROOT)/lib -lcustatevec -llz4

SRCS := src/main.cpp src/simulator.cpp src/chunk_manager.cpp
OBJS := $(SRCS:.cpp=.o)
TARGET := build/edge_quantum

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
