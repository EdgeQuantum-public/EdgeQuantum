# EdgeQuantum
**edgeQsim: Quantum Circuit Simulation Framework for Resource-Constrained Edge Devices** (Submitted IoTJ)

## Overview
EdgeQuantum is a full state vector quantum circuit simulator designed for resource-constrained IoT edge devices.
Its primary strategy is to extend beyond in-memory simulation bounded by physical RAM and introduce a unified tiered-memory architecture that integrates GPU VRAM, CPU DRAM, and NVMe SSDs into a streaming framework.
With this design, EdgeQuantum partitions the full state vector into chunks, manages their residency via a triple-buffered UVM zero-copy pipeline, overlaps computation with data movement through asynchronous execution, and employs on-the-fly LZ4 compression to maximize effective storage capacity.
EdgeQuantum introduces:
- **A unified tiered-memory architecture** that partitions the state vector across NVMe storage and allocates UVM buffers to act as a staging area, enabling simulation of quantum circuits that far exceed the physical memory limits of the device.
- **An asynchronous multi-buffer pipeline** that overlaps NVMe I/O with GPU computation through dynamic `cudaStreamAttachMemAsync` access control, hiding storage latency behind gate execution time.
- **A mantissa-aware LZ4 compression pipeline** combining mantissa truncation and byte-shuffle pre-filtering to maximize effective storage capacity.

## Key Features
- **Memory Capacity Expansion**: Extends simulation scale beyond physical RAM limits by utilizing NVMe storage as a backing store, seamlessly expanding the addressable state vector size.
- **UVM Zero-Copy Data Path**: Exploits the Unified Memory Architecture of edge SoCs to enable direct POSIX I/O on UVM buffers, eliminating explicit `cudaMemcpy` overhead between host and device.
- **Cross-Chunk Gate Execution**: Supports quantum gates spanning chunk boundaries by streaming chunk pairs into contiguous UVM pair buffers and invoking cuStateVec on the combined state.
- **Compression-Aware Storage**: Applies mantissa truncation and byte-lane transposition as pre-filters before LZ4 compression, significantly improving compression ratios on dense floating-point quantum states.

## Key Components
- **Chunk Manager**: Allocates UVM buffers with `cudaMemAttachHost` for async I/O and a contiguous pair buffer for cross-chunk gate operations. Manages the mapping between logical state vector indices and physical NVMe offsets.
- **Asynchronous Pipeline**: Orchestrates a multi-buffer ring pipeline across Load, Compute, and Store stages. Dynamically switches UVM access modes between `AttachHost` (CPU I/O) and `AttachGlobal` (GPU compute) to prevent data races without explicit memory copies.
- **Compressed Storage**: Implements ping-pong dual-file variable-size compression. Each write cycle applies mantissa truncation, byte-shuffle transposition, and LZ4 compression before storing to NVMe. Offset tables track variable-size chunks for random access.
- **Cross-Chunk Gate Handler**: For gates on global qubits (index >= chunk_bits), identifies chunk pairs differing in the target bit, loads both into a contiguous pair buffer, and invokes cuStateVec with extended qubit count.
