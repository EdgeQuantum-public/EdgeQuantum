# EdgeQuantum
**edgeQsim: Quantum Circuit Simulation Framework for Resource-Constrained Edge Devices** (Submitted IoTJ)

## Overview
EdgeQuantum is a scalable full state vector quantum circuit simulator designed for resource-constrained IoT edge devices such as the NVIDIA Jetson Orin Nano.
To overcome the memory bottleneck that limits in-memory simulation to small qubit counts, EdgeQuantum introduces:
- **A unified tiered-memory architecture** that integrates GPU VRAM, CPU DRAM, and NVMe SSDs into a streaming framework, partitioning the full state vector into chunks and managing their residency via UVM zero-copy buffers.
- **An asynchronous 4-buffer pipeline** that overlaps NVMe I/O with GPU computation through dynamic `cudaStreamAttachMemAsync` access control, hiding storage latency behind gate execution.
- **A mantissa-aware LZ4 compression pipeline** combining mantissa truncation and byte-shuffle pre-filtering to achieve up to 242.7x storage reduction while maintaining simulation fidelity.

The framework is implemented as a standalone C++ application with approximately 1,200 lines of core code.
Our evaluation demonstrates that EdgeQuantum executes up to **37-qubit simulations** (~1 TB state vector) on an 8 GB NVIDIA Jetson Orin Nano ($200, 15W), achieving a **128x capacity expansion** beyond device RAM and up to **242.7x storage reduction** through compression.

## Key Features
- **128x Memory Capacity Expansion**: Extends simulation scale from 26 qubits (in-memory) to 37 qubits (1 TB state vector) by offloading state chunks to NVMe storage.
- **UVM Zero-Copy Data Path**: Exploits Jetson's Unified Memory Architecture to enable direct `pread`/`pwrite` on UVM buffers, eliminating `cudaMemcpy` overhead and achieving 2x speedup over pinned-memory approaches.
- **Cross-Chunk Gate Execution**: Supports quantum gates spanning chunk boundaries by streaming chunk pairs into contiguous UVM pair buffers and invoking cuStateVec on the combined state.
- **Mantissa Truncation + Byte Shuffle**: Zeros lower mantissa bits and transposes byte lanes to create LZ4-friendly patterns, boosting compression from ~1x to 3-45x on dense quantum states.

## Key Components
- **Chunk Manager (III-A)**: Allocates UVM buffers with `cudaMemAttachHost` for async I/O and a contiguous pair buffer for cross-chunk gate operations. Manages the mapping between logical state vector indices and physical NVMe offsets.
- **Asynchronous Pipeline (III-C)**: Orchestrates a 4-buffer ring pipeline across Load, Compute, and Store stages. Dynamically switches UVM access modes between `AttachHost` (CPU I/O) and `AttachGlobal` (GPU compute) to prevent data races without explicit memory copies.
- **Compressed Storage (III-D)**: Implements ping-pong dual-file variable-size compression. Each write cycle applies mantissa truncation, byte-shuffle transposition, and LZ4 compression before storing to NVMe. Offset tables track variable-size chunks for random access.
- **Cross-Chunk Gate Handler (III-E)**: For gates on global qubits (index >= chunk_bits), identifies chunk pairs differing in the target bit, loads both into a contiguous 512 MB pair buffer, and invokes cuStateVec with `nBits = chunk_bits + 1`.
