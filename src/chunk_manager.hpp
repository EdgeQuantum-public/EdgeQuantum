#pragma once
#include <vector>
#include <cuda_runtime.h>
#include "io_backend.hpp"
#include "utils.hpp"

class ChunkManager {
    size_t chunk_size;
    size_t n_chunks;
    int n_buffers;
    void** host_buffer;

    void* block[2];

public:

    ChunkManager(size_t chunk_bytes, size_t num_chunks, int num_buffers = 3);
    ~ChunkManager();

    void* get_buffer(int idx);

    void* get_pair_buffer() { return block[0]; }

    void* get_pair_slot(int slot) { return (char*)block[0] + slot * chunk_size; }
    int get_num_buffers() const { return n_buffers; }
    void read_chunk(IOBackend& io, int chunk_idx, int buf_idx);
    void write_chunk(IOBackend& io, int chunk_idx, int buf_idx);
    size_t get_chunk_size() const { return chunk_size; }
};
