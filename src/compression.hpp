#pragma once
#include <lz4.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/statvfs.h>
#include <mutex>
#include <chrono>
#include <cstdlib>

static inline int get_truncation_bits() {
    const char* env = std::getenv("TRUNC_BITS");
    if (env) {
        int v = std::atoi(env);
        if (v >= 0 && v <= 23) return v;
    }
    return 10;
}

static inline void truncate_mantissa(void* data, size_t nbytes, int keep_bits) {
    if (keep_bits >= 23) return;
    uint32_t mask = ~((1u << (23 - keep_bits)) - 1);
    uint32_t* p = (uint32_t*)data;
    size_t n = nbytes / 4;
    for (size_t i = 0; i < n; i++) {
        p[i] &= mask;
    }
}

static inline void byte_shuffle(const void* src, void* dst, size_t nbytes, int elem_size = 8) {
    const uint8_t* in = (const uint8_t*)src;
    uint8_t* out = (uint8_t*)dst;
    size_t n_elems = nbytes / elem_size;
    constexpr size_t TILE_ELEMS = 1024;

    for (size_t base = 0; base < n_elems; base += TILE_ELEMS) {
        size_t end = (base + TILE_ELEMS < n_elems) ? base + TILE_ELEMS : n_elems;
        for (size_t i = base; i < end; i++) {
            const uint8_t* elem = in + i * elem_size;
            for (int b = 0; b < elem_size; b++) {
                out[(size_t)b * n_elems + i] = elem[b];
            }
        }
    }
}

static inline void byte_unshuffle(const void* src, void* dst, size_t nbytes, int elem_size = 8) {
    const uint8_t* in = (const uint8_t*)src;
    uint8_t* out = (uint8_t*)dst;
    size_t n_elems = nbytes / elem_size;
    constexpr size_t TILE_ELEMS = 1024;

    for (size_t base = 0; base < n_elems; base += TILE_ELEMS) {
        size_t end = (base + TILE_ELEMS < n_elems) ? base + TILE_ELEMS : n_elems;
        for (size_t i = base; i < end; i++) {
            uint8_t* elem = out + i * elem_size;
            for (int b = 0; b < elem_size; b++) {
                elem[b] = in[(size_t)b * n_elems + i];
            }
        }
    }
}

class CompressedStorage {
    int fd[2];
    std::string paths[2];
    std::vector<size_t> offsets[2];
    int read_file;
    int write_file;

    size_t chunk_size;
    size_t n_chunks;
    bool compression_enabled;

    std::vector<char> compress_buf;
    std::vector<char> decompress_buf;
    std::vector<char> shuffle_buf;

    int trunc_bits;

    std::mutex io_mutex;

    size_t write_offset;
    size_t total_compressed;
    size_t total_uncompressed;

public:
    CompressedStorage(const std::string& filepath, size_t chunk_bytes, size_t num_chunks, bool enable_compression = true)
        : read_file(0),
          write_file(1),
          chunk_size(chunk_bytes),
          n_chunks(num_chunks),
          compression_enabled(enable_compression),
          write_offset(0),
          total_compressed(0),
          total_uncompressed(0) {

        paths[0] = filepath + ".A";
        paths[1] = filepath + ".B";

        int max_compressed = LZ4_compressBound(chunk_size);
        compress_buf.resize(4 + max_compressed + chunk_size);
        decompress_buf.resize(max_compressed + chunk_size);
        shuffle_buf.resize(chunk_size);

        trunc_bits = get_truncation_bits();

        offsets[0].resize(n_chunks + 1, 0);
        offsets[1].resize(n_chunks + 1, 0);

        for (int i = 0; i < 2; i++) {
            fd[i] = open(paths[i].c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
            if (fd[i] < 0) {
                perror(("open " + paths[i]).c_str());
                exit(1);
            }
        }

        std::cout << "[CompressedStorage] Ping-pong mode (variable-size compression)" << std::endl;
        std::cout << "  Files: " << paths[0] << ", " << paths[1] << std::endl;
        std::cout << "  Chunks: " << n_chunks << " x " << chunk_size/(1024*1024) << " MB" << std::endl;
        std::cout << "  Compression: " << (compression_enabled ? "ON" : "OFF")
                  << " | Mantissa: " << trunc_bits << "/23 bits"
                  << " | Shuffle: ON" << std::endl;
    }

    ~CompressedStorage() {
        for (int i = 0; i < 2; i++) {
            if (fd[i] >= 0) {
                fsync(fd[i]);
                close(fd[i]);
            }

            unlink(paths[i].c_str());
        }
    }

    CompressedStorage(const CompressedStorage&) = delete;
    CompressedStorage& operator=(const CompressedStorage&) = delete;

    bool is_compression_enabled() const { return compression_enabled; }

    void init_write_chunk(size_t chunk_idx, const void* data) {
        std::lock_guard<std::mutex> lock(io_mutex);

        if (chunk_idx >= n_chunks) {
            std::cerr << "[CompressedStorage] Invalid chunk index: " << chunk_idx << std::endl;
            return;
        }

        offsets[0][chunk_idx] = write_offset;

        size_t bytes_written;

        if (compression_enabled) {

            memcpy(shuffle_buf.data(), data, chunk_size);
            truncate_mantissa(shuffle_buf.data(), chunk_size, trunc_bits);
            byte_shuffle(shuffle_buf.data(), decompress_buf.data(), chunk_size);

            int compressed_size = LZ4_compress_default(
                decompress_buf.data(), compress_buf.data() + 4,
                chunk_size, LZ4_compressBound(chunk_size)
            );

            if (compressed_size <= 0) {
                std::cerr << "[CompressedStorage] LZ4 compression failed at init!" << std::endl;
                exit(1);
            }

            memcpy(compress_buf.data(), &compressed_size, 4);
            bytes_written = 4 + compressed_size;

            ssize_t ret = pwrite(fd[0], compress_buf.data(), bytes_written, write_offset);
            if (ret != (ssize_t)bytes_written) {
                perror("pwrite init");
                exit(1);
            }

            total_compressed += compressed_size;
            total_uncompressed += chunk_size;

            if (chunk_idx < 3 || chunk_idx == n_chunks - 1) {
                float ratio = 100.0f * compressed_size / chunk_size;
                std::cout << "[Init] Chunk " << chunk_idx
                          << ": " << chunk_size/(1024*1024) << "MB -> "
                          << compressed_size/1024 << "KB ("
                          << ratio << "%)" << std::endl;
            }
        } else {
            bytes_written = chunk_size;
            ssize_t ret = pwrite(fd[0], data, chunk_size, write_offset);
            if (ret != (ssize_t)chunk_size) {
                perror("pwrite init raw");
                exit(1);
            }
        }

        write_offset += bytes_written;
        offsets[0][chunk_idx + 1] = write_offset;
    }

    void finish_init() {
        std::lock_guard<std::mutex> lock(io_mutex);

        fsync(fd[0]);
        read_file = 0;
        write_file = 1;
        write_offset = 0;
        offsets[1][0] = 0;

        if (compression_enabled && total_uncompressed > 0) {
            float ratio = 100.0f * total_compressed / total_uncompressed;
            double reduction = (double)total_uncompressed / total_compressed;
            size_t file_size = lseek(fd[0], 0, SEEK_END);
            std::cout << "[Compression Init] File size: "
                      << file_size / (1024ULL*1024*1024) << " GB ("
                      << ratio << "% of uncompressed, " << reduction << "x reduction)" << std::endl;
        }
    }

    void read_chunk(size_t chunk_idx, void* data) {
        std::lock_guard<std::mutex> lock(io_mutex);

        if (chunk_idx >= n_chunks) {
            std::cerr << "[CompressedStorage] Invalid read index: " << chunk_idx << std::endl;
            return;
        }

        size_t file_offset = offsets[read_file][chunk_idx];

        if (compression_enabled) {

            auto t0 = std::chrono::high_resolution_clock::now();

            int compressed_size = 0;
            ssize_t ret = pread(fd[read_file], &compressed_size, 4, file_offset);
            if (ret != 4 || compressed_size <= 0) {
                std::cerr << "[CompressedStorage] Bad header at chunk " << chunk_idx
                          << " offset " << file_offset << std::endl;
                exit(1);
            }

            ret = pread(fd[read_file], decompress_buf.data(), compressed_size, file_offset + 4);
            if (ret != compressed_size) {
                std::cerr << "[CompressedStorage] Read failed at chunk " << chunk_idx << std::endl;
                exit(1);
            }

            auto t1 = std::chrono::high_resolution_clock::now();

            int decompressed = LZ4_decompress_safe(
                decompress_buf.data(), shuffle_buf.data(),
                compressed_size, chunk_size
            );

            if (decompressed != (int)chunk_size) {
                std::cerr << "[CompressedStorage] Decompress failed! chunk " << chunk_idx
                          << " got " << decompressed << " expected " << chunk_size << std::endl;
                exit(1);
            }

            byte_unshuffle(shuffle_buf.data(), (char*)data, chunk_size);

            auto t2 = std::chrono::high_resolution_clock::now();

            time_pread_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            time_decompress_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
        } else {
            ssize_t ret = pread(fd[read_file], data, chunk_size, file_offset);
            if (ret != (ssize_t)chunk_size) {
                std::cerr << "[CompressedStorage] Raw read failed at chunk " << chunk_idx << std::endl;
                exit(1);
            }
        }
    }

    void write_chunk(size_t chunk_idx, const void* data) {
        std::lock_guard<std::mutex> lock(io_mutex);

        if (chunk_idx >= n_chunks) {
            std::cerr << "[CompressedStorage] Invalid write index: " << chunk_idx << std::endl;
            return;
        }

        if (offsets[write_file][chunk_idx] != write_offset) {

            if (chunk_idx == 0) {
                write_offset = 0;
            }
        }
        offsets[write_file][chunk_idx] = write_offset;

        size_t bytes_written;

        if (compression_enabled) {
            auto tw0 = std::chrono::high_resolution_clock::now();

            memcpy(shuffle_buf.data(), data, chunk_size);
            truncate_mantissa(shuffle_buf.data(), chunk_size, trunc_bits);
            byte_shuffle(shuffle_buf.data(), decompress_buf.data(), chunk_size);

            int compressed_size = LZ4_compress_default(
                decompress_buf.data(), compress_buf.data() + 4,
                chunk_size, LZ4_compressBound(chunk_size)
            );

            if (compressed_size <= 0) {
                std::cerr << "[CompressedStorage] Compression failed!" << std::endl;
                exit(1);
            }

            memcpy(compress_buf.data(), &compressed_size, 4);
            bytes_written = 4 + compressed_size;

            auto tw1 = std::chrono::high_resolution_clock::now();

            ssize_t ret = pwrite(fd[write_file], compress_buf.data(), bytes_written, write_offset);
            if (ret != (ssize_t)bytes_written) {
                perror("pwrite chunk");
                exit(1);
            }

            auto tw2 = std::chrono::high_resolution_clock::now();

            time_compress_ms += std::chrono::duration<double, std::milli>(tw1 - tw0).count();
            time_pwrite_ms += std::chrono::duration<double, std::milli>(tw2 - tw1).count();
        } else {
            bytes_written = chunk_size;
            ssize_t ret = pwrite(fd[write_file], data, chunk_size, write_offset);
            if (ret != (ssize_t)chunk_size) {
                perror("pwrite raw chunk");
                exit(1);
            }
        }

        write_offset += bytes_written;
        offsets[write_file][chunk_idx + 1] = write_offset;

        runtime_compressed += bytes_written;
        runtime_uncompressed += chunk_size;
    }

    void swap_files() {
        std::lock_guard<std::mutex> lock(io_mutex);

        fsync(fd[write_file]);

        std::swap(read_file, write_file);
        write_offset = 0;
        offsets[write_file][0] = 0;
    }

    void sync() {
        fsync(fd[0]);
        fsync(fd[1]);
    }

    size_t get_file_size() {
        return lseek(fd[read_file], 0, SEEK_END);
    }

    int get_read_fd() const { return fd[read_file]; }
    int get_write_fd() const { return fd[write_file]; }

    size_t runtime_compressed = 0;
    size_t runtime_uncompressed = 0;

    double time_pread_ms = 0;
    double time_decompress_ms = 0;
    double time_compress_ms = 0;
    double time_pwrite_ms = 0;

    void print_compression_summary() const {
        if (runtime_uncompressed > 0) {
            double ratio_pct = 100.0 * runtime_compressed / runtime_uncompressed;
            double reduction = (double)runtime_uncompressed / runtime_compressed;
            std::cout << "[Compression Summary] Total uncompressed: "
                      << runtime_uncompressed / (1024ULL*1024*1024) << " GB, compressed: "
                      << runtime_compressed / (1024ULL*1024) << " MB ("
                      << ratio_pct << "%), reduction: " << reduction << "x" << std::endl;
        }
        if (time_pread_ms > 0 || time_pwrite_ms > 0) {
            std::cout << "[IO Detail] pread: " << time_pread_ms << " ms, decompress: "
                      << time_decompress_ms << " ms, compress: " << time_compress_ms
                      << " ms, pwrite: " << time_pwrite_ms << " ms" << std::endl;
        }
    }
};

inline bool should_enable_compression(const std::string& path, size_t required_bytes) {
    struct statvfs stat;
    std::string dir = path.substr(0, path.rfind('/'));
    if (dir.empty()) dir = ".";

    if (statvfs(dir.c_str(), &stat) != 0) {
        return true;
    }

    size_t available = (size_t)stat.f_bavail * stat.f_frsize;

    bool need_compression = (available < required_bytes * 11 / 10);

    std::cout << "[Storage] Available: " << available / (1024ULL*1024*1024) << " GB, "
              << "Required (raw): " << required_bytes / (1024ULL*1024*1024) << " GB -> "
              << (need_compression ? "COMPRESSION ENABLED" : "compression disabled")
              << std::endl;

    return need_compression;
}
