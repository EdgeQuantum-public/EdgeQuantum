#pragma once
#include <vector>
#include <string>
#include <complex>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <cuda_runtime.h>
#include <custatevec.h>
#include <cuComplex.h>
#include "chunk_manager.hpp"
#include "io_backend.hpp"
#include "compression.hpp"

enum class SimMode {
    Native,
    UVM,
    Tiered
};

class IoWorker {
public:
    IoWorker() : stop(false) {
        worker_thread = std::thread([this]{ run(); });
    }

    ~IoWorker() {
        {
            std::lock_guard<std::mutex> lock(m);
            stop = true;
        }
        cv.notify_one();
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
    }

    void submit(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(m);
            tasks.push_back(task);
        }
        cv.notify_one();
    }

private:
    void run() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(m);
                cv.wait(lock, [this]{ return stop || !tasks.empty(); });
                if (stop && tasks.empty()) return;

                task = std::move(tasks.front());
                tasks.erase(tasks.begin());
            }
            task();
        }
    }

    std::thread worker_thread;
    std::vector<std::function<void()>> tasks;
    std::mutex m;
    std::condition_variable cv;
    bool stop;
};

class EdgeQuantumSim {
    int n_qubits;
    int chunk_bits;
    size_t chunk_size;
    size_t n_chunks;

    IOBackend io_read;
    IOBackend io_write;
    size_t state_size;
    SimMode mode;
    std::string storage_path;

    CompressedStorage* compressed_storage;
    bool use_compression;

    ChunkManager* chunk_mgr;
    IOBackend* io;

    void* full_state_ptr;

    int NUM_PIPELINE_BUFS = 4;
    bool device_buf_ready;

    custatevecHandle_t handle;
    void* d_ws;
    size_t ws_size;
    cudaStream_t stream;
    cudaStream_t copy_stream;

    void* d_gate_matrix;
    int target_idx[1];
    int control_idx[1];

    IoWorker* write_worker;

public:

    EdgeQuantumSim(int qubits, std::string path, std::string mode_str = "auto", bool force_mode = false);
    ~EdgeQuantumSim();

    void run_qv(int depth);
    void run_vqc(int layers);
    void run_qsvm(int feature_dim);
    void run_ghz();
    void run_random(int depth);
    void run_vqe(int batch);

    bool validate_hadamard();
    bool get_first_two_amplitudes(std::complex<float>& a0, std::complex<float>& a1);

private:
    void init_storage();
    void reset_zero_state();

    using KernelFunc = std::function<void(int, void*, cudaStream_t)>;
    void process_pipeline(KernelFunc kernel);

    void process_pipeline_global_1q(int global_qubit, const void* d_mat);

    void process_pipeline_global_cnot(int control, int target);

    void apply_gate_1q(void* d_sv, int target, const void* d_mat, cudaStream_t s);
    void apply_cnot_local(void* d_sv, int c, int t, cudaStream_t s);
    void apply_rz(void* d_sv, int target, float theta, cudaStream_t s);
    void apply_ry(void* d_sv, int target, float theta, cudaStream_t s);

    void* d_temp_gate;
};
