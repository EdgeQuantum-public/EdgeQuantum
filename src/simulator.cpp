#include "simulator.hpp"
#include <iostream>
#include <cstring>
#include <future>
#include <chrono>
#include <random>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <sys/statvfs.h>

std::future<void> submit_async_task(IoWorker* worker, std::function<void()> task) {
    auto p = std::make_shared<std::promise<void>>();
    auto f = p->get_future();
    worker->submit([p, task]() {
        task();
        p->set_value();
    });
    return f;
}

static size_t get_available_gpu_memory() {
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    return free_mem;
}

static size_t get_available_disk_space(const std::string& path) {
    struct statvfs stat;
    std::string dir = path.substr(0, path.rfind('/'));
    if (dir.empty()) dir = ".";
    if (statvfs(dir.c_str(), &stat) != 0) {
        return 0;
    }
    return (size_t)stat.f_bavail * stat.f_frsize;
}

EdgeQuantumSim::EdgeQuantumSim(int qubits, std::string path, std::string mode_str, bool force_mode)
        : n_qubits(qubits),
            chunk_bits(25),
            chunk_size(0),
            n_chunks(0),
            io_read(path),
            io_write(path),
            state_size((1ULL << qubits) * sizeof(cuComplex)),
            mode(SimMode::Native),
            storage_path(path),
            compressed_storage(nullptr),
            use_compression(false),
            chunk_mgr(nullptr),
            io(nullptr),
            full_state_ptr(nullptr),
            device_buf_ready(false),
            write_worker(nullptr)
{

    size_t state_bytes = (1ULL << qubits) * sizeof(cuComplex);
    size_t gpu_free = get_available_gpu_memory();
    size_t gpu_threshold = (gpu_free * 80) / 100;

    if (mode_str == "native") {
        if (!force_mode && state_bytes > gpu_threshold) {
            std::cerr << "\n[ERROR] Mode 'native' requested but state ("
                      << state_bytes / (1024*1024) << " MB) exceeds GPU memory ("
                      << gpu_free / (1024*1024) << " MB free)." << std::endl;
            throw std::runtime_error("OOM: Insufficient GPU memory for native mode");
        }
        mode = SimMode::Native;
        std::cout << "\n[Mode] native -> cudaMalloc (device memory)\n" << std::endl;
    } else if (mode_str == "uvm") {
        if (!force_mode && state_bytes > gpu_threshold) {
            std::cerr << "\n[ERROR] Mode 'uvm' requested but state ("
                      << state_bytes / (1024*1024) << " MB) exceeds GPU memory ("
                      << gpu_free / (1024*1024) << " MB free)." << std::endl;
            throw std::runtime_error("OOM: Insufficient GPU memory for uvm mode");
        }
        mode = SimMode::UVM;
        std::cout << "\n[Mode] uvm -> cudaMallocManaged (unified virtual memory)\n" << std::endl;
    } else if (mode_str == "tiered" || mode_str == "blocking" || mode_str == "async") {

        mode = SimMode::Tiered;
        std::cout << "\n[Mode] " << mode_str << " -> Using Tiered mode (NVMe pipeline)\n" << std::endl;
    } else {

        if (state_bytes <= gpu_threshold) {
            mode = SimMode::Native;
            std::cout << "\n[Auto] State (" << state_bytes / (1024*1024) << " MB) fits in GPU memory ("
                      << gpu_free / (1024*1024) << " MB free). Using Native mode.\n" << std::endl;
        } else {
            mode = SimMode::Tiered;
            std::cout << "\n[Auto] State (" << state_bytes / (1024*1024) << " MB) exceeds GPU memory ("
                      << gpu_free / (1024*1024) << " MB free). Using Tiered mode.\n" << std::endl;
        }
    }

    const char* num_bufs_env = std::getenv("NUM_BUFS");
    if (num_bufs_env) {
        NUM_PIPELINE_BUFS = std::atoi(num_bufs_env);
        if (NUM_PIPELINE_BUFS < 1) NUM_PIPELINE_BUFS = 1;
        if (NUM_PIPELINE_BUFS > 16) NUM_PIPELINE_BUFS = 16;
        std::cout << "[Config] Pipeline buffers overridden to " << NUM_PIPELINE_BUFS << std::endl;
    }

    int chunk_pow = 25;
    if (chunk_pow > n_qubits) chunk_pow = n_qubits;
    chunk_bits = chunk_pow;
    chunk_size = (1ULL << chunk_pow) * sizeof(cuComplex);
    n_chunks = (1ULL << n_qubits) >> chunk_pow;
    if (n_chunks == 0) n_chunks = 1;

    std::string mode_display = (mode == SimMode::Native) ? "Native (cudaMalloc)" :
                               (mode == SimMode::UVM) ? "UVM (cudaMallocManaged)" : "Tiered (EdgeQuantum)";

    std::cout << "[Sim] Mode: " << mode_display
              << " | Qubits: " << n_qubits
              << " | State Size: " << state_size / (1024ULL*1024*1024) << " GB" << std::endl;

    CUSV_CHECK(custatevecCreate(&handle));

    int leastPriority, greatestPriority;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    CUDA_CHECK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatestPriority));
    CUDA_CHECK(cudaStreamCreateWithPriority(&copy_stream, cudaStreamNonBlocking, greatestPriority));

    CUSV_CHECK(custatevecSetStream(handle, stream));

    float s2 = 1.0f / sqrt(2.0f);
    std::complex<float> h_gate[4] = {{s2,0}, {s2,0}, {s2,0}, {-s2,0}};
    CUDA_CHECK(cudaMalloc(&d_gate_matrix, sizeof(h_gate)));
    CUDA_CHECK(cudaMemcpy(d_gate_matrix, h_gate, sizeof(h_gate), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_temp_gate, 4 * sizeof(cuComplex)));

    int ws_nbits = ((mode == SimMode::Native || mode == SimMode::UVM)) ? n_qubits : chunk_bits;
    size_t required_ws = 0;
    CUSV_CHECK(custatevecApplyMatrixGetWorkspaceSize(
        handle, CUDA_C_32F, ws_nbits, d_gate_matrix, CUDA_C_32F,
        CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, 1, 0, CUSTATEVEC_COMPUTE_32F, &required_ws
    ));
    ws_size = std::max(required_ws, static_cast<size_t>(128 * 1024 * 1024));
    CUDA_CHECK(cudaMalloc(&d_ws, ws_size));

    if ((mode == SimMode::Native || mode == SimMode::UVM)) {

        size_t total_bytes = (1ULL << n_qubits) * sizeof(cuComplex);
        std::cout << "[Alloc] cudaMalloc " << total_bytes/(1024.0*1024.0) << " MB..." << std::endl;
        CUDA_CHECK(cudaMalloc(&full_state_ptr, total_bytes));

        CUDA_CHECK(cudaMemset(full_state_ptr, 0, total_bytes));
        cuComplex one = {1.0f, 0.0f};
        CUDA_CHECK(cudaMemcpy(full_state_ptr, &one, sizeof(cuComplex), cudaMemcpyHostToDevice));

    } else if (mode == SimMode::UVM) {

        size_t total_bytes = (1ULL << n_qubits) * sizeof(cuComplex);
        std::cout << "[Alloc] cudaMallocManaged " << total_bytes/(1024.0*1024.0) << " MB..." << std::endl;
        CUDA_CHECK(cudaMallocManaged(&full_state_ptr, total_bytes));

        CUDA_CHECK(cudaMemset(full_state_ptr, 0, total_bytes));
        cuComplex one = {1.0f, 0.0f};

        ((cuComplex*)full_state_ptr)[0] = one;
        CUDA_CHECK(cudaDeviceSynchronize());

    } else {

        chunk_mgr = new ChunkManager(chunk_size, n_chunks, NUM_PIPELINE_BUFS);
        write_worker = new IoWorker();

        if (std::getenv("FORCE_COMPRESS") != nullptr) {
            use_compression = true;
            std::cout << "[Storage] Compression FORCED ON by env" << std::endl;
        } else if (std::getenv("NO_COMPRESS") != nullptr) {
            use_compression = false;
            std::cout << "[Storage] Compression FORCED OFF by env" << std::endl;
        } else {
            use_compression = should_enable_compression(path, state_size);
        }
        if (use_compression) {
            compressed_storage = new CompressedStorage(path + ".lz4", chunk_size, n_chunks, true);
        }

        device_buf_ready = true;

        init_storage();
    }
}

EdgeQuantumSim::~EdgeQuantumSim() {
    if (write_worker) delete write_worker;

    if (full_state_ptr) cudaFree(full_state_ptr);

    device_buf_ready = false;

    if (compressed_storage) {
        compressed_storage->print_compression_summary();
        delete compressed_storage;
    }
    if (chunk_mgr) delete chunk_mgr;

    cudaFree(d_ws);
    cudaFree(d_gate_matrix);
    cudaFree(d_temp_gate);
    custatevecDestroy(handle);
    cudaStreamDestroy(stream);
    cudaStreamDestroy(copy_stream);
}

void EdgeQuantumSim::init_storage() {

    if (!chunk_mgr) return;

    void* buf = chunk_mgr->get_buffer(0);
    memset(buf, 0, chunk_size);

    float amp = 1.0f / sqrtf((float)n_chunks);
    ((std::complex<float>*)buf)[0] = {amp, 0.0f};

    std::cout << "[Init] Distributed |0> (amp=" << amp << ", " << n_chunks
              << " chunks";
    if (use_compression) std::cout << ", LZ4 compressed";
    std::cout << ")..." << std::endl;

    size_t total_written = 0;
    for (size_t i = 0; i < n_chunks; i++) {
        if (use_compression && compressed_storage) {
            compressed_storage->init_write_chunk(i, buf);
        } else {
            io_write.write(i * chunk_size, buf, chunk_size);
            total_written += chunk_size;
        }

        if (n_chunks >= 100 && (i+1) % (n_chunks/10) == 0) {
            std::cout << "[Init] Progress: " << (100*(i+1)/n_chunks) << "%" << std::endl;
        }
    }

    if (use_compression && compressed_storage) {
        compressed_storage->finish_init();
    } else {
        io_write.sync();
    }
}

void EdgeQuantumSim::reset_zero_state() {
    if ((mode == SimMode::Native || mode == SimMode::UVM)) {
        size_t total_bytes = (1ULL << n_qubits) * sizeof(cuComplex);
        CUDA_CHECK(cudaMemset(full_state_ptr, 0, total_bytes));
        cuComplex one = {1.0f, 0.0f};
        CUDA_CHECK(cudaMemcpy(full_state_ptr, &one, sizeof(cuComplex), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
        return;
    }

    if (chunk_mgr) {
        init_storage();
    }
}

bool EdgeQuantumSim::get_first_two_amplitudes(std::complex<float>& a0, std::complex<float>& a1) {
    if ((mode == SimMode::Native || mode == SimMode::UVM)) {
        cuComplex host_vals[2];
        CUDA_CHECK(cudaMemcpyAsync(host_vals, full_state_ptr, sizeof(host_vals), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        a0 = {host_vals[0].x, host_vals[0].y};
        a1 = {host_vals[1].x, host_vals[1].y};
        return true;
    }

    if (!chunk_mgr) return false;

    if (use_compression && compressed_storage) {
        compressed_storage->read_chunk(0, chunk_mgr->get_buffer(0));
    } else {
        chunk_mgr->read_chunk(io_read, 0, 0);
    }
    auto* buf = reinterpret_cast<std::complex<float>*>(chunk_mgr->get_buffer(0));
    a0 = buf[0];
    a1 = buf[1];
    return true;
}

bool EdgeQuantumSim::validate_hadamard() {
    reset_zero_state();

    auto kernel = [this](int chunk_id, void* d_sv, cudaStream_t s) {
        apply_gate_1q(d_sv, 0, d_gate_matrix, s);
    };

    if ((mode == SimMode::Native || mode == SimMode::UVM)) {
        kernel(0, full_state_ptr, this->stream);
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        process_pipeline(kernel);
    }

    std::complex<float> a0, a1;
    if (!get_first_two_amplitudes(a0, a1)) {
        std::cout << "[Verify] Failed to read amplitudes." << std::endl;
        return false;
    }

    const float s2 = 1.0f / sqrt(2.0f);
    const float eps = 1e-3f;
    bool ok = (std::abs(a0.real() - s2) < eps && std::abs(a1.real() - s2) < eps &&
               std::abs(a0.imag()) < eps && std::abs(a1.imag()) < eps);

    std::cout << "[Verify] |0>=" << a0.real() << "+" << a0.imag() << "j, |1>="
              << a1.real() << "+" << a1.imag() << "j" << std::endl;
    std::cout << (ok ? "[Verify] PASS" : "[Verify] FAIL") << std::endl;
    return ok;
}

void EdgeQuantumSim::process_pipeline(KernelFunc kernel) {
    if ((mode == SimMode::Native || mode == SimMode::UVM)) {
        std::cerr << "[Error] process_pipeline called in Native mode!" << std::endl;
        return;
    }

    IOBackend* io_read_ptr = &io_read;
    IOBackend* io_write_ptr = &io_write;

    auto read_chunk_data = [this, io_read_ptr](size_t chunk_idx, int buf_idx) {
        if (use_compression && compressed_storage) {
            compressed_storage->read_chunk(chunk_idx, chunk_mgr->get_buffer(buf_idx));
        } else {
            chunk_mgr->read_chunk(*io_read_ptr, (int)chunk_idx, buf_idx);
        }
    };

    auto write_chunk_data = [this, io_write_ptr](size_t chunk_idx, int buf_idx) {
        if (use_compression && compressed_storage) {
            compressed_storage->write_chunk(chunk_idx, chunk_mgr->get_buffer(buf_idx));
        } else {
            chunk_mgr->write_chunk(*io_write_ptr, (int)chunk_idx, buf_idx);
        }
    };

    bool force_blocking = (std::getenv("FORCE_BLOCKING") != nullptr);

    bool true_bmqsim = (std::getenv("TRUE_BMQSIM") != nullptr);

    bool enable_breakdown = (std::getenv("IO_BREAKDOWN") != nullptr);
    double total_read_ms = 0, total_compute_ms = 0, total_write_ms = 0;

    if (true_bmqsim) {
        std::cout << "[Pipeline] TRUE BMQSim (naive malloc + cudaMemcpy, blocking)" << std::endl;

        void* h_buf = nullptr;
        if (posix_memalign(&h_buf, 4096, chunk_size) != 0) {
            std::cerr << "posix_memalign failed" << std::endl; exit(1);
        }
        void* d_buf = nullptr;
        CUDA_CHECK(cudaMalloc(&d_buf, chunk_size));

        IOBackend* io_r = &io_read;
        IOBackend* io_w = &io_write;

        for (size_t i = 0; i < n_chunks; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();

            if (use_compression && compressed_storage) {
                compressed_storage->read_chunk(i, h_buf);
            } else {
                io_r->read(i * chunk_size, h_buf, chunk_size);
            }

            CUDA_CHECK(cudaMemcpy(d_buf, h_buf, chunk_size, cudaMemcpyHostToDevice));
            auto t1 = std::chrono::high_resolution_clock::now();

            kernel((int)i, d_buf, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto t2 = std::chrono::high_resolution_clock::now();

            CUDA_CHECK(cudaMemcpy(h_buf, d_buf, chunk_size, cudaMemcpyDeviceToHost));

            if (use_compression && compressed_storage) {
                compressed_storage->write_chunk(i, h_buf);
            } else {
                io_w->write(i * chunk_size, h_buf, chunk_size);
            }
            auto t3 = std::chrono::high_resolution_clock::now();

            if (enable_breakdown) {
                total_read_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
                total_compute_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
                total_write_ms += std::chrono::duration<double, std::milli>(t3 - t2).count();
            }
        }

        if (use_compression && compressed_storage) {
            compressed_storage->swap_files();
        }

        free(h_buf);
        CUDA_CHECK(cudaFree(d_buf));

        if (enable_breakdown) {
            std::cout << "[Breakdown] Read+H2D: " << total_read_ms << " ms, Compute: " << total_compute_ms
                      << " ms, D2H+Write: " << total_write_ms << " ms" << std::endl;
        }
        return;
    }

    if (force_blocking) {
        std::cout << "[Pipeline] UVM Blocking mode" << std::endl;
        for (size_t i = 0; i < n_chunks; i++) {
            int buf = i % NUM_PIPELINE_BUFS;

            auto t0 = std::chrono::high_resolution_clock::now();
            read_chunk_data(i, buf);
            CUDA_CHECK(cudaDeviceSynchronize());
            auto t1 = std::chrono::high_resolution_clock::now();

            void* uvm_buffer = chunk_mgr->get_buffer(buf);
            CUDA_CHECK(cudaStreamAttachMemAsync(stream, uvm_buffer, 0, cudaMemAttachGlobal));
            kernel((int)i, uvm_buffer, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaStreamAttachMemAsync(stream, uvm_buffer, 0, cudaMemAttachHost));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto t2 = std::chrono::high_resolution_clock::now();

            write_chunk_data(i, buf);
            auto t3 = std::chrono::high_resolution_clock::now();

            if (enable_breakdown) {
                total_read_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
                total_compute_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
                total_write_ms += std::chrono::duration<double, std::milli>(t3 - t2).count();
            }
        }
        if (use_compression && compressed_storage) {
            compressed_storage->swap_files();
        }
        if (enable_breakdown) {
            std::cout << "[Breakdown] Read: " << total_read_ms << " ms, Compute: " << total_compute_ms
                      << " ms, Write: " << total_write_ms << " ms" << std::endl;
        }
        return;
    }

    std::cout << "[Pipeline] Async 4-buffer (EdgeQuantum";
    if (use_compression) std::cout << ", LZ4";
    std::cout << ")" << std::endl;

    std::future<void> pending_write;
    int pending_write_chunk = -1;
    int pending_write_buf = -1;

    for (int i = 0; i < NUM_PIPELINE_BUFS; i++) {
        void* ptr = chunk_mgr->get_buffer(i);
        CUDA_CHECK(cudaStreamAttachMemAsync(stream, ptr, 0, cudaMemAttachHost));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (size_t c = 0; c < n_chunks; c++) {
        int buf = c % NUM_PIPELINE_BUFS;
        void* uvm_buffer = chunk_mgr->get_buffer(buf);

        auto tw0 = std::chrono::high_resolution_clock::now();
        if (pending_write.valid()) {
            pending_write.get();
        }
        auto tw1 = std::chrono::high_resolution_clock::now();

        auto tr0 = std::chrono::high_resolution_clock::now();
        read_chunk_data(c, buf);
        auto tr1 = std::chrono::high_resolution_clock::now();

        if (pending_write_chunk >= 0) {
            const int chunk_to_write = pending_write_chunk;
            const int buf_to_write = pending_write_buf;
            auto* write_ms_ptr = enable_breakdown ? &total_write_ms : nullptr;
            pending_write = submit_async_task(write_worker, [write_chunk_data, chunk_to_write, buf_to_write, write_ms_ptr]() {
                auto ws0 = std::chrono::high_resolution_clock::now();
                write_chunk_data((size_t)chunk_to_write, buf_to_write);
                auto ws1 = std::chrono::high_resolution_clock::now();
                if (write_ms_ptr) {

                    *write_ms_ptr += std::chrono::duration<double, std::milli>(ws1 - ws0).count();
                }
            });
        }

        auto tc0 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaStreamAttachMemAsync(stream, uvm_buffer, 0, cudaMemAttachGlobal));
        kernel((int)c, uvm_buffer, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaStreamAttachMemAsync(stream, uvm_buffer, 0, cudaMemAttachHost));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto tc1 = std::chrono::high_resolution_clock::now();

        if (enable_breakdown) {
            total_read_ms += std::chrono::duration<double, std::milli>(tr1 - tr0).count();
            total_compute_ms += std::chrono::duration<double, std::milli>(tc1 - tc0).count();
        }

        pending_write_chunk = (int)c;
        pending_write_buf = buf;
    }

    if (pending_write.valid()) {
        pending_write.get();
    }
    if (pending_write_chunk >= 0) {
        auto td0 = std::chrono::high_resolution_clock::now();
        write_chunk_data((size_t)pending_write_chunk, pending_write_buf);
        auto td1 = std::chrono::high_resolution_clock::now();
        if (enable_breakdown) {
            total_write_ms += std::chrono::duration<double, std::milli>(td1 - td0).count();
        }
    }

    if (use_compression && compressed_storage) {
        compressed_storage->swap_files();
    }

    if (enable_breakdown) {
        std::cout << "[Breakdown] Read: " << total_read_ms << " ms, Compute: " << total_compute_ms
                  << " ms, Write: " << total_write_ms << " ms" << std::endl;
    }
}

void EdgeQuantumSim::process_pipeline_global_1q(int global_qubit, const void* d_mat) {
    if (mode == SimMode::Native || mode == SimMode::UVM) {

        target_idx[0] = global_qubit;
        CUSV_CHECK(custatevecApplyMatrix(
            handle, full_state_ptr, CUDA_C_32F, n_qubits, (void*)d_mat, CUDA_C_32F,
            CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, target_idx, 1, nullptr, nullptr, 0,
            CUSTATEVEC_COMPUTE_32F, d_ws, ws_size
        ));
        CUDA_CHECK(cudaDeviceSynchronize());
        return;
    }

    int global_bit = global_qubit - chunk_bits;
    size_t pair_stride = 1ULL << global_bit;
    int combined_bits = chunk_bits + 1;

    IOBackend* io_read_ptr = &io_read;
    IOBackend* io_write_ptr = &io_write;

    auto read_chunk_data = [this, io_read_ptr](size_t chunk_idx, int buf_idx) {
        if (use_compression && compressed_storage) {
            compressed_storage->read_chunk(chunk_idx, chunk_mgr->get_buffer(buf_idx));
        } else {
            chunk_mgr->read_chunk(*io_read_ptr, (int)chunk_idx, buf_idx);
        }
    };

    auto write_chunk_data = [this, io_write_ptr](size_t chunk_idx, int buf_idx) {
        if (use_compression && compressed_storage) {
            compressed_storage->write_chunk(chunk_idx, chunk_mgr->get_buffer(buf_idx));
        } else {
            chunk_mgr->write_chunk(*io_write_ptr, (int)chunk_idx, buf_idx);
        }
    };

    std::future<void> pending_write;
    int pending_pair = -1;
    size_t pending_c0 = 0, pending_c1 = 0;

    void* pair_buf = chunk_mgr->get_pair_buffer();
    void* slot0 = chunk_mgr->get_pair_slot(0);
    void* slot1 = chunk_mgr->get_pair_slot(1);

    CUDA_CHECK(cudaStreamAttachMemAsync(stream, pair_buf, 0, cudaMemAttachHost));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (size_t c = 0; c < n_chunks; c++) {
        if (c & (1ULL << global_bit)) continue;
        size_t c0 = c;
        size_t c1 = c | (1ULL << global_bit);

        if (use_compression && compressed_storage) {
            compressed_storage->read_chunk(c0, slot0);
            compressed_storage->read_chunk(c1, slot1);
        } else {
            io_read.read(c0 * chunk_size, slot0, chunk_size);
            io_read.read(c1 * chunk_size, slot1, chunk_size);
        }

        CUDA_CHECK(cudaStreamAttachMemAsync(stream, pair_buf, 0, cudaMemAttachGlobal));
        target_idx[0] = chunk_bits;
        CUSV_CHECK(custatevecApplyMatrix(
            handle, pair_buf, CUDA_C_32F, combined_bits, (void*)d_mat, CUDA_C_32F,
            CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, target_idx, 1, nullptr, nullptr, 0,
            CUSTATEVEC_COMPUTE_32F, d_ws, ws_size
        ));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamAttachMemAsync(stream, pair_buf, 0, cudaMemAttachHost));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (use_compression && compressed_storage) {
            compressed_storage->write_chunk(c0, slot0);
            compressed_storage->write_chunk(c1, slot1);
        } else {
            io_write.write(c0 * chunk_size, slot0, chunk_size);
            io_write.write(c1 * chunk_size, slot1, chunk_size);
        }
    }

    if (use_compression && compressed_storage) {
        compressed_storage->swap_files();
    }
}

void EdgeQuantumSim::process_pipeline_global_cnot(int ctrl, int tgt) {
    if (mode == SimMode::Native || mode == SimMode::UVM) {
        target_idx[0] = tgt;
        control_idx[0] = ctrl;
        CUSV_CHECK(custatevecApplyMatrix(
            handle, full_state_ptr, CUDA_C_32F, n_qubits, d_gate_matrix, CUDA_C_32F,
            CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, target_idx, 1, control_idx, nullptr, 1,
            CUSTATEVEC_COMPUTE_32F, d_ws, ws_size
        ));
        CUDA_CHECK(cudaDeviceSynchronize());
        return;
    }

    bool ctrl_global = (ctrl >= chunk_bits);
    bool tgt_global = (tgt >= chunk_bits);

    if (ctrl_global && tgt_global) {

        std::cerr << "[Warning] 2-global-qubit CNOT not yet optimized, using fallback" << std::endl;
        return;
    }

    int global_q = ctrl_global ? ctrl : tgt;
    int local_q = ctrl_global ? tgt : ctrl;
    int global_bit = global_q - chunk_bits;

    IOBackend* io_read_ptr = &io_read;
    IOBackend* io_write_ptr = &io_write;

    auto read_chunk_data = [this, io_read_ptr](size_t chunk_idx, int buf_idx) {
        if (use_compression && compressed_storage) {
            compressed_storage->read_chunk(chunk_idx, chunk_mgr->get_buffer(buf_idx));
        } else {
            chunk_mgr->read_chunk(*io_read_ptr, (int)chunk_idx, buf_idx);
        }
    };

    auto write_chunk_data = [this, io_write_ptr](size_t chunk_idx, int buf_idx) {
        if (use_compression && compressed_storage) {
            compressed_storage->write_chunk(chunk_idx, chunk_mgr->get_buffer(buf_idx));
        } else {
            chunk_mgr->write_chunk(*io_write_ptr, (int)chunk_idx, buf_idx);
        }
    };

    int combined_bits = chunk_bits + 1;

    int pair_ctrl = ctrl_global ? chunk_bits : ctrl;
    int pair_tgt = tgt_global ? chunk_bits : tgt;

    void* pair_buf = chunk_mgr->get_pair_buffer();
    void* slot0 = chunk_mgr->get_pair_slot(0);
    void* slot1 = chunk_mgr->get_pair_slot(1);

    CUDA_CHECK(cudaStreamAttachMemAsync(stream, pair_buf, 0, cudaMemAttachHost));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (size_t c = 0; c < n_chunks; c++) {
        if (c & (1ULL << global_bit)) continue;
        size_t c0 = c;
        size_t c1 = c | (1ULL << global_bit);

        if (use_compression && compressed_storage) {
            compressed_storage->read_chunk(c0, slot0);
            compressed_storage->read_chunk(c1, slot1);
        } else {
            io_read.read(c0 * chunk_size, slot0, chunk_size);
            io_read.read(c1 * chunk_size, slot1, chunk_size);
        }

        CUDA_CHECK(cudaStreamAttachMemAsync(stream, pair_buf, 0, cudaMemAttachGlobal));
        target_idx[0] = pair_tgt;
        control_idx[0] = pair_ctrl;
        CUSV_CHECK(custatevecApplyMatrix(
            handle, pair_buf, CUDA_C_32F, combined_bits, d_gate_matrix, CUDA_C_32F,
            CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, target_idx, 1, control_idx, nullptr, 1,
            CUSTATEVEC_COMPUTE_32F, d_ws, ws_size
        ));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamAttachMemAsync(stream, pair_buf, 0, cudaMemAttachHost));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (use_compression && compressed_storage) {
            compressed_storage->write_chunk(c0, slot0);
            compressed_storage->write_chunk(c1, slot1);
        } else {
            io_write.write(c0 * chunk_size, slot0, chunk_size);
            io_write.write(c1 * chunk_size, slot1, chunk_size);
        }
    }

    if (use_compression && compressed_storage) {
        compressed_storage->swap_files();
    }
}

void EdgeQuantumSim::apply_gate_1q(void* d_sv, int target, const void* d_mat, cudaStream_t s) {
    target_idx[0] = target;
    int n_bits = ((mode == SimMode::Native || mode == SimMode::UVM)) ? n_qubits : chunk_bits;

    CUSV_CHECK(custatevecApplyMatrix(
        handle, d_sv, CUDA_C_32F, n_bits, (void*)d_mat, CUDA_C_32F,
        CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, target_idx, 1, nullptr, nullptr, 0,
        CUSTATEVEC_COMPUTE_32F, d_ws, ws_size
    ));
}

void EdgeQuantumSim::apply_rz(void* d_sv, int target, float theta, cudaStream_t s) {
    float ct = cosf(theta * 0.5f);
    float st = sinf(theta * 0.5f);

    cuComplex gate[4] = {{ct, -st}, {0,0}, {0,0}, {ct, st}};
    CUDA_CHECK(cudaMemcpyAsync(d_temp_gate, gate, sizeof(gate), cudaMemcpyHostToDevice, s));
    apply_gate_1q(d_sv, target, d_temp_gate, s);
}

void EdgeQuantumSim::apply_ry(void* d_sv, int target, float theta, cudaStream_t s) {
    float ct = cosf(theta * 0.5f);
    float st = sinf(theta * 0.5f);

    cuComplex gate[4] = {{ct,0}, {-st,0}, {st,0}, {ct,0}};
    CUDA_CHECK(cudaMemcpyAsync(d_temp_gate, gate, sizeof(gate), cudaMemcpyHostToDevice, s));
    apply_gate_1q(d_sv, target, d_temp_gate, s);
}

void EdgeQuantumSim::apply_cnot_local(void* d_sv, int c, int t, cudaStream_t s) {
    target_idx[0] = t;
    control_idx[0] = c;
    int n_bits = ((mode == SimMode::Native || mode == SimMode::UVM)) ? n_qubits : chunk_bits;
    CUSV_CHECK(custatevecApplyMatrix(
        handle, d_sv, CUDA_C_32F, n_bits, d_gate_matrix, CUDA_C_32F,
        CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, target_idx, 1, control_idx, nullptr, 1,
        CUSTATEVEC_COMPUTE_32F, d_ws, ws_size
    ));
}

void EdgeQuantumSim::run_qv(int depth) {
    std::cout << "[Circuit] Quantum Volume (Depth=" << depth << ")" << std::endl;
    for (int d = 0; d < depth; d++) {
        if (mode == SimMode::Native || mode == SimMode::UVM) {

            for (int k = 0; k < n_qubits - 1; k += 2) {
                apply_gate_1q(full_state_ptr, k, d_gate_matrix, stream);
                apply_cnot_local(full_state_ptr, k, k + 1, stream);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        } else {

            auto local_kernel = [this](int chunk_id, void* d_sv, cudaStream_t s) {
                for (int k = 0; k < chunk_bits - 1; k += 2) {
                    apply_gate_1q(d_sv, k, d_gate_matrix, s);
                    apply_cnot_local(d_sv, k, k + 1, s);
                }
            };
            process_pipeline(local_kernel);

            for (int k = chunk_bits - (chunk_bits % 2 == 0 ? 0 : 1); k < n_qubits - 1; k += 2) {
                if (k < chunk_bits && k + 1 < chunk_bits) continue;

                if (k >= chunk_bits) {
                    process_pipeline_global_1q(k, d_gate_matrix);
                }

                process_pipeline_global_cnot(k, k + 1);
            }
        }
    }
}

void EdgeQuantumSim::run_vqc(int layers) {
    std::cout << "[Circuit] VQC (Layers=" << layers << ")" << std::endl;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(M_PI / 3.0f, 0.5f);

    for (int l = 0; l < layers; l++) {
        int nbits = ((mode == SimMode::Native || mode == SimMode::UVM)) ? n_qubits : chunk_bits;
        std::vector<float> rz_angles(nbits);
        for (int k = 0; k < nbits; k++) rz_angles[k] = dist(rng);

        auto kernel = [this, nbits, &rz_angles](int chunk_id, void* d_sv, cudaStream_t s) {
            for (int k = 0; k < nbits; k++) {
                apply_rz(d_sv, k, rz_angles[k], s);
                apply_gate_1q(d_sv, k, d_gate_matrix, s);
            }
            for (int k = 0; k < nbits - 1; k++) apply_cnot_local(d_sv, k, k+1, s);
        };

        if ((mode == SimMode::Native || mode == SimMode::UVM)) {
             kernel(0, full_state_ptr, this->stream);
             CUDA_CHECK(cudaDeviceSynchronize());
        } else {
             process_pipeline(kernel);
        }
    }
}

void EdgeQuantumSim::run_qsvm(int feature_dim) {
    std::cout << "[Circuit] QSVM (FeatureDim=" << feature_dim << ")" << std::endl;

    int nbits_local = ((mode == SimMode::Native || mode == SimMode::UVM)) ? n_qubits : chunk_bits;
    std::vector<float> features(nbits_local);
    for (int j = 0; j < nbits_local; j++) {
        features[j] = M_PI * 0.3f * (1.0f + 0.5f * sinf(2.0f * M_PI * j / nbits_local));
    }

    for (int rep = 0; rep < 2; rep++) {
        int nbits = nbits_local;

        auto kernel = [this, nbits, &features](int chunk_id, void* d_sv, cudaStream_t s) {

             int n_feat = (nbits * 2) / 3;
             for (int j = 0; j < n_feat; j++) apply_gate_1q(d_sv, j, d_gate_matrix, s);
             for (int j = 0; j < n_feat; j++) apply_rz(d_sv, j, features[j], s);

             for (int j = 0; j < n_feat - 1; j += 2) apply_cnot_local(d_sv, j, j+1, s);
        };

        if ((mode == SimMode::Native || mode == SimMode::UVM)) {
             kernel(0, full_state_ptr, this->stream);
             CUDA_CHECK(cudaDeviceSynchronize());
        } else {
             process_pipeline(kernel);
        }
    }
}

void EdgeQuantumSim::run_ghz() {
    std::cout << "[Circuit] GHZ" << std::endl;
    if (mode == SimMode::Native || mode == SimMode::UVM) {
        apply_gate_1q(full_state_ptr, 0, d_gate_matrix, stream);
        for (int k = 0; k < n_qubits - 1; k++)
            apply_cnot_local(full_state_ptr, k, k + 1, stream);
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {

        auto local_kernel = [this](int chunk_id, void* d_sv, cudaStream_t s) {
            apply_gate_1q(d_sv, 0, d_gate_matrix, s);
            for (int k = 0; k < chunk_bits - 1; k++)
                apply_cnot_local(d_sv, k, k + 1, s);
        };
        process_pipeline(local_kernel);

        for (int k = chunk_bits - 1; k < n_qubits - 1; k++) {
            process_pipeline_global_cnot(k, k + 1);
        }
    }
}

void EdgeQuantumSim::run_random(int depth) {
    std::cout << "[Circuit] Random (Depth=" << depth << ")" << std::endl;
    std::mt19937 rng(77);
    std::uniform_real_distribution<float> angle_dist(0.0f, 2.0f * M_PI);
    std::uniform_int_distribution<int> gate_dist(0, 3);

    for (int d = 0; d < depth; d++) {
        int nbits = ((mode == SimMode::Native || mode == SimMode::UVM)) ? n_qubits : chunk_bits;

        std::vector<int> gate_types(nbits);
        std::vector<float> angles(nbits);
        std::vector<bool> do_cnot(nbits, false);
        for (int k = 0; k < nbits; k++) {
            gate_types[k] = gate_dist(rng);
            angles[k] = angle_dist(rng);
            if (k < nbits - 1) do_cnot[k] = (gate_dist(rng) > 1);
        }

        auto kernel = [this, nbits, &gate_types, &angles, &do_cnot](int chunk_id, void* d_sv, cudaStream_t s) {
            for (int k = 0; k < nbits; k++) {
                switch (gate_types[k]) {
                    case 0: apply_gate_1q(d_sv, k, d_gate_matrix, s); break;
                    case 1: apply_rz(d_sv, k, angles[k], s); break;
                    case 2: apply_ry(d_sv, k, angles[k], s); break;
                    default: apply_gate_1q(d_sv, k, d_gate_matrix, s); break;
                }
            }
            for (int k = 0; k < nbits - 1; k++) {
                if (do_cnot[k]) apply_cnot_local(d_sv, k, k+1, s);
            }
        };

        if ((mode == SimMode::Native || mode == SimMode::UVM)) {
             kernel(0, full_state_ptr, this->stream);
             CUDA_CHECK(cudaDeviceSynchronize());
        } else {
             process_pipeline(kernel);
        }
    }
}

void EdgeQuantumSim::run_vqe(int batch) {
    std::cout << "[Circuit] VQE (Ansatz Layers=" << batch << ")" << std::endl;

    std::mt19937 rng(99);
    std::normal_distribution<float> ry_dist(0.15f, 0.1f);
    std::normal_distribution<float> rz_dist(0.1f, 0.08f);

    for (int l = 0; l < batch; l++) {
        int nbits = ((mode == SimMode::Native || mode == SimMode::UVM)) ? n_qubits : chunk_bits;
        std::vector<float> ry_angles(nbits), rz_angles(nbits);
        for (int k = 0; k < nbits; k++) {
            ry_angles[k] = ry_dist(rng);
            rz_angles[k] = rz_dist(rng);
        }

        auto kernel = [this, nbits, l, &ry_angles, &rz_angles](int chunk_id, void* d_sv, cudaStream_t s) {

            int start = (l % 2);
            for (int k = start; k < nbits; k += 2) {
                apply_ry(d_sv, k, ry_angles[k], s);
                apply_rz(d_sv, k, rz_angles[k], s);
            }

            int cnot_start = (l % 2);
            for (int k = cnot_start; k < nbits - 1; k += 2) apply_cnot_local(d_sv, k, k+1, s);
        };

        if ((mode == SimMode::Native || mode == SimMode::UVM)) {
             kernel(0, full_state_ptr, this->stream);
             CUDA_CHECK(cudaDeviceSynchronize());
        } else {
             process_pipeline(kernel);
        }
    }
}
