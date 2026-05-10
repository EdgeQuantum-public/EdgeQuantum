#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <custatevec.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

#define CUSV_CHECK(call) \
    do { \
        custatevecStatus_t err = call; \
        if (err != CUSTATEVEC_STATUS_SUCCESS) { \
            std::cerr << "cuStateVec Error: " << err \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)
