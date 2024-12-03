
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include "sha256_cuda.h"
#include "ripemd160_cuda.h"

// Kernel to compute Bitcoin addresses in parallel
__global__ void bitcoin_address_kernel(const unsigned char* public_keys, unsigned char* bitcoin_addresses, const unsigned char* target_address, int* match_found, int num_keys) {
    extern __shared__ unsigned char shared_memory[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_keys) {
        if (*match_found) return;

        unsigned char* sha256_hash = &shared_memory[threadIdx.x * SHA256_DIGEST_SIZE];
        unsigned char* ripemd160_hash = &shared_memory[blockDim.x * SHA256_DIGEST_SIZE + threadIdx.x * RIPEMD160_DIGEST_SIZE];

        // Step 1: Apply SHA-256 to the public key
        sha256_gpu(&public_keys[idx * 33], 33, sha256_hash);

        // Step 2: Apply RIPEMD-160 to the SHA-256 hash
        ripemd160_gpu(sha256_hash, SHA256_DIGEST_SIZE, ripemd160_hash);

        // Step 3: Copy the result to the global memory
        memcpy(&bitcoin_addresses[idx * RIPEMD160_DIGEST_SIZE], ripemd160_hash, RIPEMD160_DIGEST_SIZE);

        // Step 4: Check for a match
        if (memcmp(ripemd160_hash, target_address, RIPEMD160_DIGEST_SIZE) == 0) {
            atomicExch(match_found, 1);
        }
    }
}
