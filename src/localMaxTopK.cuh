#include <cuda.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_allocator.cuh>

using namespace std;
using namespace cub;

#include "radixSelectTopK.cuh"
#include "bitonicTopK.cuh"
#include "sortTopK.cuh"

template <typename KeyT, uint KeysPerThread>
__global__ void getLocalMax(KeyT *d_keys) {
    KeyT x[KeysPerThread];
    uint offset = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < KeysPerThread; i++) x[i] = d_keys[offset * KeysPerThread + i];
    KeyT max = x[0];
    for (int i = 1; i < KeysPerThread; i++)
        if (max < x[i]) max = x[i];
    d_keys[offset] = max;
}

template <typename KeyT, int KeysPT, int MaxItems>
__global__ void getLocalMax2(KeyT *d_keys) {
    KeyT x[KeysPT];
    int offset = KeysPT * (blockIdx.x * blockDim.x + threadIdx.x);
    int keys_per_loop = KeysPT * blockDim.x * gridDim.x;

    d_keys += offset;
    for (int j = 0; j < KeysPT; j++) x[j] = d_keys[j];
    for (int i = 1; i < MaxItems; i++) {
        d_keys += keys_per_loop;
        for (int j = 0; j < KeysPT; j++) x[j] = max(d_keys[j], x[j]);
    }
    d_keys -= (MaxItems - 1) * keys_per_loop;
    for (int j = 0; j < KeysPT; j++) d_keys[j] = x[j];
}


#define LOCAL_MAX_ITEMS 16
#define LOCAL_MAX_KPT 4
template <typename KeyT>
cudaError_t localMaxTopK(KeyT *d_keys_in, unsigned int num_items, unsigned int k, KeyT *d_keys_out, unsigned int &out_items,
                         CachingDeviceAllocator &g_allocator) {
    out_items = k;

    if (num_items >= 32 * k) {
        uint block_size = 1024;
        getLocalMax2<KeyT, LOCAL_MAX_KPT, LOCAL_MAX_ITEMS><<<num_items / block_size / LOCAL_MAX_KPT / LOCAL_MAX_ITEMS, block_size>>>(d_keys_in);
        num_items /= LOCAL_MAX_ITEMS;
    }

    uint log2_n = log2_32(num_items);
    if (log2_n < 10)
        sortTopK(d_keys_in, num_items, k, d_keys_out, out_items, g_allocator);
    else if (log2_n < 19 && k <= 1024 && num_items == (1 << log2_n))
        bitonicTopK(d_keys_in, num_items, k, d_keys_out, out_items, g_allocator);
    else
        radixSelectTopK(d_keys_in, num_items, k, d_keys_out, out_items, g_allocator);

    return cudaSuccess;
}
