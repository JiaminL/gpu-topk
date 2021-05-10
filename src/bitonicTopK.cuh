#pragma once

#include <cuda.h>
#include <cub/util_allocator.cuh>
#include <algorithm>

#include "bitonicUsing.cuh"
#include "sharedmem.cuh"

using namespace std;
using namespace cub;

#define MAX_SHARE_MEM_SIZE 49152  // Geforce RTX 2070 Super 每个 block 最大的 share memory 大小（单位：byte）

template <typename KeyT>
cudaError_t bitonicTopK(KeyT* d_keys_in, unsigned int num_items, unsigned int k, KeyT* d_keys_out,
                        CachingDeviceAllocator& g_allocator) {
    int old_k = k;
    int klog2 = log2_32(k);
    if ((1 << klog2) < k) {
        klog2++;
        k = 1 << klog2;
    }

    int numThreads = num_items;
    int share_mem_size;

    int wg_size = max(64, k);
    int max_reduce_times = log2_32(MAX_SHARE_MEM_SIZE / sizeof(KeyT) * 32 / 33 / wg_size);
    bool share_too_small = max_reduce_times < NUM_ELEM_BITSHIFT;

    if (!share_too_small && num_items >= (wg_size << NUM_ELEM_BITSHIFT)) {
        // 如果 num_items >= 16 * wg_size，每个 kernel 将进行 4 次 reduce
        numThreads >>= NUM_ELEM_BITSHIFT;  // Each thread processes 16 elements.
        share_mem_size = ((NUM_ELEM_PT * wg_size * 33) / 32) * sizeof(KeyT);
        Bitonic_TopKLocalSortInPlace<KeyT><<<numThreads / wg_size, wg_size, share_mem_size>>>(d_keys_in, k, klog2);

        while (numThreads >= (wg_size << NUM_ELEM_BITSHIFT)) {
            numThreads >>= NUM_ELEM_BITSHIFT;  // Each thread processes 16 elements.
            Bitonic_TopKReduce<KeyT><<<numThreads / wg_size, wg_size, share_mem_size>>>(d_keys_in, k, klog2, NUM_ELEM_PT);
        }

    } else if (!share_too_small && num_items > wg_size) {
        // 如果 wg_size < num_items < 16 * wg_size，local sort 每个线程处理 2 个数据，然后进行一次 reduce
        numThreads >>= 1;  // Each thread processes 2 elements.
        share_mem_size = ((2 * wg_size * 33) / 32) * sizeof(KeyT);
        Bitonic_TopKLocalSortOneReduce<KeyT><<<numThreads / wg_size, wg_size, share_mem_size>>>(d_keys_in, k, klog2);

    } else if (share_too_small && num_items > wg_size && max_reduce_times >= 1) {
        // share memory 不足以放 4 次 reduce所需的数据，并且至少可以做一次 reduce
        numThreads >>= 1;  // Each thread processes 2 elements.
        share_mem_size = ((2 * wg_size * 33) / 32) * sizeof(KeyT);
        Bitonic_TopKLocalSortOneReduce<KeyT><<<numThreads / wg_size, wg_size, share_mem_size>>>(d_keys_in, k, klog2);
        
        while (numThreads >= (wg_size << max_reduce_times)) {
            numThreads >>= max_reduce_times;  // Each thread processes 2^max_reduce_times elements.
            share_mem_size = (((1 << max_reduce_times) * wg_size * 33) / 32) * sizeof(KeyT);
            Bitonic_TopKReduce<KeyT><<<numThreads / wg_size, wg_size, share_mem_size>>>(d_keys_in, k, klog2, max_reduce_times);
        }
    }

    if (numThreads > wg_size && max_reduce_times >= 1) {
        // 如果剩下的数据还是比 wg_size 大，继续 reduce，次数为 reduce_times
        int reduce_times = log2_32(numThreads / wg_size);
        numThreads >>= reduce_times;
        share_mem_size = (((wg_size << reduce_times) * 33) / 32) * sizeof(KeyT);
        Bitonic_TopKReduce<KeyT><<<1, wg_size, share_mem_size>>>(d_keys_in, k, klog2, reduce_times);
    }

    // 此时，筛选剩余 wg_size 个数据
    if (numThreads > old_k) {
        // 如果剩下的数据比 old_k 大，将筛选出的数据排序，取出前 old_k 个数据作为结果返回
        KeyT* res_vec = (KeyT*)malloc(sizeof(KeyT) * numThreads);
        CubDebugExit(cudaMemcpy(res_vec, d_keys_in, numThreads * sizeof(KeyT), cudaMemcpyDeviceToHost));
        std::sort(res_vec, res_vec + numThreads, std::greater<KeyT>());
        CubDebugExit(cudaMemcpy(d_keys_out, res_vec, old_k * sizeof(KeyT), cudaMemcpyHostToDevice));
        free(res_vec);
    } else {
        CubDebugExit(cudaMemcpy(d_keys_out, d_keys_in, old_k * sizeof(KeyT), cudaMemcpyDeviceToDevice));
    }

    return cudaSuccess;
}
