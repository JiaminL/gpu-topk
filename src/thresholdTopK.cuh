#pragma once

#include <cuda.h>
#include <time.h>
#include <cub/util_allocator.cuh>
#include <cub/cub.cuh>

#include "bitonicTopK.cuh"
#include "radixSelectTopK.cuh"

using namespace cub;

template <typename KeyT, int KeyPT>
__global__ void getRandomData(KeyT* d_keys_in, uint* random_order, KeyT* d_random_keys, uint mod) {
    KeyT x[KeyPT];
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int order = random_order[offset] & mod;
    for (int i = 0; i < KeyPT; i++) x[i] = d_keys_in[order + i];
    KeyT max = x[0];
    for (int i = 1; i < KeyPT; i++)
        if (max < x[i]) max = x[i];
    d_random_keys[offset] = max;
}

template <
    typename KeyT,  // Data type of the keys within device memory. Data will be twiddled (if necessary) to unsigned type
    int KeyPT,      // Number of keys per thread
    int ThreadPB    // Number of threads per block
    >
__global__ void selectGeThreshold(KeyT* d_keys, uint num_items, KeyT threshold, uint* out_size) {
    // Specialize BlockLoad for a 1D block of ThreadPB threads owning KeyPT integer items each
    typedef cub::BlockLoad<KeyT, ThreadPB, KeyPT, BLOCK_LOAD_TRANSPOSE> BlockLoadT;

    // Specialize BlockScan type for our thread block
    typedef BlockScan<int, ThreadPB, BLOCK_SCAN_RAKING> BlockScanT;

    const int tile_size = ThreadPB * KeyPT;
    int tile_idx = blockIdx.x;  // Current tile index
    int tile_offset = tile_idx * tile_size;

    // Allocate shared memory for BlockLoad
    __shared__ union TempStorage {
        typename BlockLoadT::TempStorage load_items;
        typename BlockScanT::TempStorage scan;
        int offset[1];
        KeyT raw_exchange[ThreadPB * KeyPT];
    } temp_storage;

    // Load a segment of consecutive items that are blocked across threads
    KeyT key_entries[KeyPT];
    int selection_flags[KeyPT];
    int selection_indices[KeyPT];

    // 即 num_tiles := ceil(num_items/tile_size)
    int num_tiles = (num_items + tile_size - 1) / tile_size;
    int num_tile_items = tile_size;
    bool is_last_tile = false;
    if (tile_idx == num_tiles - 1) {
        num_tile_items = num_items - tile_offset;
        is_last_tile = true;
    }

    // Load keys
    if (is_last_tile)
        BlockLoadT(temp_storage.load_items).Load(d_keys + tile_offset, key_entries, num_tile_items);
    else
        BlockLoadT(temp_storage.load_items).Load(d_keys + tile_offset, key_entries);

    __syncthreads();

    /*** Step 1: Find keys >= threshold ***/
#pragma unroll
    for (int item = 0; item < KeyPT; ++item) {
        // Out-of-bounds items are selection_flags
        selection_flags[item] = 0;
        if (!is_last_tile || (int(threadIdx.x * KeyPT) + item < num_tile_items)) {
            selection_flags[item] = (key_entries[item] >= threshold);
        }
    }

    __syncthreads();

    // Compute exclusive prefix sum
    int num_selected;
    // 将整个 block 的 selection_flags 的前缀和输出到 selection_indices
    //（threadIdx.x 为 i 的线程，selection_indices 统计的前缀和包含了该 block 前 i 个线程的数据）
    // num_selected 为该 block 中 selection_flags 的 1 的个数（即该 block 中的 masked_key 中大于 digit_val 的个数）
    BlockScanT(temp_storage.scan).ExclusiveSum(selection_flags, selection_indices, num_selected);

    __syncthreads();

    if (num_selected > 0) {
        int index_out;
        if (threadIdx.x == 0) {
            // Find index into keys_out array
            // 由于后面的 __syncthreads()，所以所有 block 只有 0 号线程能有效工作，而且由于 out_size 只能互斥访问，所以这些线程0只能顺序工作
            // index_out 是 out_size 的 old value
            index_out = atomicAdd(out_size, num_selected);
            // 并不能确定哪个 block 的线程 0 会先执行，这 temp_storage.offset[0] 依赖于执行顺序？？？？
            temp_storage.offset[0] = index_out;
        }

        __syncthreads();

        // block 中所有线程获得同样的 index_out
        index_out = temp_storage.offset[0];

        __syncthreads();

        // Compact and scatter items
        // 将选中的元素复制到 temp_storage.raw_exchange 中前 num_selected 个位置上
#pragma unroll
        for (int item = 0; item < KeyPT; ++item) {
            int local_scatter_offset = selection_indices[item];
            if (selection_flags[item]) {
                temp_storage.raw_exchange[local_scatter_offset] = key_entries[item];
            }
        }

        __syncthreads();

        // Write out matched entries to output array
        for (int item = threadIdx.x; item < num_selected; item += ThreadPB) {
            d_keys[index_out + item] = temp_storage.raw_exchange[item];
        }

        __syncthreads();
    }
}

#define SELECT_GE_BLOCK_SIZE 320
#define SELECT_GE_ELEM_PT 16
#define GROUP_SIZE 16
#define GROUP_SHIFT 4
template <typename KeyT>
cudaError_t thresholdTopK(KeyT* d_keys_in, unsigned int num_items, unsigned int k, KeyT* d_keys_out,
                          CachingDeviceAllocator& g_allocator) {
    uint log2_n = log2_32(num_items);
    uint log2_k = log2_32(k);
    uint log2_r;
    uint sub_log = log2_n - log2_k;
    bool use_bitonic = false;
    if (sub_log >= 8) {
        log2_r = (sub_log >= 20) ? 17 : (sub_log - 3);
        uint loop_times = 3;  // 寻找 3 次 1 << log2_r 个数中的最大值，取最大值中的中值作为最终的 threshold

        // 16 个数据划为一组
        uint random_size = 1 << (log2_r - GROUP_SHIFT);
        uint mod = (uint)((1 << log2_32(num_items)) - 1) - (GROUP_SIZE - 1);
        uint block_size = min(random_size, 256);

        // 申请空间
        uint* d_random_order;
        KeyT* d_random_keys;
        KeyT* d_max;
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_random_order, sizeof(uint) * (random_size)));
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_random_keys, sizeof(KeyT) * (random_size)));
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_max, sizeof(KeyT) * loop_times));
        uint* h_random_order = (uint*)malloc(sizeof(uint) * random_size);
        KeyT* h_max = (KeyT*)malloc(sizeof(KeyT) * loop_times);

        for (int i = 0; i < loop_times; i++) {
            // 设置随机数种子
            timeval t1;
            gettimeofday(&t1, NULL);
            unsigned long long seed = t1.tv_usec * t1.tv_sec;
            srand(seed);

            // 生成随机的 radom_size 个数据
            for (int j = 0; j < random_size; j++) {
                h_random_order[j] = rand();
            }
            cudaMemcpy(d_random_order, h_random_order, sizeof(uint) * random_size, cudaMemcpyHostToDevice);

            // 将输入数据 d_keys_in 划分为 16 个一组，从 d_keys_in 中取 random_size 个组，每个线程取 16 个数，将最大的输出到 d_random_keys
            getRandomData<KeyT, GROUP_SIZE><<<random_size / block_size, block_size>>>(d_keys_in, d_random_order, d_random_keys, mod);

            // 找出 d_random_keys 中最大值
            void* d_temp_storage = NULL;
            size_t temp_storage_bytes = 0;
            DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_random_keys, d_max + i, random_size);
            CubDebugExit(g_allocator.DeviceAllocate((void**)&d_temp_storage, temp_storage_bytes));
            DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_random_keys, d_max + i, random_size);
            CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
        }

        cudaMemcpy(h_max, d_max, sizeof(KeyT) * loop_times, cudaMemcpyDeviceToHost);
        // 寻找 h_max 中的中位数
        sort(h_max, h_max + loop_times);
        KeyT threshold = h_max[loop_times / 2];

        // 释放空间
        CubDebugExit(g_allocator.DeviceFree(d_random_order));
        CubDebugExit(g_allocator.DeviceFree(d_random_keys));
        CubDebugExit(g_allocator.DeviceFree(d_max));
        free(h_random_order);
        free(h_max);

        // 将原数组中大于等于 threshold 的值都拷贝进 d_keys_in 的开始位置
        uint* d_new_len;
        uint h_new_len;
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_new_len, sizeof(uint)));
        cudaMemset(d_new_len, 0, sizeof(uint));

        uint elem_pb = SELECT_GE_BLOCK_SIZE * SELECT_GE_ELEM_PT;
        uint block_num = (num_items + elem_pb - 1) / elem_pb;
        block_size = SELECT_GE_BLOCK_SIZE;
        selectGeThreshold<KeyT, SELECT_GE_ELEM_PT, SELECT_GE_BLOCK_SIZE><<<block_num, block_size>>>(d_keys_in, num_items, threshold, d_new_len);
        cudaMemcpy(&h_new_len, d_new_len, sizeof(uint), cudaMemcpyDeviceToHost);
        CubDebugExit(g_allocator.DeviceFree(d_new_len));

        // 如果 h_new_len < k，从 d_keys_in 的前 k * 8 个元素里面寻找 top-k，否则，就从 h_keys_in 中前 d_keys_in 中寻找 top-k
        num_items = (h_new_len < k) ? (k << 3) : h_new_len;  // 由于 sub_log >= 8，一定有 k * 8 < num_items

        // 在搜索的总数据量小于等于 2^19 时，bitonic 有更好的性能（bitonic 的 k 最大只能到 512）
        uint log2_new_len = log2_32(h_new_len);
        if (k <= 512 && log2_new_len < 19) {
            // 使用 bitonic top-k 算法，搜索长度必须是 2 的幂
            num_items = (h_new_len - (1 << log2_new_len)) ? (2 << log2_new_len) : (1 << log2_new_len);
            use_bitonic = true;
        }
    } else {
        // 未进行 threshold 筛选
        use_bitonic = (k <= 512) && (log2_n <= 19) && (num_items & ((1 << log2_n) - 1) != 0);
    }

    if (use_bitonic)
        bitonicTopK(d_keys_in, num_items, k, d_keys_out, g_allocator);
    else
        radixSelectTopK(d_keys_in, num_items, k, d_keys_out, g_allocator);

    return cudaSuccess;
}