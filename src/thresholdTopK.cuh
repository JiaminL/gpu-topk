#pragma once

#include <cuda.h>
#include <time.h>
#include <cub/util_allocator.cuh>
#include <cub/cub.cuh>

#include "bitonicTopK.cuh"
#include "radixSelectTopK.cuh"
#include "sortTopK.cuh"

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
template <typename KeyT, uint KeyPerT>
__global__ void MemFull(KeyT* d_keys, KeyT threshold, uint fill_num) {
    uint offset = (blockIdx.x * blockDim.x + threshold) * KeyPerT;
    if (fill_num > offset) {
        if (fill_num - offset > KeyPerT)
            for (int i = 0; i < KeyPerT; ++i) d_keys[i] = threshold;
        else
            for (int i = 0; i < fill_num - offset; ++i) d_keys[i] = threshold;
    }
}

template <typename KeyT>
cudaError_t thresholdTopK(KeyT* d_keys_in, unsigned int num_items, unsigned int k, KeyT* d_keys_out, unsigned int& out_items,
                          CachingDeviceAllocator& g_allocator) {
    uint log2_n = log2_32(num_items);
    uint log2_k = log2_32(k);
    uint log2_r;
    uint sub_log = log2_n - log2_k;
    uint loop_times = 0;

    if (sub_log >= 15) {
        // 寻找一次 max
        log2_r = sub_log - 10;
        loop_times = 1;
    } else if (sub_log >= 10) {
        log2_r = (log2_n < 19) ? 4 : sub_log - 6;
        loop_times = 3;
    } else if (sub_log >= 8 && log2_n >= 19) {
        log2_r = sub_log - 5;
        loop_times = 5;
    }

    if (loop_times > 0) {
        uint random_size = (log2_r <= GROUP_SHIFT) ? 1 : (1 << (log2_r - GROUP_SHIFT));
        uint mod = (uint)((1 << log2_n) - 1) - (GROUP_SIZE - 1);
        uint block_size = min(random_size, 256);
        KeyT threshold;
        KeyT h_max[5];

        // 设置随机数种子
        timeval t1;
        gettimeofday(&t1, NULL);
        unsigned long long seed = t1.tv_usec * t1.tv_sec;
        srand(seed);

        if (log2_r >= 6) {
            // 用 kernel 取数并寻找 max
            uint* d_random_order;
            KeyT* d_random_keys;
            KeyT* d_max;
            // 申请空间
            CubDebugExit(g_allocator.DeviceAllocate((void**)&d_random_order, sizeof(uint) * (random_size)));
            CubDebugExit(g_allocator.DeviceAllocate((void**)&d_random_keys, sizeof(KeyT) * (random_size)));
            CubDebugExit(g_allocator.DeviceAllocate((void**)&d_max, sizeof(KeyT) * loop_times));
            uint* h_random_order = (uint*)malloc(sizeof(uint) * random_size);

            for (int i = 0; i < loop_times; i++) {
                // 生成随机的 radom_size 个数据
                for (int j = 0; j < random_size; j++) h_random_order[j] = rand();
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
            CubDebugExit(g_allocator.DeviceFree(d_random_order));
            CubDebugExit(g_allocator.DeviceFree(d_random_keys));
            CubDebugExit(g_allocator.DeviceFree(d_max));
            free(h_random_order);
        } else {
            // 用 CPU 取数并寻找 max
            for (int i = 0; i < loop_times; i++) {
                // 生成随机的 radom_size 个数据
                KeyT x[GROUP_SIZE];
                int size = 1 << ((log2_r < GROUP_SHIFT) ? log2_r : GROUP_SHIFT);
                for (uint offset = 0; offset < random_size; offset++) {
                    int order = rand() & mod;
                    cudaMemcpy(x, d_keys_in + order, sizeof(KeyT) * size, cudaMemcpyDeviceToHost);
                    if (offset == 0 || h_max[i] < x[0]) h_max[i] = x[0];
                    for (int j = 1; j < size; j++)
                        if (h_max[i] < x[j]) h_max[i] = x[j];
                }
            }
        }

        // 寻找 h_max 中的中位数
        sort(h_max, h_max + loop_times);
        threshold = h_max[loop_times / 2];

        // 分配空间
        uint* d_new_len;
        uint h_new_len;
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_new_len, sizeof(uint)));
        cudaMemset(d_new_len, 0, sizeof(uint));

        // 寻找大于等于 threshold 的元素，为 h_new_len，放入原数组的空间
        uint elem_pb = SELECT_GE_BLOCK_SIZE * SELECT_GE_ELEM_PT;
        uint block_num = (num_items + elem_pb - 1) / elem_pb;
        block_size = SELECT_GE_BLOCK_SIZE;
        selectGeThreshold<KeyT, SELECT_GE_ELEM_PT, SELECT_GE_BLOCK_SIZE><<<block_num, block_size>>>(d_keys_in, num_items, threshold, d_new_len);
        cudaMemcpy(&h_new_len, d_new_len, sizeof(uint), cudaMemcpyDeviceToHost);
        CubDebugExit(g_allocator.DeviceFree(d_new_len));

        // 更新数组大小

        num_items = h_new_len;

        // 填充耗时比较多, 所以舍弃
        // 在搜索的总数据量小于等于 2^19 时，bitonic 有更好的性能
        uint new_log2_n = log2_32(h_new_len);
        if (k <= 1024 && new_log2_n < 18) {
            // 但是该算法数据集大小必须是 2 的幂
            if (num_items > (1 << new_log2_n)) {
                // 用 threshold 将数组填充到 2 的幂
                uint fill_num = (2 << new_log2_n) - num_items;
                uint block_size = min(fill_num, 256);
                uint block_num = (fill_num + GROUP_SIZE * block_size - 1) / GROUP_SIZE * block_size;
                MemFull<KeyT, GROUP_SIZE><<<block_num, block_size>>>(d_keys_in + h_new_len, threshold, fill_num);
                num_items = 2 << new_log2_n;
            }
        }
    }

    out_items = k;
    if (num_items <= k) {
        // 筛选后的数据量已小于 k 个
        out_items = num_items;
        cudaMemcpy(d_keys_out, d_keys_in, num_items * sizeof(KeyT), cudaMemcpyDeviceToDevice);
    } else {
        log2_n = log2_32(num_items);
        // 在搜索的总数据量小于等于 2^19 时，bitonic 有更好的性能
        if (log2_n < 10)
            sortTopK(d_keys_in, num_items, k, d_keys_out, out_items, g_allocator);
        else if (num_items == (1 << log2_n) && k <= 1024 && log2_n <= 18)
            bitonicTopK(d_keys_in, num_items, k, d_keys_out, out_items, g_allocator);
        else
            radixSelectTopK(d_keys_in, num_items, k, d_keys_out, out_items, g_allocator);
    }
    return cudaSuccess;
}

// template <typename KeyT>
// cudaError_t testTimeFindMax(KeyT* d_keys_in, unsigned int num_items, unsigned int k, KeyT* d_keys_out, uint m_log,
//                      CachingDeviceAllocator& g_allocator) {
// uint log2_n = log2_32(num_items);
// uint log2_k = log2_32(k);
// uint log2_r = m_log;
// uint loop_times = 1;

// if (loop_times > 0) {
//     uint random_size = (log2_r <= GROUP_SHIFT) ? 1 : (1 << (log2_r - GROUP_SHIFT));
//     uint mod = (uint)((1 << log2_n) - 1) - (GROUP_SIZE - 1);
//     uint block_size = min(random_size, 256);
//     KeyT h_max[3];

//     // 设置随机数种子
//     timeval t1;
//     gettimeofday(&t1, NULL);
//     unsigned long long seed = t1.tv_usec * t1.tv_sec;
//     srand(seed);

//     if (log2_r >= 21) {
//         // 用 kernel 取数并寻找 max
//         uint* d_random_order;
//         KeyT* d_random_keys;
//         KeyT* d_max;
//         // 申请空间
//         CubDebugExit(g_allocator.DeviceAllocate((void**)&d_random_order, sizeof(uint) * (random_size)));
//         CubDebugExit(g_allocator.DeviceAllocate((void**)&d_random_keys, sizeof(KeyT) * (random_size)));
//         CubDebugExit(g_allocator.DeviceAllocate((void**)&d_max, sizeof(KeyT) * loop_times));

//         for (int i = 0; i < loop_times; i++) {
//             // 生成随机的 radom_size 个数据
//             curandGenerator_t generator;
//             curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
//             curandSetPseudoRandomGeneratorSeed(generator, seed);
//             curandGenerate(generator, d_random_order, random_size);
//             curandDestroyGenerator(generator);

//             // 将输入数据 d_keys_in 划分为 16 个一组，从 d_keys_in 中取 random_size 个组，每个线程取 16 个数，将最大的输出到 d_random_keys
//             getRandomData<KeyT, GROUP_SIZE><<<random_size / block_size, block_size>>>(d_keys_in, d_random_order, d_random_keys, mod);
//             // 找出 d_random_keys 中最大值
//             void* d_temp_storage = NULL;
//             size_t temp_storage_bytes = 0;
//             DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_random_keys, d_max + i, random_size);
//             CubDebugExit(g_allocator.DeviceAllocate((void**)&d_temp_storage, temp_storage_bytes));
//             DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_random_keys, d_max + i, random_size);
//             CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
//         }

//         cudaMemcpy(h_max, d_max, sizeof(KeyT) * loop_times, cudaMemcpyDeviceToHost);
//         CubDebugExit(g_allocator.DeviceFree(d_random_order));
//         CubDebugExit(g_allocator.DeviceFree(d_random_keys));
//         CubDebugExit(g_allocator.DeviceFree(d_max));
//     } else if (log2_r >= 6) {
//         // 用 kernel 取数并寻找 max
//         uint* d_random_order;
//         KeyT* d_random_keys;
//         KeyT* d_max;
//         // 申请空间
//         CubDebugExit(g_allocator.DeviceAllocate((void**)&d_random_order, sizeof(uint) * (random_size)));
//         CubDebugExit(g_allocator.DeviceAllocate((void**)&d_random_keys, sizeof(KeyT) * (random_size)));
//         CubDebugExit(g_allocator.DeviceAllocate((void**)&d_max, sizeof(KeyT) * loop_times));
//         uint* h_random_order = (uint*)malloc(sizeof(uint) * random_size);

//         for (int i = 0; i < loop_times; i++) {
//             // 生成随机的 radom_size 个数据
//             if (log2_r >= 22) {
//                 curandGenerator_t generator;
//                 curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
//                 curandSetPseudoRandomGeneratorSeed(generator, seed);
//                 curandGenerate(generator, d_random_order, random_size);
//                 curandDestroyGenerator(generator);
//             }
//             for (int j = 0; j < random_size; j++) h_random_order[j] = rand();
//             cudaMemcpy(d_random_order, h_random_order, sizeof(uint) * random_size, cudaMemcpyHostToDevice);
//             // 将输入数据 d_keys_in 划分为 16 个一组，从 d_keys_in 中取 random_size 个组，每个线程取 16 个数，将最大的输出到 d_random_keys
//             getRandomData<KeyT, GROUP_SIZE><<<random_size / block_size, block_size>>>(d_keys_in, d_random_order, d_random_keys, mod);
//             // 找出 d_random_keys 中最大值
//             void* d_temp_storage = NULL;
//             size_t temp_storage_bytes = 0;
//             DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_random_keys, d_max + i, random_size);
//             CubDebugExit(g_allocator.DeviceAllocate((void**)&d_temp_storage, temp_storage_bytes));
//             DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_random_keys, d_max + i, random_size);
//             CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
//         }

//         cudaMemcpy(h_max, d_max, sizeof(KeyT) * loop_times, cudaMemcpyDeviceToHost);
//         CubDebugExit(g_allocator.DeviceFree(d_random_order));
//         CubDebugExit(g_allocator.DeviceFree(d_random_keys));
//         CubDebugExit(g_allocator.DeviceFree(d_max));
//         free(h_random_order);
//     } else {
//         // 用 CPU 取数并寻找 max
//         for (int i = 0; i < loop_times; i++) {
//             // 生成随机的 radom_size 个数据
//             KeyT x[GROUP_SIZE];
//             int size = 1 << ((log2_r < GROUP_SHIFT) ? log2_r : GROUP_SHIFT);
//             for (uint offset = 0; offset < random_size; offset++) {
//                 int order = rand() & mod;
//                 cudaMemcpy(x, d_keys_in + order, sizeof(KeyT) * size, cudaMemcpyDeviceToHost);
//                 if (offset == 0 || h_max[i] < x[0]) h_max[i] = x[0];
//                 for (int j = 1; j < size; j++)
//                     if (h_max[i] < x[j]) h_max[i] = x[j];
//             }
//         }
//     }

//     // 寻找 h_max 中的中位数
//     sort(h_max, h_max + loop_times);
//     threshold = h_max[loop_times / 2];

// // 分配空间
// uint* d_new_len;
// uint h_new_len;
// CubDebugExit(g_allocator.DeviceAllocate((void**)&d_new_len, sizeof(uint)));
// cudaMemset(d_new_len, 0, sizeof(uint));
// KeyT threshold = (KeyT)0;

// // // 寻找大于等于 threshold 的元素，为 h_new_len，放入原数组的空间
// uint elem_pb = SELECT_GE_BLOCK_SIZE * SELECT_GE_ELEM_PT;
// uint block_num = (num_items + elem_pb - 1) / elem_pb;
// uint block_size = SELECT_GE_BLOCK_SIZE;
// selectGeThreshold<KeyT, SELECT_GE_ELEM_PT, SELECT_GE_BLOCK_SIZE><<<block_num, block_size>>>(d_keys_in, num_items, threshold, d_new_len);
// cudaMemcpy(&h_new_len, d_new_len, sizeof(uint), cudaMemcpyDeviceToHost);
// CubDebugExit(g_allocator.DeviceFree(d_new_len));

// // 更新数组大小

// num_items = h_new_len;

// 填充耗时比较多, 所以舍弃
// 在搜索的总数据量小于等于 2^19 时，bitonic 有更好的性能
// uint new_log2_n = log2_32(h_new_len);
// if (k <= 1024 && new_log2_n < 19) {
//     // 但是该算法数据集大小必须是 2 的幂
//     if (num_items > (1 << new_log2_n)) {
//         // 用 threshold 将数组填充到 2 的幂
//         uint fill_num = (2 << new_log2_n) - num_items;
//         uint block_size = min(fill_num, 256);
//         uint block_num = (fill_num + GROUP_SIZE * block_size - 1) / GROUP_SIZE * block_size;
//         MemFull<KeyT, GROUP_SIZE><<<block_num, block_size>>>(d_keys_in + h_new_len, threshold, fill_num);
//         num_items = 2 << new_log2_n;
//     }
// }
// }

// out_items = k;
// if (num_items <= k) {
//     // 筛选后的数据量已小于 k 个
//     out_items = num_items;
//     cudaMemcpy(d_keys_out, d_keys_in, num_items * sizeof(KeyT), cudaMemcpyDeviceToDevice);
// } else {
//     log2_n = log2_32(num_items);
//     // 在搜索的总数据量小于等于 2^19 时，bitonic 有更好的性能
//     if (num_items == (1 << log2_n) && k <= 1024 && log2_n <= 19)
//         bitonicTopK(d_keys_in, num_items, k, d_keys_out, out_items, g_allocator);
//     else
//         radixSelectTopK(d_keys_in, num_items, k, d_keys_out, out_items, g_allocator);
// }
//     return cudaSuccess;
// }
