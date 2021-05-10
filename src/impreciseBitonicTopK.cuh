#pragma once

#include <cuda.h>
#include <cub/util_allocator.cuh>
#include <algorithm>

#include "bitonicUsing.cuh"
#include "sharedmem.cuh"

using namespace std;
using namespace cub;

template <typename T>
__global__ void ImpreBitonic_TopKLocalSortInPlace(T* data, const int k, const int klog2, const int impre_pe) {
    // const int k = K;
    // const int klog2 = KLog2;

    // Shared mem size is determined by the host app at run time.
    // For n elements, we have n * 33/32 shared memory.
    // We use this to break bank conflicts.
    SharedMemory<T> smem;
    T* sdata = smem.getPointer();
    T *in = data, *out = data;

    const int t = threadIdx.x;  // index in workgroup
    const int wg = blockDim.x;  // workgroup size = block size, power of 2
    const int gid = blockIdx.x;
    const int g_num = gridDim.x;  // workgroup 数量

    int length = min(NUM_GROUPS, k >> 1);
    int inc = length;
    inc >>= NUM_GROUPS_BITSHIFT;
    int low = t & (inc - 1);
    int dir = length << 1;
    bool reverse;

    T x[NUM_ELEM_PT];

    // Move IN, OUT to block start
    // gid = blockIdx.x, wg = blockDim.x, in 加上了前面 block 处理的元素数量
    in += NUM_ELEM_PT * gid * wg;

    // Step1: Complete the remaining steps to create sorted sequences of length k.
    // t = threadId.x, in 是原始数据数组，而 NUM_ELEM_PT = 1 << NUM_ELEM_BITSHIFT，
    // 从而下面这两行代码将该线程需要处理的连续 NUM_ELEM_PT 个数据都放到 local memory 中的 x 数组里面了
    int tCur = t << NUM_ELEM_BITSHIFT;
    for (int j = 0; j < NUM_ELEM_PT; j++) x[j] = in[tCur + j];

    for (int i = 1; i < impre_pe; i++) {
        in += NUM_ELEM_PT * wg * g_num;
        for (int j = 0; j < NUM_ELEM_PT; j++) x[j] = max(in[tCur + j], x[j]);
    }

    // 将数组 x 变为每段排序长度为 min(NUM_ELEM_PT, k) 的交替升降的序列
    for (int i = 0; i < NUM_ELEM_PT; i += 2) {
        reverse = ((i >> 1) + 1) & 1;
        B2V(x, i);
    }
    if (k > 2) {
#if NUM_ELEM_PT > 4
        for (int i = 0; i < NUM_ELEM_PT; i += 4) {
            reverse = ((i >> 2) + 1) & 1;
            B4V(x, i);
        }
        if (k > 4) {
#if NUM_ELEM_PT > 8
            for (int i = 0; i < NUM_ELEM_PT; i += 8) {
                reverse = ((i >> 3) + 1) & 1;
                B8V(x, i);
            }
            if (k > 8) {
#if NUM_ELEM_PT > 16
                for (int i = 0; i < NUM_ELEM_PT; i += 16) {
                    reverse = ((i >> 4) + 1) & 1;
                    B16V(x, i);
                }
                if (k > 16) {
#if NUM_ELEM_PT > 32
                    for (int i = 0; i < NUM_ELEM_PT; i += 32) {
                        reverse = ((i >> 5) + 1) & 1;
                        B32V(x, i);
                    }
                    if (k > 32) {
                        // 意即，NUM_ELEM_PT 最大只支持 64
                        reverse = ((dir & tCur) == 0);
                        B64V(x, 0);
                    }
#else
                    reverse = ((dir & tCur) == 0);
                    B32V(x, 0);
#endif
                }
#else
                reverse = ((dir & tCur) == 0);
                B16V(x, 0);
#endif
            }
#else
            reverse = ((dir & tCur) == 0);
            B8V(x, 0);
#endif
        }
#else
        // 只有 NUM_ELEM_PT = 4 使下面代码合法，NUM_ELEM_PT < 4 会使得 x 数组越界
        reverse = ((dir & tCur) == 0);
        B4V(x, 0);
#endif
    }

    // set: temp = tCur + i; sdata[temp + (temp >> 5)] = x[i]
    // 每 32 个数据就空出一个位置，以消除 bank 冲突
    for (int i = 0; i < NUM_ELEM_PT; i++) set(sdata, tCur + i, x[i]);

    __syncthreads();

    int mod;
    unsigned int mask;

    // 当 NUM_ELEM_PT < k 时，下面的循环才会执行，初始序列为排序段长度为 NUM_ELEM_PT 的序列
    // 而当 NUM_ELEM_PT >= k 时，序列的排序段长度已经是 k 了
    for (length = NUM_ELEM_PT; length < k; length <<= 1) {
        dir = length << 1;
        // Loop on comparison distance (between keys)
        inc = length;
        mod = inc;
        mask = ~(NUM_ELEM_PT / (1) - 1);
        // 意即：mod = 1 << (log(length) % NUM_ELEM_BITSHIFT)
        while ((mod & mask) != 0) mod >>= (NUM_ELEM_BITSHIFT - 0);

        if (mod & 1) {
            RUN_2(1)
            __syncthreads();
        }
        if (mod & 2) {
            RUN_4(1)
            __syncthreads();
        }
#if NUM_ELEM_PT > 8
        if (mod & 4) {
            RUN_8(1)
            __syncthreads();
        }
#if NUM_ELEM_PT > 16
        // 最大只支持到 NUM_ELEM_PT = 32，缺少 NUM_ELEM_PT = 64 的代码
        if (mod & 8) {
            RUN_16(1)
            __syncthreads();
        }
        while (inc > 8) {
            RUN_32(1)
            __syncthreads();
        }
#else   // when NUM_ELEM_PT = 16
        while (inc > 4) {
            RUN_16(1)
            __syncthreads();
        }
#endif  // NUM_ELEM_PT > 16
#else   // when NUM_ELEM_PT = 4 或 8                                                                                              \
        // 缺少了 NUM_ELEM_PT = 4 的代码
        while (inc > 2) {
            RUN_8(1)
            __syncthreads();
        }
#endif  // NUM_ELEM_PT > 8
    }

    // Step 2: Reduce the size by factor 2 by pairwise comparing adjacent sequences.
    REDUCE(1)
    __syncthreads();
    // 执行结束后，sdata 中的数据变为一系列长度为 k 的双调序列（由相邻的两个长度k的排序序列中对应位置较大值组成），并且 sdata 有效数据减半

    // Step 3: Construct sorted sequence of length k from bitonic sequence of length k.
    // We now have n/2 elements.
    length = k >> 1;
    dir = length << 1;
    // Loop on comparison distance (between keys)
    inc = length;
    mod = inc;
    mask = ~(NUM_ELEM_PT / (1) - 1);
    while ((mod & mask) != 0) mod >>= (NUM_ELEM_BITSHIFT - 0);

    if (mod & 1) {
        RUN_2(2)
        __syncthreads();
    }
#if NUM_ELEM_PT > 4
    if (mod & 2) {
        RUN_4(2)
        __syncthreads();
    }
#if NUM_ELEM_PT > 8
    if (mod & 4) {
        RUN_8(2)
        __syncthreads();
    }
    // 只适用于 NUM_ELEM_PT = 16
    while (inc > 4) {
        // 只让前一半的进程进行排序
        if (t < (wg >> 1)) {
            RUN_16(1)
        } else {
            inc >>= 4;
        }
        __syncthreads();
    }
#else   // NUM_ELEM_PT = 8
    while (inc > 2) {
        RUN_8(2)
        __syncthreads();
    }
#endif  // NUM_ELEM_PT > 16
#else   // NUM_ELEM_PT = 4
    while (inc > 1) {
        RUN_4(2)
        __syncthreads();
    }
#endif  // NUM_ELEM_PT > 8
    // 执行结束，重新获得排序长度为 k 的升降交替的序列，不过数据量变为原来的一半

    // Step 4: Reduce size again by 2.
    REDUCE(2)
    __syncthreads();
    // 运行结果，获得一系列长度为k的双调序列，总数据量是原始数据量的 1/4

    // Step 5: Construct sorted sequence of length k from bitonic sequence of length k.
    length = k >> 1;
    dir = length << 1;
    // Loop on comparison distance (between keys)
    inc = length;
    mod = inc;
    mask = ~(NUM_ELEM_PT / (2) - 1);
    while ((mod & mask) != 0) mod >>= (NUM_ELEM_BITSHIFT - 1);

#if NUM_ELEM_PT > 4
    if (mod & 1) {
        RUN_2(4)
        __syncthreads();
    }
#if NUM_ELEM_PT > 8
    if (mod & 2) {
        RUN_4(4)
        __syncthreads();
    }
    // 只适用于 NUM_ELEM_PT = 16
    while (inc > 2) {
        if (t < (wg >> 1)) {
            RUN_8(2)
        } else {
            inc >>= 3;
        }
        __syncthreads();
    }
#else   // NUM_ELEM_PT = 8
    while (inc > 1) {
        RUN_4(4)
        __syncthreads();
    }
#endif  // NUM_ELEM_PT > 16
#else   // NUM_ELEM_PT = 4
    while (inc > 0) {
        RUN_2(4)
        __syncthreads();
    }
#endif  // NUM_ELEM_PT > 8 while (inc > 0)
    // 执行结束，重新获得排序长度为 k 的升降交替的序列，不过数据量变为原始数据的 1/4

    // Step 6: Reduce size again by 2.
    REDUCE(4)
    __syncthreads();
    // 运行结果，获得一系列长度为k的双调序列，总数据量是原始数据量的 1/8

    // Step 7: Construct sorted sequence of length k from bitonic sequence of length k.
    length = k >> 1;
    dir = length << 1;
    // Loop on comparison distance (between keys)
    inc = length;
    mod = inc;
    mask = ~(NUM_ELEM_PT / (4) - 1);
    // 当 NUM_ELEM_PT = 4 时，NUM_ELEM_BITSHIFT = 2，如下 while 语句会陷入死循环，所以，NUM_ELEM_PT != 4
    while ((mod & mask) != 0) mod >>= (NUM_ELEM_BITSHIFT - 2);

    if (mod & 1) {
        RUN_2(8)
        __syncthreads();
    }
    while (inc > 0) {
        if (t < (wg >> 1)) {
            // 只适用于 NUM_ELEM_PT = 16，
            // 如果 NUM_ELEM_PT = 8, 最后一次 RUN_4(4) 开始时 inc = 1，执行第一行以后，inc 变为 0，
            // 然后后面要排序的 x 数组就都是从同一个 share memory 地址中取出来的了，操作不对
            RUN_4(4)
        } else {
            inc >>= 2;
        }
        __syncthreads();
    }
    // 执行结束，重新获得排序长度为 k 的升降交替的序列，不过数据量变为原始数据的 1/8

    // Step 8: Reduce size again by 2.
    out += (NUM_ELEM_PT / 16) * gid * wg;
    tCur = ((t >> klog2) << (klog2 + 1)) + (t & (k - 1));
    for (int j = 0; j < NUM_GROUPS / 8; j++) {
        T x0 = get(sdata, 2 * wg * j + tCur);
        T x1 = get(sdata, 2 * wg * j + tCur + k);
        out[wg * j + t] = max(x0, x1);
    }
    // 最终输出，一系列长度k的双调序列，总数据量：原数据的 1/16
}

template <typename KeyT>
cudaError_t impreciseBitonicTopK(KeyT* d_keys_in, unsigned int num_items, unsigned int k, KeyT* d_keys_out,
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

    int nlog2 = log2_32(num_items);
    int impre_pe_shift = nlog2 - 6 - 2 * klog2;
    impre_pe_shift = (impre_pe_shift < 2) ? 0 : (impre_pe_shift > 4) ? 4
                                                                     : impre_pe_shift;
    int impre_pe = 1 << impre_pe_shift;

    if (num_items >= (wg_size << NUM_ELEM_BITSHIFT << impre_pe_shift)) {
        // 如果 num_items >= 16 * impre_pe * wg_size，每个 thread 读取 16 * impre_pe 个数据，但是只会保留 16 个局部最大用来做后面的运算，
        // 然后每个 kernel 进行 4 次 reduce
        numThreads >>= NUM_ELEM_BITSHIFT + impre_pe_shift;  // Each thread processes 16 * 8 elements.
        share_mem_size = ((NUM_ELEM_PT * wg_size * 33) / 32) * sizeof(KeyT);
        ImpreBitonic_TopKLocalSortInPlace<KeyT><<<numThreads / wg_size, wg_size, share_mem_size>>>(d_keys_in, k, klog2, impre_pe);
    } else if (num_items > wg_size) {
        // 如果 wg_size < num_items < impre_pe * 8 * wg_size，local sort 每个线程处理 2 个数据，然后进行一次 reduce
        numThreads >>= 1;  // Each thread processes 2 elements.
        share_mem_size = ((2 * wg_size * 33) / 32) * sizeof(KeyT);
        Bitonic_TopKLocalSortOneReduce<KeyT><<<numThreads / wg_size, wg_size, share_mem_size>>>(d_keys_in, k, klog2);
    }

    while (numThreads >= (wg_size << NUM_ELEM_BITSHIFT)) {
        numThreads >>= NUM_ELEM_BITSHIFT;  // Each thread processes 16 elements.
        Bitonic_TopKReduce<KeyT><<<numThreads / wg_size, wg_size, share_mem_size>>>(d_keys_in, k, klog2, NUM_ELEM_PT);
    }

    if (numThreads > wg_size) {
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
