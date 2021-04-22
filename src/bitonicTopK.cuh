#pragma once

#include <cuda.h>
#include <cub/util_allocator.cuh>
#include <algorithm>

#include "sharedmem.cuh"

using namespace std;
using namespace cub;

// 通过后面的代码，知道，NUM_ELEM_PT 只能是 16，而且正好等于 1 << NUM_ELEM_BITSHIFT
#define NUM_ELEM_PT 16
#define NUM_ELEM_BITSHIFT 4

//#define K 32
//#define KLog2 5

#define ORDERV(x, a, b)                      \
    {                                        \
        bool swap = reverse ^ (x[a] < x[b]); \
        T auxa = x[a];                       \
        if (swap) {                          \
            x[a] = x[b];                     \
            x[b] = auxa;                     \
        }                                    \
    }
// T auxa = x[a]; T auxb = x[b]; \
// x[a] = (swap)?auxb:auxa; x[b] = (swap)?auxa:auxb;}

// 要能排好序，首先，x 得是一个双调序列
#define B2V(x, a) \
    { ORDERV(x, a, a + 1) }
#define B4V(x, a)                         \
    {                                     \
        for (int i4 = 0; i4 < 2; i4++) {  \
            ORDERV(x, a + i4, a + i4 + 2) \
        }                                 \
        B2V(x, a)                         \
        B2V(x, a + 2)                     \
    }
#define B8V(x, a)                         \
    {                                     \
        for (int i8 = 0; i8 < 4; i8++) {  \
            ORDERV(x, a + i8, a + i8 + 4) \
        }                                 \
        B4V(x, a)                         \
        B4V(x, a + 4)                     \
    }
#define B16V(x, a)                          \
    {                                       \
        for (int i16 = 0; i16 < 8; i16++) { \
            ORDERV(x, a + i16, a + i16 + 8) \
        }                                   \
        B8V(x, a)                           \
        B8V(x, a + 8)                       \
    }
#define B32V(x, a)                           \
    {                                        \
        for (int i32 = 0; i32 < 16; i32++) { \
            ORDERV(x, a + i32, a + i32 + 16) \
        }                                    \
        B16V(x, a)                           \
        B16V(x, a + 16)                      \
    }
#define B64V(x, a)                           \
    {                                        \
        for (int i64 = 0; i64 < 32; i64++) { \
            ORDERV(x, a + i64, a + i64 + 32) \
        }                                    \
        B32V(x, a)                           \
        B32V(x, a + 32)                      \
    }

template <typename T>
__forceinline__
    __device__ T
    get(T* sdata, int i) {
    return sdata[i + (i >> 5)];
}

#define set(a, b, c)                         \
    {                                        \
        int tempIndex = b;                   \
        a[tempIndex + (tempIndex >> 5)] = c; \
    }

#define NUM_GROUPS (NUM_ELEM_PT / 2)
#define NUM_GROUPS_BITSHIFT (NUM_ELEM_BITSHIFT - 1)

// RUN_m(X)：（m 是 2,4,8,16,32,64）
// X 的含义：
//      如果 block 中前 1/y 的进程运行了 RUN_m(X)，那么 wg * NUM_ELEM_PT / (X * y) 正好是 sdata 中有效数据的长度
//      （如果全部都执行 y = 1）
// 运行之前：
//      令 len = 2 * inc，sdata 中的数组每 m * len 个数据正好会组成一个双调序列
// 其他变量含义：
//      1. t：线程 id
//      2. wg：块中线程数量
//      2. dir：目标排序长度（2 的幂）
// 内层循环：
//      每个线程每隔 inc 取一个数，共取 m 个数来排序，所以内层循环一共处理 wg/y * m 个数据
// 外存循环：
//      共执行内层循环 NUM_ELEM_PT / (m * X) 次，每次推进 wg/y * m 个数据，执行完毕，所有数据都处理过了
// 运行结果：
//      原来的一个双调序列拆为 m 个长度为 len 的双调序列，其比较方向依赖于 dir，
// <？？？？>: m = 8, 16, 32, 64 时，内层次循环并未给数据编号加 m * wg，可能是 bug
// 为什么要在数组 x 中完成交换？直接在 share memory 里面不行吗？
#define RUN_64(X)                                                           \
    {                                                                       \
        inc >>= 5;                                                          \
        low = t & (inc - 1);                                                \
        tCur = ((t - low) << 6) + low;                                      \
        reverse = ((dir & tCur) == 0);                                      \
        for (int j = 0; j < NUM_GROUPS / (32 * X); j++) {                   \
            for (int i = 0; i < 64; i++) x[i] = get(sdata, tCur + i * inc); \
            B64V(x, 0);                                                     \
            for (int i = 0; i < 64; i++) set(sdata, tCur + i * inc, x[i]);  \
        }                                                                   \
        inc >>= 1;                                                          \
    }

#define RUN_32(X)                                                                         \
    {                                                                                     \
        inc >>= 4;                                                                        \
        low = t & (inc - 1);                                                              \
        tCur = ((t - low) << 5) + low;                                                    \
        reverse = ((dir & tCur) == 0);                                                    \
        for (int j = 0; j < NUM_GROUPS / (16 * X); j++) {                                 \
            for (int i = 0; i < 32; i++) x[i] = get(sdata, 32 * wg * j + tCur + i * inc); \
            B32V(x, 0);                                                                   \
            for (int i = 0; i < 32; i++) set(sdata, 32 * wg * j + tCur + i * inc, x[i]);  \
        }                                                                                 \
        inc >>= 1;                                                                        \
    }

#define RUN_16(X)                                                                         \
    {                                                                                     \
        inc >>= 3;                                                                        \
        low = t & (inc - 1);                                                              \
        tCur = ((t - low) << 4) + low;                                                    \
        reverse = ((dir & tCur) == 0);                                                    \
        for (int j = 0; j < NUM_GROUPS / (8 * X); j++) {                                  \
            for (int i = 0; i < 16; i++) x[i] = get(sdata, 16 * wg * j + tCur + i * inc); \
            B16V(x, 0);                                                                   \
            for (int i = 0; i < 16; i++) set(sdata, 16 * wg * j + tCur + i * inc, x[i]);  \
        }                                                                                 \
        inc >>= 1;                                                                        \
    }

#define RUN_8(X)                                                                        \
    {                                                                                   \
        inc >>= 2;                                                                      \
        low = t & (inc - 1);                                                            \
        tCur = ((t - low) << 3) + low;                                                  \
        reverse = ((dir & tCur) == 0);                                                  \
        for (int j = 0; j < NUM_GROUPS / (4 * X); j++) {                                \
            for (int i = 0; i < 8; i++) x[i] = get(sdata, 8 * wg * j + tCur + i * inc); \
            B8V(x, 0);                                                                  \
            for (int i = 0; i < 8; i++) set(sdata, 8 * wg * j + tCur + i * inc, x[i]);  \
        }                                                                               \
        inc >>= 1;                                                                      \
    }

#define RUN_4(X)                                                                        \
    {                                                                                   \
        inc >>= 1;                                                                      \
        low = t & (inc - 1);                                                            \
        tCur = ((t - low) << 2) + low;                                                  \
        reverse = ((dir & tCur) == 0);                                                  \
        for (int j = 0; j < NUM_GROUPS / (2 * X); j++) {                                \
            for (int i = 0; i < 4; i++) x[i] = get(sdata, 4 * wg * j + tCur + i * inc); \
            B4V(x, 0);                                                                  \
            for (int i = 0; i < 4; i++) set(sdata, 4 * wg * j + tCur + i * inc, x[i]);  \
        }                                                                               \
        inc >>= 1;                                                                      \
    }

#define RUN_2(X)                                                                        \
    {                                                                                   \
        low = t & (inc - 1);                                                            \
        tCur = ((t - low) << 1) + low;                                                  \
        reverse = ((dir & tCur) == 0);                                                  \
        for (int j = 0; j < NUM_GROUPS / (X); j++) {                                    \
            for (int i = 0; i < 2; i++) x[i] = get(sdata, 2 * wg * j + tCur + i * inc); \
            B2V(x, 0);                                                                  \
            for (int i = 0; i < 2; i++) set(sdata, 2 * wg * j + tCur + i * inc, x[i]);  \
        }                                                                               \
        inc >>= 1;                                                                      \
    }

// REDUCE(X)：
// 初始设定：sdata 中的数据是排序长度为 k 的升降交替的排序序列，
//      如果 block 中所有进程都执行 REDUCE(X) 的话，sdata 中的有效数据量应该为 2 * wg * NUM_GROUPS / (X)
// 运行结果：每两个相邻的k长度的排序序列，对应位置比较后保留较大值，得到长度为k的双调序列，
//      整个 sdata 变为一连串的长度为k的双调序列，总的数据量减半
#define REDUCE(X)                                                                         \
    {                                                                                     \
        tCur = ((t >> klog2) << (klog2 + 1)) + (t & (k - 1));                             \
        for (int j = 0; j < NUM_GROUPS / (X); j++) {                                      \
            x[j] = max(get(sdata, 2 * wg * j + tCur), get(sdata, 2 * wg * j + tCur + k)); \
        }                                                                                 \
        __syncthreads();                                                                  \
        for (int j = 0; j < NUM_GROUPS / (X); j++) {                                      \
            set(sdata, wg* j + t, x[j]);                                                  \
        }                                                                                 \
    }

template <typename T>
__global__ void Bitonic_TopKLocalSortInPlace(T* __restrict__ in, T* __restrict__ out,
                                             const int k, const int klog2) {
    // const int k = K;
    // const int klog2 = KLog2;

    // Shared mem size is determined by the host app at run time.
    // For n elements, we have n * 33/32 shared memory.
    // We use this to break bank conflicts.
    SharedMemory<T> smem;
    T* sdata = smem.getPointer();

    const int t = threadIdx.x;  // index in workgroup
    const int wg = blockDim.x;  // workgroup size = block size, power of 2
    const int gid = blockIdx.x;

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

    // t = threadId.x, in 是原始数据数组，而 NUM_ELEM_PT = 1 << NUM_ELEM_BITSHIFT，
    // 从而下面这两行代码将该线程需要处理的连续 NUM_ELEM_PT 个数据都放到 local memory 中的 x 数组里面了
    int tCur = t << NUM_ELEM_BITSHIFT;
    // for (int i = 0; i < NUM_ELEM_PT; i++) set(sdata, tCur + i, in[tCur + i]);
    // for (int i = 0; i < NUM_ELEM_PT; i++) x[i] = get(sdata, tCur + i);
    for (int i = 0; i < NUM_ELEM_PT; i++) x[i] = in[tCur + i];

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

    // Complete the remaining steps to create sorted sequences of length k.
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
        // if (t == 0 && gid == 0) {
        //     printf("wg=%d, k=%d, dir=%d, length=%d, mod=%d\n", wg, k, dir, length, mod);
        // }

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
            // int begin = wg/2;
            // int p_num = 8;
            // if (t == 0 && gid == 0) {
            // printf("begin: %d\n", begin * 16);
            // printf("inc: %d\n", inc);
            // printf("before\n");
            // for (int m = 0; m < wg * NUM_ELEM_PT; m++) {
            //     if (get(sdata, m) == (uint)(~0)) printf("sdata[%d]=%u\n", m, (uint)(~0));
            // }
            // for(int m = begin; m < begin+p_num; m++) {
            //     printf(" *%d* ", m);
            //     for (int n = 0; n < 16; n++) {
            //         uint now =  get(sdata, 16 * m + n);
            //         uint after =  get(sdata, 16 * m + n + 1);
            //         printf("%u %c ", now, (now > after) ? 'v': ((now == after) ? '=' : '^'));
            //     }
            //     printf("\n");
            // }
            // }
            RUN_8(1)
            __syncthreads();
            // if (t == 0 && gid == 0) {
            //     printf("after\n");
            //     for (int m = 0; m < wg * NUM_ELEM_PT; m++) {
            //         if (get(sdata, m) == (uint)(~0)) printf("sdata[%d]=%u\n", m, (uint)(~0));
            //     }
            // for(int m = begin; m < begin+p_num; m++) {
            //     printf(" *%d* ", m);
            //     for (int n = 0; n < 16; n++) {
            //         uint now =  get(sdata, 16 * m + n);
            //         uint after =  get(sdata, 16 * m + n + 1);
            //         printf("%u %c ", now, (now > after) ? 'v': ((now == after) ? '=' : '^'));
            //     }
            //     printf("\n");
            // }
            // }
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
    // End of Step 2;

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
    // End of Step 1;

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

    // Step 4: Reduce size again by 2.
    REDUCE(4)
    __syncthreads();
    // 运行结果，获得一系列长度为k的双调序列，总数据量是原始数据量的 1/8
    // End of Step 1;

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

    // 从 1/8 也可以看出，剩余的数据量 wg * NUM_ELEM_PT / 8 >= 2 * k
    // 而从调用方可知 wg = max(64, k), 所以如果要在该函数达到 1/8 的数据量缩减， NUM_ELEM_PT >= 16
    out += (NUM_ELEM_PT / 16) * gid * wg;
    tCur = ((t >> klog2) << (klog2 + 1)) + (t & (k - 1));
    for (int j = 0; j < NUM_GROUPS / 8; j++) {
        T x0 = get(sdata, 2 * wg * j + tCur);
        T x1 = get(sdata, 2 * wg * j + tCur + k);
        out[wg * j + t] = max(x0, x1);
    }
    // 最终输出，一系列长度k的双调序列，总数据量：原数据的 1/16

    // out += (NUM_ELEM_PT / 8) * gid * wg;
    // tCur = ((t >> klog2) << (klog2+1)) + (t&(k-1));
    // for (int j=0; j<NUM_GROUPS/4; j++) {
    // T x0 = get(sdata, 2*wg*j + tCur);
    // T x1 = get(sdata, 2*wg*j + tCur + k);
    // out[wg*j + t] = max(x0, x1);
    // }
}

template <typename T>
__global__ void Bitonic_TopKReduce(T* __restrict__ in, T* __restrict__ out,
                                   const int k, const int klog2) {
    // const int k = K;
    // const int klog2 = KLog2;

    // Shared mem size is determined by the host app at run time.
    // For n elements, we have n * 33/32 shared memory.
    // We use this to break bank conflicts.
    SharedMemory<T> smem;
    T* sdata = smem.getPointer();

    const int t = threadIdx.x;  // index in workgroup
    const int wg = blockDim.x;  // workgroup size = block size, power of 2
    const int gid = blockIdx.x;

    int length = min(NUM_GROUPS, k >> 1);
    int inc = length;
    inc >>= NUM_GROUPS_BITSHIFT;
    int low = t & (inc - 1);
    int dir = length << 1;
    bool reverse;

    T x[NUM_ELEM_PT];

    // Move IN, OUT to block start
    in += NUM_ELEM_PT * gid * wg;

    int tCur = t << NUM_ELEM_BITSHIFT;
    for (int i = 0; i < NUM_ELEM_PT; i++) x[i] = in[tCur + i];
    for (int i = 0; i < NUM_ELEM_PT; i++) set(sdata, tCur + i, x[i]);

    __syncthreads();

    // Complete the remaining steps to create sorted sequences of length k.
    int mod;
    unsigned int mask;

    length = (k >> 1);
    dir = length << 1;
    // Loop on comparison distance (between keys)
    inc = length;
    mod = inc;
    mask = ~(NUM_ELEM_PT / (1) - 1);
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
    if (mod & 8) {
        RUN_16(1)
        __syncthreads();
    }
    while (inc > 8) {
        RUN_32(1)
        __syncthreads();
    }
#else
    while (inc > 4) {
        RUN_16(1)
        __syncthreads();
    }
#endif  // NUM_ELEM_PT > 16
#else
    while (inc > 2) {
        RUN_8(1)
        __syncthreads();
    }
#endif  // NUM_ELEM_PT > 8

    // Step 2: Reduce the size by factor 2 by pairwise comparing adjacent sequences.
    REDUCE(1)
    __syncthreads();
    // End of Step 2;

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
    while (inc > 4) {
        if (t < (wg >> 1)) {
            RUN_16(1)
        } else {
            inc >>= 4;
        }
        __syncthreads();
    }
#else
    while (inc > 2) {
        RUN_8(2)
        __syncthreads();
    }
#endif  // NUM_ELEM_PT > 16
#else
    while (inc > 1) {
        RUN_4(2)
        __syncthreads();
    }
#endif  // NUM_ELEM_PT > 8

    // Step 4: Reduce size again by 2.
    REDUCE(2)
    __syncthreads();
    // End of Step 1;

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
    while (inc > 2) {
        if (t < (wg >> 1)) {
            RUN_8(2)
        } else {
            inc >>= 3;
        }
        __syncthreads();
    }
#else
    while (inc > 1) {
        RUN_4(4)
        __syncthreads();
    }
#endif  // NUM_ELEM_PT > 16
#else
    while (inc > 0) {
        RUN_2(4)
        __syncthreads();
    }
#endif  // NUM_ELEM_PT > 8 while (inc > 0)

    // Step 4: Reduce size again by 2.
    REDUCE(4)
    __syncthreads();
    // End of Step 1;

    length = k >> 1;
    dir = length << 1;
    // Loop on comparison distance (between keys)
    inc = length;
    mod = inc;
    mask = ~(NUM_ELEM_PT / (4) - 1);
    while ((mod & mask) != 0) mod >>= (NUM_ELEM_BITSHIFT - 2);

    if (mod & 1) {
        RUN_2(8)
        __syncthreads();
    }
    while (inc > 0) {
        if (t < (wg >> 1)) {
            RUN_4(4)
        } else {
            inc >>= 2;
        }
        __syncthreads();
    }

    out += (NUM_ELEM_PT / 16) * gid * wg;
    tCur = ((t >> klog2) << (klog2 + 1)) + (t & (k - 1));
    for (int j = 0; j < NUM_GROUPS / 8; j++) {
        T x0 = get(sdata, 2 * wg * j + tCur);
        T x1 = get(sdata, 2 * wg * j + tCur + k);
        out[wg * j + t] = max(x0, x1);
    }
}

const int tab32[32] = {
    0, 9, 1, 10, 13, 21, 2, 29,
    11, 14, 16, 18, 22, 25, 3, 30,
    8, 12, 20, 28, 15, 17, 24, 7,
    19, 27, 23, 6, 26, 5, 4, 31};

int log2_32(uint value) {
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    return tab32[(uint)(value * 0x07C4ACDD) >> 27];
}

template <typename KeyT>
cudaError_t bitonicTopK(KeyT* d_keys_in, unsigned int num_items, unsigned int k, KeyT* d_keys_out,
                        CachingDeviceAllocator& g_allocator) {
    // <ERROR>: compareTopKAlogorithms.cu 调用时，d_keys_out 只分配了 k * sizeof(KeyT) 大小的空间，
    //          所以，如果在这修改了 k，如果传入的 k < 16，将导致函数最后复制进 d_keys_out 的数据超过分配的空间
    // if (k < 16) k = 16;

    int klog2 = log2_32(k);

    DoubleBuffer<KeyT> d_keys;
    d_keys.d_buffers[0] = d_keys_in;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[1], sizeof(KeyT) * num_items));

    int current = 0;
    int numThreads = num_items;

    int wg_size = max(64, k);

    numThreads >>= 1;  // Each thread processes 2 elements.
    numThreads >>= NUM_GROUPS_BITSHIFT;
    // 如果 k > 64，而 num_items 小于 NUM_ELEM_PT * k 的话，numThreads / wg_size 会等于 0，如下 kernel 无法执行
    // 如果 k <= 64, 而 num_items 小于 NUM_ELEM_PT * 64 的话，numThreads / wg_size 也会等于 0
    // k 能取的最大值受限于一个 block 能承载的最大线程数，
    // 并且 k * NUM_ELEM_PT * 33 / 32 * sizeof(KeyT) 也应该小于一个 block 最多能分配的 share memory 的大小
    Bitonic_TopKLocalSortInPlace<KeyT><<<numThreads / wg_size, wg_size, ((2 * NUM_GROUPS * wg_size * 33) / 32) * sizeof(KeyT)>>>(d_keys.Current(), d_keys.Alternate(), k, klog2);
    current = 1 - current;

    // Toggle the buffer index in the double buffer
    d_keys.selector = d_keys.selector ^ 1;

    numThreads >>= (1 + NUM_GROUPS_BITSHIFT);

    while (numThreads >= wg_size) {
        Bitonic_TopKReduce<KeyT><<<numThreads / wg_size, wg_size, ((2 * NUM_GROUPS * wg_size * 33) / 32) * sizeof(KeyT)>>>(d_keys.Current(), d_keys.Alternate(), k, klog2);
        // Toggle the buffer index in the double buffer
        d_keys.selector = d_keys.selector ^ 1;

        numThreads >>= (1 + NUM_GROUPS_BITSHIFT);
    }

    KeyT* res_vec = (KeyT*)malloc(sizeof(KeyT) * 2 * numThreads * NUM_GROUPS);
    cudaMemcpy(res_vec, d_keys.Current(), 2 * numThreads * NUM_GROUPS * sizeof(KeyT), cudaMemcpyDeviceToHost);
    std::sort(res_vec, res_vec + 2 * numThreads * NUM_GROUPS, std::greater<KeyT>());
    cudaMemcpy(d_keys_out, res_vec, k * sizeof(KeyT), cudaMemcpyHostToDevice);

    if (d_keys.d_buffers[1])
        CubDebugExit(g_allocator.DeviceFree(d_keys.d_buffers[1]));

    return cudaSuccess;
}
