#pragma once

#include <cuda.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_allocator.cuh>

using namespace cub;
using namespace std;

/**
 * Computes the histogram over the digit values of an array of keys that MUST have a length of an integer multiple of (KPT * blockDim.x).
 * The padding to the integer multiple can be done by adding 0's at the end and subtracting the number of padded 0's from the final result's 0 bin.
 * The 2^NUM_BITS possible counts (0..2^NUM_BITSNUM_BITS-1) will be placed in global_histo.
 * @param keys            [IN]  The keys for which to compute the histogram
 * @param digit           [IN]
 * @param global_histo        [OUT] The array of element counts, MUST be 256 in size.
 * @param per_block_histo     [OUT]
 */
template <
    typename KeyT,            // Data type of the keys within device memory. Data will be twiddled (if necessary) to unsigned type
    typename IndexT,          // Data type used for key's offsets and counters (limits number of supported keys, uint = 2^32)
    int NUM_BITS,             // Number of bits being sorted at a time
    int KPT,                  // Number of keys per thread
    int TPB,                  // Number of threads per block
    int PRE_SORT_RUNS_LENGTH  // For values greater than 1, this causes to sort a thread's keys by runs of a given length to improve run-length encoded updates to shared memory.
    >
__global__ void rdxsrt_histogram(KeyT* __restrict__ keys, const uint digit, IndexT* global_histo) {
    /*** TYPEDEFs ***/
    typedef Traits<KeyT> KeyTraits;
    typedef typename KeyTraits::UnsignedBits UnsignedBits;

    /*** DECLARATIONS ***/
    UnsignedBits tloc_keys[KPT];
    uint tloc_masked[KPT];
    // 0x01 << NUM_BITS := 2 ^ NUM_BITS = 2^8 = 256
    __shared__ uint shared_bins[0x01 << NUM_BITS];

    /*** INIT SHARED HISTO ***/
    // 将 shared_bins 初始化为全0
    if (threadIdx.x < 32) {
#pragma unroll
        for (int i = 0; i < (0x01 << NUM_BITS); i += 32) {
            shared_bins[i + threadIdx.x] = 0;
        }
    }
    __syncthreads();

    /*** GET KEYS & PREPARE KEYS FOR HISTO ***/
    // This thread block's keys memory offset, pointing to the index of its first key
    const IndexT block_offset = (blockDim.x * blockIdx.x * KPT);

// Load keys
#pragma unroll
    for (int i = 0; i < KPT; i++) {
        // 对于第1个线程来说 tloc_keys 为 keys 的第 0，TPB, 2*TPB，3*TPB，……，15*TPB 个元素
        // 对于第2个线程来说 tloc_keys 为 keys 的第 1，TPB+1, 2*TPB+1，3*TPB+1，……，15*TPB+1 个元素
        // 对于第384个线程来说 tloc_keys 为 keys 的第 383，TPB+383, 2*TPB+383，3*TPB+383，……，15*TPB+383 个元素
        // 对于第385个线程来说 tloc_keys 为 keys 的第 16*TPB，TPB+16*TPB, 2*TPB+16*TPB，3*TPB+16*TPB，……，15*TPB+16*TPB 个元素
        tloc_keys[i] = reinterpret_cast<UnsignedBits*>(keys)[block_offset + threadIdx.x + blockDim.x * i];
    }

    // Mask
    uint digit_num = (sizeof(KeyT) * 8 + NUM_BITS - 1) / NUM_BITS;
    if (digit == digit_num - 1) {  // last digit
#pragma unroll
        for (int i = 0; i < KPT; i++) {
            tloc_keys[i] = KeyTraits::TwiddleIn(tloc_keys[i]);
            tloc_masked[i] = tloc_keys[i] & ((0x01 << (sizeof(KeyT) * 8 - digit * NUM_BITS)) - 1);
        }
    } else {
#pragma unroll
        for (int i = 0; i < KPT; i++) {
            tloc_keys[i] = KeyTraits::TwiddleIn(tloc_keys[i]);
            tloc_masked[i] = (tloc_keys[i] >> ((sizeof(KeyT) * 8) - (NUM_BITS * (digit + 1)))) & ((0x01 << NUM_BITS) - 1);
        }
    }

    /*** COMPUTE HISTO ***/
    uint rle = 1;
#pragma unroll
    // 在shared_bins中统计tloc_masked的分布
    for (int i = 1; i < KPT; i++) {
        if (tloc_masked[i] == tloc_masked[i - 1])
            rle++;
        else {
            atomicAdd(&shared_bins[tloc_masked[i - 1]], rle);
            rle = 1;
        }
    }
    atomicAdd(&shared_bins[tloc_masked[KPT - 1]], rle);

    // Make sure we've got the counts from all threads
    __syncthreads();

    /*** Write shared histo to global histo ***/
    if (threadIdx.x < 32) {
        for (int i = 0; i < (0x01 << NUM_BITS); i += 32) {
            atomicAdd(&global_histo[i + threadIdx.x], shared_bins[i + threadIdx.x]);
        }
    }
}

template <
    typename KeyT,            // Data type of the keys within device memory. Data will be twiddled (if necessary) to unsigned type
    typename IndexT,          // Data type used for key's offsets and counters (limits number of supported keys, uint = 2^32)
    int NUM_BITS,             // Number of bits being sorted at a time
    int KPT,                  // Number of keys per thread
    int TPB,                  // Number of threads per block
    int PRE_SORT_RUNS_LENGTH  // For values greater than 1, this causes to sort a thread's keys by runs of a given length to improve run-length encoded updates to shared memory.
    >
__global__ void rdxsrt_histogram_with_guards(KeyT* __restrict__ keys, const uint digit, IndexT* global_histo, const IndexT total_keys, const int block_index_offset) {
    /*** TYPEDEFs ***/
    typedef Traits<KeyT> KeyTraits;
    typedef typename KeyTraits::UnsignedBits UnsignedBits;
    /* typedef LoadUnit<IndexT, RDXSRT_LOAD_STRIDE_WARP, KPT, TPB> KeyLoader; */

    /*** DECLARATIONS ***/
    UnsignedBits tloc_keys[KPT];
    uint tloc_masked[KPT];
    __shared__ uint shared_bins[(0x01 << NUM_BITS) + 1];

    /*** INIT SHARED HISTO ***/
    if (threadIdx.x < 32) {
#pragma unroll
        for (int i = 0; i < (0x01 << NUM_BITS); i += 32) {
            shared_bins[i + threadIdx.x] = 0;
        }
    }
    __syncthreads();

    /*** GET KEYS & PREPARE KEYS FOR HISTO ***/
    // This thread block's keys memory offset, pointing to the index of its first key
    const IndexT block_offset = (blockDim.x * (block_index_offset + blockIdx.x) * KPT);

    // Maximum number of keys the block may fetch
    const IndexT block_max_num_keys = total_keys - block_offset;
    // KeyLoader(block_offset, threadIdx.x).template LoadStridedWithGuards<UnsignedBits, KeyT, 0, KPT>(keys, tloc_keys, block_max_num_keys);
#pragma unroll
    for (int i = 0; i < KPT; i++) {
        if ((threadIdx.x + blockDim.x * i) < block_max_num_keys) {
            tloc_keys[i] = reinterpret_cast<UnsignedBits*>(keys)[block_offset + threadIdx.x + blockDim.x * i];
        }
    }

    uint digit_num = (sizeof(KeyT) * 8 + NUM_BITS - 1) / NUM_BITS;
    if (digit == digit_num - 1) {  // last digit
#pragma unroll
        for (int i = 0; i < KPT; i++) {
            if ((threadIdx.x + blockDim.x * i) < block_max_num_keys) {
                tloc_keys[i] = KeyTraits::TwiddleIn(tloc_keys[i]);
                tloc_masked[i] = tloc_keys[i] & ((0x01 << (sizeof(KeyT) * 8 - digit * NUM_BITS)) - 1);
                atomicAdd(&shared_bins[tloc_masked[i]], 1);
            }
        }
    } else {
#pragma unroll
        for (int i = 0; i < KPT; i++) {
            if ((threadIdx.x + blockDim.x * i) < block_max_num_keys) {
                tloc_keys[i] = KeyTraits::TwiddleIn(tloc_keys[i]);
                tloc_masked[i] = (tloc_keys[i] >> ((sizeof(KeyT) * 8) - (NUM_BITS * (digit + 1)))) & ((0x01 << NUM_BITS) - 1);
                atomicAdd(&shared_bins[tloc_masked[i]], 1);
            }
        }
    }

    // Make sure we've got the counts from all threads
    __syncthreads();

    /*** Write shared histo to global histo ***/
    if (threadIdx.x < 32) {
        for (int i = 0; i < (0x01 << NUM_BITS); i += 32) {
            atomicAdd(&global_histo[i + threadIdx.x], shared_bins[i + threadIdx.x]);
        }
    }
}

/**
 * Makes a single pass over the input array to find entries whose digit is equal to selected digit value and greater than
 * digit value. Entries equal to digit value are written to keys_buffer for future processing, entries greater
 * are written to output array.
 * @param d_keys        [IN] The keys for which to compute the histogram
 *                      [OUT] Entries with x[digit] = digit_val.
 * @param digit            [IN] Digit index (0 => highest digit, 3 => lowest digit for 32-bit)
 * @param digit_val        [IN] Digit value.
 * @param num_items        [IN] Number of entries.
 * @param d_keys_out       [OUT] Entries with x[digit] > digit_val.
 * @param d_index_buffer   [OUT] Index into d_keys[OUT].
 * @param d_index_out      [OUT] Index into d_keys_out.
 */
template <
    typename KeyT,    // Data type of the keys within device memory. Data will be twiddled (if necessary) to unsigned type
    typename IndexT,  // Data type used for key's offsets and counters (limits number of supported keys, uint = 2^32)
    int NUM_BITS,     // Number of bits being sorted at a time
    int KPT,          // Number of keys per thread
    int TPB           // Number of threads per block
    >
__global__ void select_kth_bucket(KeyT* d_keys, const uint digit, const uint digit_val, uint num_items,
                                  KeyT* d_keys_out, uint* d_index_buffer, uint* d_index_out) {
    typedef Traits<KeyT> KeyTraits;
    typedef typename KeyTraits::UnsignedBits UnsignedBits;

    // Specialize BlockLoad for a 1D block of TPB threads owning KPT integer items each
    typedef cub::BlockLoad<UnsignedBits, TPB, KPT, BLOCK_LOAD_TRANSPOSE> BlockLoadT;

    // Specialize BlockScan type for our thread block
    typedef BlockScan<int, TPB, BLOCK_SCAN_RAKING> BlockScanT;

    const int tile_size = TPB * KPT;
    int tile_idx = blockIdx.x;  // Current tile index
    int tile_offset = tile_idx * tile_size;

    // Allocate shared memory for BlockLoad
    __shared__ union TempStorage {
        typename BlockLoadT::TempStorage load_items;
        typename BlockScanT::TempStorage scan;
        int offset[1];
        UnsignedBits raw_exchange[TPB * KPT];
    } temp_storage;

    // Load a segment of consecutive items that are blocked across threads
    UnsignedBits key_entries[KPT];
    /* float payload_entries[KPT]; */
    int selection_flags[KPT];
    int selection_indices[KPT];

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
        BlockLoadT(temp_storage.load_items).Load(reinterpret_cast<UnsignedBits*>(d_keys) + tile_offset, key_entries, num_tile_items);
    else
        BlockLoadT(temp_storage.load_items).Load(reinterpret_cast<UnsignedBits*>(d_keys) + tile_offset, key_entries);

#if 0
  if (is_last_tile)
    BlockLoadT(temp_storage.load_items).Load(payload + tile_offset, payload_entries, num_tile_items);
  else
    BlockLoadT(temp_storage.load_items).Load(payload + tile_offset, payload_entries);
#endif

    __syncthreads();

    /*** Step 1: Find keys with digit value to selected digit value ***/
    uint digit_num = (sizeof(KeyT) * 8 + NUM_BITS - 1) / NUM_BITS;
    if (digit == digit_num - 1) {  // last digit
#pragma unroll
        for (int ITEM = 0; ITEM < KPT; ++ITEM) {
            // Out-of-bounds items are selection_flags
            selection_flags[ITEM] = 0;
            if (!is_last_tile || (int(threadIdx.x * KPT) + ITEM < num_tile_items)) {
                UnsignedBits key = KeyTraits::TwiddleIn(key_entries[ITEM]);
                uint masked_key = key & ((0x01 << (sizeof(KeyT) * 8 - digit * NUM_BITS)) - 1);
                selection_flags[ITEM] = (masked_key > digit_val);
            }
        }

    } else {
#pragma unroll
        for (int ITEM = 0; ITEM < KPT; ++ITEM) {
            // Out-of-bounds items are selection_flags
            selection_flags[ITEM] = 0;
            if (!is_last_tile || (int(threadIdx.x * KPT) + ITEM < num_tile_items)) {
                UnsignedBits key = KeyTraits::TwiddleIn(key_entries[ITEM]);
                uint masked_key = (key >> ((sizeof(KeyT) * 8) - (NUM_BITS * (digit + 1)))) & ((0x01 << NUM_BITS) - 1);
                selection_flags[ITEM] = (masked_key > digit_val);
            }
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
            // 由于后面的 __syncthreads()，所以所有 block 只有 0 号线程能有效工作，而且由于 d_index_out 只能互斥访问，所以这些线程0只能顺序工作
            // index_out 是 d_index_out 的 old value
            index_out = atomicAdd(d_index_out, num_selected);
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
        for (int ITEM = 0; ITEM < KPT; ++ITEM) {
            int local_scatter_offset = selection_indices[ITEM];
            if (selection_flags[ITEM]) {
                temp_storage.raw_exchange[local_scatter_offset] = key_entries[ITEM];
                /* temp_storage.raw_exchange[tile_size + local_scatter_offset] = payload_entries[ITEM]; */
            }
        }

        __syncthreads();

        // Write out matched entries to output array
        for (int item = threadIdx.x; item < num_selected; item += TPB) {
            reinterpret_cast<UnsignedBits*>(d_keys_out)[index_out + item] = temp_storage.raw_exchange[item];
        }

        __syncthreads();

#if 0
        for (int item = threadIdx.x; item < num_selected; item += TPB) {
            payload_out[num_selections_prefix + item] = temp_storage.raw_exchange[tile_size + item];
        }
#endif
    }

    /*** Step 2: Find entries that have digit equal to digit value ***/
    if (digit == digit_num - 1) {  // last digit
#pragma unroll
        for (int ITEM = 0; ITEM < KPT; ++ITEM) {
            // Out-of-bounds items are selection_flags
            selection_flags[ITEM] = 0;
            if (!is_last_tile || (int(threadIdx.x * KPT) + ITEM < num_tile_items)) {
                UnsignedBits key = KeyTraits::TwiddleIn(key_entries[ITEM]);
                uint masked_key = key & ((0x01 << (sizeof(KeyT) * 8 - digit * NUM_BITS)) - 1);
                selection_flags[ITEM] = (masked_key == digit_val);
            }
        }
    } else {
#pragma unroll
        for (int ITEM = 0; ITEM < KPT; ++ITEM) {
            // Out-of-bounds items are selection_flags
            selection_flags[ITEM] = 0;

            if (!is_last_tile || (int(threadIdx.x * KPT) + ITEM < num_tile_items)) {
                UnsignedBits key = KeyTraits::TwiddleIn(key_entries[ITEM]);
                uint masked_key = (key >> ((sizeof(KeyT) * 8) - (NUM_BITS * (digit + 1)))) & ((0x01 << NUM_BITS) - 1);
                selection_flags[ITEM] = (masked_key == digit_val);
            }
        }
    }

    __syncthreads();

    // Compute exclusive prefix sum
    BlockScanT(temp_storage.scan).ExclusiveSum(selection_flags, selection_indices, num_selected);

    __syncthreads();

    if (num_selected > 0) {
        int index_buffer;
        if (threadIdx.x == 0) {
            index_buffer = atomicAdd(d_index_buffer, num_selected);
            temp_storage.offset[0] = index_buffer;
        }

        __syncthreads();

        index_buffer = temp_storage.offset[0];

        __syncthreads();

// Compact and scatter items
#pragma unroll
        for (int ITEM = 0; ITEM < KPT; ++ITEM) {
            int local_scatter_offset = selection_indices[ITEM];
            if (selection_flags[ITEM]) {
                temp_storage.raw_exchange[local_scatter_offset] = key_entries[ITEM];
                /* temp_storage.raw_exchange[tile_size + local_scatter_offset] = payload_entries[ITEM]; */
            }
        }

        __syncthreads();

        // Write out output entries
        for (int item = threadIdx.x; item < num_selected; item += TPB) {
            reinterpret_cast<UnsignedBits*>(d_keys)[index_buffer + item] = temp_storage.raw_exchange[item];
        }

        __syncthreads();
    }
}

// <ERROR>: 只能支持sizeof(KeyT)为4的数据类型
#define KPT 16
#define TPB 320
#define DIGIT_BITS_FOR_4BYTES 10
#define DIGIT_BITS_FOR_8BYTES 13
template <typename KeyT>
cudaError_t radixSelectTopK(KeyT* d_keys_in, uint num_items, uint k, KeyT* d_keys_out, unsigned int& out_items,
                            CachingDeviceAllocator& g_allocator) {
    cudaError error = cudaSuccess;
    out_items = k;

    int digit_bits = (sizeof(KeyT) == 8) ? DIGIT_BITS_FOR_8BYTES : DIGIT_BITS_FOR_4BYTES;
    uint histogram_size = 0x01 << digit_bits;

    uint* d_histogram;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_histogram, sizeof(uint) * histogram_size));

    // We allocate two indices, one that maintains index into output array (this goes till K)
    // second maintains index into the output buffer containing reduced set of top-k candidates.
    uint* d_index_out;
    uint* d_index_buffer;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_index_out, sizeof(uint)));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_index_buffer, sizeof(uint)));

    // Set the index into output array to 0.
    cudaMemset(d_index_out, 0, sizeof(uint));

    uint* h_histogram = new uint[histogram_size];

    uint KPB = KPT * TPB;

    uint digit_num = (sizeof(KeyT) * 8 + digit_bits - 1) / digit_bits;
    for (uint digit = 0; digit < digit_num; digit++) {
        uint num_blocks = num_items / KPB;                             // Pass-0 rough processing blocks (floor on purpose)
        uint processed_elements = num_blocks * KPB;                    // Pass-0 number of rough processed elements
        uint remaining_elements = num_items - processed_elements;      // Do the remaining elements with a check in the inner loop
        uint remainder_blocks = (KPB - 1 + remaining_elements) / KPB;  // Number of blocks required for remaining elements (typically 0 or 1)

        // Zero out the histogram
        cudaMemset(d_histogram, 0, histogram_size * sizeof(uint));

        if (num_blocks > 0) {
            if (sizeof(KeyT) == 8)
                rdxsrt_histogram<KeyT, uint, DIGIT_BITS_FOR_8BYTES, KPT, TPB, 9><<<num_blocks, TPB, 0>>>(d_keys_in, digit, d_histogram);
            else
                rdxsrt_histogram<KeyT, uint, DIGIT_BITS_FOR_4BYTES, KPT, TPB, 9><<<num_blocks, TPB, 0>>>(d_keys_in, digit, d_histogram);
        }

        if (remaining_elements > 0) {
            if (sizeof(KeyT) == 8)
                rdxsrt_histogram_with_guards<KeyT, uint, DIGIT_BITS_FOR_8BYTES, KPT, TPB, 9><<<remainder_blocks, TPB, 0>>>(d_keys_in, digit, d_histogram, num_items, num_blocks);
            else
                rdxsrt_histogram_with_guards<KeyT, uint, DIGIT_BITS_FOR_4BYTES, KPT, TPB, 9><<<remainder_blocks, TPB, 0>>>(d_keys_in, digit, d_histogram, num_items, num_blocks);
        }

        cudaMemcpy(h_histogram, d_histogram, histogram_size * sizeof(uint), cudaMemcpyDeviceToHost);

        // Check for failure to launch
        CubDebugExit(error = cudaPeekAtLastError());

        uint rolling_sum = 0;
        uint digit_val;
        for (int i = histogram_size - 1; i >= 0; i--) {
            if ((rolling_sum + h_histogram[i]) > k) {
                // 表示digit数字值为 (digit_val+1) …… 255 的子桶中的元素都是 top-k 元素
                // 并且第 k 个元素在digit数字值为 digit_val （或 digit_val+1）的子桶中
                // 如果 rolling_sum == k，那么第 k 个元素在 digit_val+1 的子桶中
                digit_val = i;
                // k 变为该子桶中的对应序号
                k -= rolling_sum;
                break;
            }
            rolling_sum += h_histogram[i];
        }

        cudaMemset(d_index_buffer, 0, sizeof(uint));

        // 运行结束后，
        // d_keys_in 指向的空间存放了 digit 数字值为 digit_val 的子桶中的所有元素
        // d_keys_out 是一定属于 top-k 的子桶的集合
        // d_index_buffer 为 d_keys_in 指向空间的有效元素个数
        // d_index_out 为 d_keys_out 中的元素个数，注意：每次循环，并未置零 d_index_out，所以会一直累加
        if (sizeof(KeyT) == 8)
            select_kth_bucket<KeyT, uint, DIGIT_BITS_FOR_8BYTES, KPT, TPB><<<num_blocks + remainder_blocks, TPB>>>(d_keys_in, digit, digit_val, num_items, d_keys_out, d_index_buffer, d_index_out);
        else
            select_kth_bucket<KeyT, uint, DIGIT_BITS_FOR_4BYTES, KPT, TPB><<<num_blocks + remainder_blocks, TPB>>>(d_keys_in, digit, digit_val, num_items, d_keys_out, d_index_buffer, d_index_out);

        CubDebugExit(error = cudaPeekAtLastError());

        uint h_index_out;
        uint h_index_buffer;

        cudaMemcpy(&h_index_out, d_index_out, sizeof(uint), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_index_buffer, d_index_buffer, sizeof(uint), cudaMemcpyDeviceToHost);

        // Update number of items to reflect reduced number of elements.
        num_items = h_index_buffer;

        if (k == 0)
            // 意即本次遍历后 d_keys_out 已经有了所有的 top-k 元素
            break;
        else if (k != 0 && digit == digit_num - 1) {
            // We are at last digit and k != 0 implies that kth value has repetition.
            // Copy any of the repeated values to out array to complete the array.
            cudaMemcpy(d_keys_out + h_index_out, d_keys_in, k * sizeof(KeyT), cudaMemcpyDeviceToDevice);
            k -= k;
        }
    }

    // Cleanup
    if (d_histogram)
        CubDebugExit(g_allocator.DeviceFree(d_histogram));
    if (d_index_buffer)
        CubDebugExit(g_allocator.DeviceFree(d_index_buffer));
    if (d_index_out)
        CubDebugExit(g_allocator.DeviceFree(d_index_out));

    return error;
}
