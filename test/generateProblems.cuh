#pragma once

#include <cstdlib>
#include <typeinfo>
#include <cuda.h>
#include <curand.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_allocator.cuh>
// #include <algorithm>
using namespace cub;
using namespace std;

enum DataType { UINT,
                ULONG,
                INT,
                LONG,
                FLOAT,
                DOUBLE };
enum Distribution { UNIFORM,
                    POISSON,
                    NORMAL,
                    LOG_NORMAL };
enum SortType { NO,
                INC,
                DEC };

template <typename KeyT>
void sort_keys(KeyT* d_vec, const unsigned int num_keys, CachingDeviceAllocator& g_allocator, bool is_inc) {
    // Allocate device memory for input/output
    DoubleBuffer<KeyT> d_keys;
    d_keys.d_buffers[0] = d_vec;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[1], sizeof(KeyT) * num_keys));

    // Allocate temporary storage
    size_t temp_storage_bytes = 0;
    void* d_temp_storage = NULL;
    if (is_inc) {
        CubDebugExit(DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_keys));
    } else {
        CubDebugExit(DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes, d_keys, num_keys));
    }

    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Sort
    if (is_inc) {
        CubDebugExit(DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_keys));
    } else {
        CubDebugExit(DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes, d_keys, num_keys));
    }

    // Copy results for verification. GPU-side part is done.
    if (d_keys.Current() != d_vec) {
        CubDebugExit(cudaMemcpy(d_vec, d_keys.Current(), sizeof(KeyT) * num_keys, cudaMemcpyDeviceToDevice));
    }

    if (d_temp_storage)
        CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

    if (d_keys.d_buffers[1])
        CubDebugExit(g_allocator.DeviceFree(d_keys.d_buffers[1]));
}

///////////////////////////////////////////////////////////////////
////           FUNCTIONS TO GENERATE UINTS
///////////////////////////////////////////////////////////////////
// 均匀分布
void generateUniformUints(uint* d_vec, uint num_keys, curandGenerator_t generator,
                          CachingDeviceAllocator& g_allocator, double* parameters) {
    curandGenerate(generator, d_vec, num_keys);
}
void generateSortedIncUniformUints(uint* d_vec, uint num_keys, curandGenerator_t generator,
                                   CachingDeviceAllocator& g_allocator, double* parameters) {
    curandGenerate(generator, d_vec, num_keys);
    sort_keys(d_vec, num_keys, g_allocator, true);
}
void generateSortedDecUniformUints(uint* d_vec, uint num_keys, curandGenerator_t generator,
                                   CachingDeviceAllocator& g_allocator, double* parameters) {
    curandGenerate(generator, d_vec, num_keys);
    sort_keys(d_vec, num_keys, g_allocator, false);
}
// 泊松分布
void generatePoissonUints(uint* d_vec, uint num_keys, curandGenerator_t generator,
                          CachingDeviceAllocator& g_allocator, double* parameters) {
    curandGeneratePoisson(generator, d_vec, num_keys, parameters[0]);
}
void generateSortedIncPoissonUints(uint* d_vec, uint num_keys, curandGenerator_t generator,
                                   CachingDeviceAllocator& g_allocator, double* parameters) {
    curandGeneratePoisson(generator, d_vec, num_keys, parameters[0]);
    sort_keys(d_vec, num_keys, g_allocator, true);
}
void generateSortedDecPoissonUints(uint* d_vec, uint num_keys, curandGenerator_t generator,
                                   CachingDeviceAllocator& g_allocator, double* parameters) {
    curandGeneratePoisson(generator, d_vec, num_keys, parameters[0]);
    sort_keys(d_vec, num_keys, g_allocator, false);
}

///////////////////////////////////////////////////////////////////
////           FUNCTIONS TO GENERATE UNSIGNEDLONGLONGS
///////////////////////////////////////////////////////////////////
// 均匀分布
void generateUniformUlongs(unsigned long long* d_vec, uint num_keys, curandGenerator_t generator,
                           CachingDeviceAllocator& g_allocator, double* parameters) {
    curandGenerateLongLong(generator, d_vec, num_keys);
}
void generateSortedIncUniformUlongs(unsigned long long* d_vec, uint num_keys, curandGenerator_t generator,
                                    CachingDeviceAllocator& g_allocator, double* parameters) {
    curandGenerateLongLong(generator, d_vec, num_keys);
    sort_keys(d_vec, num_keys, g_allocator, true);
}
void generateSortedDecUniformUlongs(unsigned long long* d_vec, uint num_keys, curandGenerator_t generator,
                                    CachingDeviceAllocator& g_allocator, double* parameters) {
    curandGenerateLongLong(generator, d_vec, num_keys);
    sort_keys(d_vec, num_keys, g_allocator, false);
}

///////////////////////////////////////////////////////////////////
////           FUNCTIONS TO GENERATE INTS
///////////////////////////////////////////////////////////////////
// 均匀分布
void generateUniformInts(int* d_vec, uint num_keys, curandGenerator_t generator,
                         CachingDeviceAllocator& g_allocator, double* parameters) {
    curandGenerate(generator, (uint*)d_vec, num_keys);
}
void generateSortedIncUniformInts(int* d_vec, uint num_keys, curandGenerator_t generator,
                                  CachingDeviceAllocator& g_allocator, double* parameters) {
    curandGenerate(generator, (uint*)d_vec, num_keys);
    sort_keys(d_vec, num_keys, g_allocator, true);
}
void generateSortedDecUniformInts(int* d_vec, uint num_keys, curandGenerator_t generator,
                                  CachingDeviceAllocator& g_allocator, double* parameters) {
    curandGenerate(generator, (uint*)d_vec, num_keys);
    sort_keys(d_vec, num_keys, g_allocator, false);
}

///////////////////////////////////////////////////////////////////
////           FUNCTIONS TO GENERATE LONGLONGS
///////////////////////////////////////////////////////////////////
// 均匀分布
void generateUniformLongs(long long* d_vec, uint num_keys, curandGenerator_t generator,
                          CachingDeviceAllocator& g_allocator, double* parameters) {
    curandGenerateLongLong(generator, (unsigned long long*)d_vec, num_keys);
}
void generateSortedIncUniformLongs(long long* d_vec, uint num_keys, curandGenerator_t generator,
                                   CachingDeviceAllocator& g_allocator, double* parameters) {
    curandGenerateLongLong(generator, (unsigned long long*)d_vec, num_keys);
    sort_keys(d_vec, num_keys, g_allocator, true);
}
void generateSortedDecUniformLongs(long long* d_vec, uint num_keys, curandGenerator_t generator,
                                   CachingDeviceAllocator& g_allocator, double* parameters) {
    curandGenerateLongLong(generator, (unsigned long long*)d_vec, num_keys);
    sort_keys(d_vec, num_keys, g_allocator, false);
}

///////////////////////////////////////////////////////////////////
////           FUNCTIONS TO GENERATE FLOATS
///////////////////////////////////////////////////////////////////
// U(0, 1) 均匀分布
void generateUniformFloats(float* d_vec, uint num_keys, curandGenerator_t generator,
                           CachingDeviceAllocator& g_allocator, double* parameters) {
    curandGenerateUniform(generator, d_vec, num_keys);
}
void generateSortedIncUniformFloats(float* d_vec, uint num_keys, curandGenerator_t generator,
                                    CachingDeviceAllocator& g_allocator, double* parameters) {
    curandGenerateUniform(generator, d_vec, num_keys);
    sort_keys(d_vec, num_keys, g_allocator, true);
}
void generateSortedDecUniformFloats(float* d_vec, uint num_keys, curandGenerator_t generator,
                                    CachingDeviceAllocator& g_allocator, double* parameters) {
    curandGenerateUniform(generator, d_vec, num_keys);
    sort_keys(d_vec, num_keys, g_allocator, false);
}
// 正态分布
void generateNormalFloats(float* d_vec, uint num_keys, curandGenerator_t generator,
                          CachingDeviceAllocator& g_allocator, double* parameters) {
    curandGenerateNormal(generator, d_vec, num_keys, parameters[0], parameters[1]);
}
void generateSortedIncNormalFloats(float* d_vec, uint num_keys, curandGenerator_t generator,
                                   CachingDeviceAllocator& g_allocator, double* parameters) {
    curandGenerateNormal(generator, d_vec, num_keys, parameters[0], parameters[1]);
    sort_keys(d_vec, num_keys, g_allocator, true);
}
void generateSortedDecNormalFloats(float* d_vec, uint num_keys, curandGenerator_t generator,
                                   CachingDeviceAllocator& g_allocator, double* parameters) {
    curandGenerateNormal(generator, d_vec, num_keys, parameters[0], parameters[1]);
    sort_keys(d_vec, num_keys, g_allocator, false);
}
// 对数正态分布
void generateLogNormalFloats(float* d_vec, uint num_keys, curandGenerator_t generator,
                             CachingDeviceAllocator& g_allocator, double* parameters) {
    curandGenerateLogNormal(generator, d_vec, num_keys, parameters[0], parameters[1]);
}
void generateSortedIncLogNormalFloats(float* d_vec, uint num_keys, curandGenerator_t generator,
                                      CachingDeviceAllocator& g_allocator, double* parameters) {
    curandGenerateLogNormal(generator, d_vec, num_keys, parameters[0], parameters[1]);
    sort_keys(d_vec, num_keys, g_allocator, true);
}
void generateSortedDecLogNormalFloats(float* d_vec, uint num_keys, curandGenerator_t generator,
                                      CachingDeviceAllocator& g_allocator, double* parameters) {
    curandGenerateLogNormal(generator, d_vec, num_keys, parameters[0], parameters[1]);
    sort_keys(d_vec, num_keys, g_allocator, false);
}

///////////////////////////////////////////////////////////////////
////           FUNCTIONS TO GENERATE DOUBLES
///////////////////////////////////////////////////////////////////
// U(0, 1) 均匀分布
void generateUniformDoubles(double* d_vec, uint num_keys, curandGenerator_t generator,
                            CachingDeviceAllocator& g_allocator, double* parameters) {
    curandGenerateUniformDouble(generator, d_vec, num_keys);
}
void generateSortedIncUniformDoubles(double* d_vec, uint num_keys, curandGenerator_t generator,
                                     CachingDeviceAllocator& g_allocator, double* parameters) {
    curandGenerateUniformDouble(generator, d_vec, num_keys);
    sort_keys(d_vec, num_keys, g_allocator, true);
}
void generateSortedDecUniformDoubles(double* d_vec, uint num_keys, curandGenerator_t generator,
                                     CachingDeviceAllocator& g_allocator, double* parameters) {
    curandGenerateUniformDouble(generator, d_vec, num_keys);
    sort_keys(d_vec, num_keys, g_allocator, false);
}
// 正态分布
void generateNormalDoubles(double* d_vec, uint num_keys, curandGenerator_t generator,
                           CachingDeviceAllocator& g_allocator, double* parameters) {
    curandGenerateNormalDouble(generator, d_vec, num_keys, parameters[0], parameters[1]);
}
void generateSortedIncNormalDoubles(double* d_vec, uint num_keys, curandGenerator_t generator,
                                    CachingDeviceAllocator& g_allocator, double* parameters) {
    curandGenerateNormalDouble(generator, d_vec, num_keys, parameters[0], parameters[1]);
    sort_keys(d_vec, num_keys, g_allocator, true);
}
void generateSortedDecNormalDoubles(double* d_vec, uint num_keys, curandGenerator_t generator,
                                    CachingDeviceAllocator& g_allocator, double* parameters) {
    curandGenerateNormalDouble(generator, d_vec, num_keys, parameters[0], parameters[1]);
    sort_keys(d_vec, num_keys, g_allocator, false);
}
// 对数正态分布
void generateLogNormalDoubles(double* d_vec, uint num_keys, curandGenerator_t generator,
                              CachingDeviceAllocator& g_allocator, double* parameters) {
    curandGenerateLogNormalDouble(generator, d_vec, num_keys, parameters[0], parameters[1]);
}
void generateSortedIncLogNormalDoubles(double* d_vec, uint num_keys, curandGenerator_t generator,
                                       CachingDeviceAllocator& g_allocator, double* parameters) {
    curandGenerateLogNormalDouble(generator, d_vec, num_keys, parameters[0], parameters[1]);
    sort_keys(d_vec, num_keys, g_allocator, true);
}
void generateSortedDecLogNormalDoubles(double* d_vec, uint num_keys, curandGenerator_t generator,
                                       CachingDeviceAllocator& g_allocator, double* parameters) {
    curandGenerateLogNormalDouble(generator, d_vec, num_keys, parameters[0], parameters[1]);
    sort_keys(d_vec, num_keys, g_allocator, false);
}

// void generateBucketKillerFloats(float* d_vec, uint num_keys, curandGenerator_t generator) {
//     int i;
//     float* d_generated = d_vec;
//     curandGenerateUniform(generator, d_generated, num_keys);
//     thrust::device_ptr<unsigned int> dev_ptr((uint*)d_generated);
//     thrust::for_each(dev_ptr, dev_ptr + num_keys, makeSmallFloat());
//     thrust::sort(dev_ptr, dev_ptr + num_keys);

//     float* h_vec = (float*)malloc(num_keys * sizeof(float));
//     cudaMemcpy(h_vec, d_generated, num_keys * sizeof(float), cudaMemcpyDeviceToHost);

//     for (i = -126; i < 127; i++) {
//         h_vec[i + 126] = pow(2.0f, (float)i);
//     }
//     cudaMemcpy(d_generated, h_vec, num_keys * sizeof(float), cudaMemcpyHostToDevice);
//     free(h_vec);
// }

template <typename KeyT>
void (*returnGenFunction(Distribution distribution, SortType sort))(KeyT*, uint, curandGenerator_t, CachingDeviceAllocator&, double*) {
    typedef void (*ptrToGeneratingFunction)(KeyT*, uint, curandGenerator_t, CachingDeviceAllocator&, double*);
    if (typeid(KeyT) == typeid(uint)) {
        if (distribution == UNIFORM && sort == NO) return (ptrToGeneratingFunction)generateUniformUints;
        if (distribution == UNIFORM && sort == INC) return (ptrToGeneratingFunction)generateSortedIncUniformUints;
        if (distribution == UNIFORM && sort == DEC) return (ptrToGeneratingFunction)generateSortedDecUniformUints;
        if (distribution == POISSON && sort == NO) return (ptrToGeneratingFunction)generatePoissonUints;
        if (distribution == POISSON && sort == INC) return (ptrToGeneratingFunction)generateSortedIncPoissonUints;
        if (distribution == POISSON && sort == DEC) return (ptrToGeneratingFunction)generateSortedDecPoissonUints;
        return (ptrToGeneratingFunction)generateUniformUints;
    } else if (typeid(KeyT) == typeid(unsigned long long)) {
        if (distribution == UNIFORM && sort == NO) return (ptrToGeneratingFunction)generateUniformUlongs;
        if (distribution == UNIFORM && sort == INC) return (ptrToGeneratingFunction)generateSortedIncUniformUlongs;
        if (distribution == UNIFORM && sort == DEC) return (ptrToGeneratingFunction)generateSortedDecUniformUlongs;
        return (ptrToGeneratingFunction)generateUniformUlongs;
    } else if (typeid(KeyT) == typeid(int)) {
        if (distribution == UNIFORM && sort == NO) return (ptrToGeneratingFunction)generateUniformInts;
        if (distribution == UNIFORM && sort == INC) return (ptrToGeneratingFunction)generateSortedIncUniformInts;
        if (distribution == UNIFORM && sort == DEC) return (ptrToGeneratingFunction)generateSortedDecUniformInts;
        return (ptrToGeneratingFunction)generateUniformInts;
    } else if (typeid(KeyT) == typeid(long long)) {
        if (distribution == UNIFORM && sort == NO) return (ptrToGeneratingFunction)generateUniformLongs;
        if (distribution == UNIFORM && sort == INC) return (ptrToGeneratingFunction)generateSortedIncUniformLongs;
        if (distribution == UNIFORM && sort == DEC) return (ptrToGeneratingFunction)generateSortedDecUniformLongs;
        return (ptrToGeneratingFunction)generateUniformLongs;
    } else if (typeid(KeyT) == typeid(float)) {
        if (distribution == UNIFORM && sort == NO) return (ptrToGeneratingFunction)generateUniformFloats;
        if (distribution == UNIFORM && sort == INC) return (ptrToGeneratingFunction)generateSortedIncUniformFloats;
        if (distribution == UNIFORM && sort == DEC) return (ptrToGeneratingFunction)generateSortedDecUniformFloats;
        if (distribution == NORMAL && sort == NO) return (ptrToGeneratingFunction)generateNormalFloats;
        if (distribution == NORMAL && sort == INC) return (ptrToGeneratingFunction)generateSortedIncNormalFloats;
        if (distribution == NORMAL && sort == DEC) return (ptrToGeneratingFunction)generateSortedDecNormalFloats;
        if (distribution == LOG_NORMAL && sort == NO) return (ptrToGeneratingFunction)generateLogNormalFloats;
        if (distribution == LOG_NORMAL && sort == INC) return (ptrToGeneratingFunction)generateSortedIncLogNormalFloats;
        if (distribution == LOG_NORMAL && sort == DEC) return (ptrToGeneratingFunction)generateSortedDecLogNormalFloats;
        return (ptrToGeneratingFunction)generateUniformFloats;
    } else {  // double
        if (distribution == UNIFORM && sort == NO) return (ptrToGeneratingFunction)generateUniformDoubles;
        if (distribution == UNIFORM && sort == INC) return (ptrToGeneratingFunction)generateSortedIncUniformDoubles;
        if (distribution == UNIFORM && sort == DEC) return (ptrToGeneratingFunction)generateSortedDecUniformDoubles;
        if (distribution == NORMAL && sort == NO) return (ptrToGeneratingFunction)generateNormalDoubles;
        if (distribution == NORMAL && sort == INC) return (ptrToGeneratingFunction)generateSortedIncNormalDoubles;
        if (distribution == NORMAL && sort == DEC) return (ptrToGeneratingFunction)generateSortedDecNormalDoubles;
        if (distribution == LOG_NORMAL && sort == NO) return (ptrToGeneratingFunction)generateLogNormalDoubles;
        if (distribution == LOG_NORMAL && sort == INC) return (ptrToGeneratingFunction)generateSortedIncLogNormalDoubles;
        if (distribution == LOG_NORMAL && sort == DEC) return (ptrToGeneratingFunction)generateSortedDecLogNormalDoubles;
        return (ptrToGeneratingFunction)generateUniformDoubles;
    }
}

char* returnNameOfGenerators(DataType type, Distribution distribution, SortType sort, double* parameters) {
    char* name = (char*)malloc(sizeof(char) * 200);
    char buf[200];
    strcpy(name, "");

    if (sort == INC) strcat(name, "SORTED INC ");
    if (sort == DEC) strcat(name, "SORTED DEC ");

    if (distribution == UNIFORM) {
        strcat(name, "UNIFORM ");
        if (type == FLOAT || type == DOUBLE) strcat(name, "U(0,1) ");
    }
    if (distribution == POISSON) strcat(name, "POISSON ");
    if (distribution == NORMAL) strcat(name, "NORMAL ");
    if (distribution == LOG_NORMAL) strcat(name, "LOG_NORMAL ");

    if (type == UINT) strcat(name, "UINTS");
    if (type == ULONG) strcat(name, "UNSIGNED_LONG_LONGS");
    if (type == INT) strcat(name, "INTS");
    if (type == LONG) strcat(name, "LONG_LONGS");
    if (type == FLOAT) strcat(name, "FLOATS");
    if (type == DOUBLE) strcat(name, "DOUBLES");

    if (distribution == POISSON) {
        snprintf(buf, 200, " (lambda: %.2e)  ", parameters[0]);
        strcat(name, buf);
    }
    if (distribution == NORMAL) {
        snprintf(buf, 200, " (mu: %.2e, sigma: %.2e)  ", parameters[0], parameters[1]);
        strcat(name, buf);
    }
    if (distribution == LOG_NORMAL) {
        snprintf(buf, 200, " (mu: %.2e, sigma: %.2e)  ", parameters[0], parameters[1]);
        strcat(name, buf);
    }

    return name;
}
