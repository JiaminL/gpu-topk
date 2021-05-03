#include <cuda.h>
#include <curand.h>
#include <cuda_runtime_api.h>
#include <cub/util_allocator.cuh>

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <algorithm>
#include <numeric>
#include <unistd.h>

#include "printFunctions.cuh"
#include "generateProblems.cuh"
#include "sort.cuh"
#include "sortTopK.cuh"
#include "radixSelectTopK.cuh"
#include "bitonicTopK.cuh"
#include "thresholdTopK.cuh"
#include "impreciseBitonicTopK.cuh"
// #include "testTime.cuh"

#define NEED_PRINT_EVERY_TESTING false
#define NEED_PRINT_DIFF false
#define NEED_ANALYSIS true

#define SETUP_TIMING()       \
    cudaEvent_t start, stop; \
    cudaEventCreate(&start); \
    cudaEventCreate(&stop);

#define TIME_FUNC(f, t)                        \
    {                                          \
        cudaEventRecord(start, 0);             \
        f;                                     \
        cudaEventRecord(stop, 0);              \
        cudaEventSynchronize(stop);            \
        cudaEventElapsedTime(&t, start, stop); \
    }

#define NUMBEROFALGORITHMS 5
#define INIT_FUNCTIONS()                                                                           \
    typedef cudaError_t (*ptrToTimingFunction)(KeyT*, uint, uint, KeyT*, CachingDeviceAllocator&); \
    const char* namesOfTimingFunctions[NUMBEROFALGORITHMS] = {                                     \
        "Sort TopK",                                                                               \
        "Radix Select",                                                                            \
        "Bitonic TopK",                                                                            \
        "Threshold TopK",                                                                          \
        "Imprecise Bitonic",                                                                       \
    };                                                                                             \
    ptrToTimingFunction arrayOfTimingFunctions[NUMBEROFALGORITHMS] = {                             \
        &sortTopK<KeyT>,                                                                           \
        &radixSelectTopK<KeyT>,                                                                    \
        &bitonicTopK<KeyT>,                                                                        \
        &thresholdTopK<KeyT>,                                                                      \
        &impreciseBitonicTopK<KeyT>,                                                               \
    };
#define SET_ALGORITHMS_RUN()                            \
    {                                                   \
        fill_n(algorithmsToRun, NUMBEROFALGORITHMS, 1); \
        algorithmsToRun[0] = 0;                         \
    }

using namespace std;

CachingDeviceAllocator g_allocator(true);  // Caching allocator for device memory

template <typename KeyT>
void compareAlgorithms(uint size, uint k, uint numTests, uint* algorithmsToTest, uint generateType) {
    KeyT* d_vec;
    KeyT* d_vec_copy;
    KeyT* d_res;
    float timeArray[NUMBEROFALGORITHMS][numTests];
    double totalTimesPerAlgorithm[NUMBEROFALGORITHMS];
    float averageTimesPerAlgorithm[NUMBEROFALGORITHMS];
    float minTimesPerAlgorithm[NUMBEROFALGORITHMS];
    float maxTimesPerAlgorithm[NUMBEROFALGORITHMS];
    double standardPerAlgorithm[NUMBEROFALGORITHMS];  // standard deviation 标准差
    KeyT* resultsArray[NUMBEROFALGORITHMS][numTests];

    uint winnerArray[numTests];
    uint timesWon[NUMBEROFALGORITHMS];
    uint i, j, m, x;
    int runOrder[NUMBEROFALGORITHMS];

    unsigned long long seed;
    timeval t1;
    float runtime;

    for (i = 0; i < numTests; i++)
        for (j = 0; j < NUMBEROFALGORITHMS; j++)
            resultsArray[j][i] = new KeyT[k];

    SETUP_TIMING()

    typedef void (*ptrToGeneratingFunction)(KeyT*, uint, curandGenerator_t, CachingDeviceAllocator&);
    // these are the functions that can be called
    INIT_FUNCTIONS()

    ptrToGeneratingFunction* arrayOfGenerators;
    const char** namesOfGeneratingFunctions;
    // this is the array of names of functions that generate problems of this type, ie float, double, or uint
    namesOfGeneratingFunctions = returnNamesOfGenerators<KeyT>();
    arrayOfGenerators = (ptrToGeneratingFunction*)returnGenFunctions<KeyT>();

    // zero out the totals and times won
    bzero(totalTimesPerAlgorithm, NUMBEROFALGORITHMS * sizeof(uint));
    bzero(timesWon, NUMBEROFALGORITHMS * sizeof(uint));
    // allocate space for d_vec, and d_vec_copy
    cudaMalloc(&d_vec, size * sizeof(KeyT));
    cudaMalloc(&d_vec_copy, size * sizeof(KeyT));
    cudaMalloc(&d_res, k * sizeof(KeyT));

    // create the random generator.
    curandGenerator_t generator;

#if NEED_ANALYSIS
    KeyT* h_sort_vec = (KeyT*)malloc(sizeof(KeyT) * size);

    uint total_topk_times[NUMBEROFALGORITHMS];  // 所有 numTests 个测试的结果中正确的 top-k 出现的总次数
    double avg_topk_rate[NUMBEROFALGORITHMS];   // 所有 numTests 个测试结果中正确 top-k 的比例均值
    fill_n(total_topk_times, NUMBEROFALGORITHMS, 0);

    uint tolerance = 2;                                         // 评价指标容忍度
    long int weight = ((tolerance * k * 2 - (k - 1)) * k) / 2;  // 评价指标标准化权重
    long int sum_noWeight_analyze_1[NUMBEROFALGORITHMS];        // 所有 numTests 个无权评价指标之和
    double avg_analyze_1[NUMBEROFALGORITHMS];                   // 所有 numTests 个有权评价指标的均值
    fill_n(sum_noWeight_analyze_1, NUMBEROFALGORITHMS, 0);

    bool res_error[NUMBEROFALGORITHMS];  // 结果中出现了原数据中没有的数，或者出现次数大于原数据中出现的次数，判定该算法出错
    fill_n(res_error, NUMBEROFALGORITHMS, false);
#endif

    printf("The distribution is: %s\n", namesOfGeneratingFunctions[generateType]);
    for (i = 0; i < numTests; i++) {
        // cudaDeviceReset();
        gettimeofday(&t1, NULL);
        seed = t1.tv_usec * t1.tv_sec;
        srand(seed);

        for (m = 0; m < NUMBEROFALGORITHMS; m++) {
            runOrder[m] = m;
        }
        std::random_shuffle(runOrder, runOrder + NUMBEROFALGORITHMS);

        curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(generator, seed);
        // curandSetPseudoRandomGeneratorSeed(generator, 0);

#if NEED_PRINT_EVERY_TESTING
        printf("Running test %u of %u for size: %u and k: %u\n", i + 1, numTests, size, k);
#endif
        // generate the random vector using the specified distribution
        arrayOfGenerators[generateType](d_vec, size, generator, g_allocator);
        if (generateType == 1 || generateType == 2) {  // 如果是升序或降序（需要申请临时空间用来排序）
            if (size == (uint)(2 << 30) / sizeof(KeyT)) {  // 2GB
                // printf("sleep 100\n");
                usleep(50000);                                    // sleep 50 ms
            } else if (size == (uint)(1 << 30) / sizeof(KeyT)) {  // 1GB
                usleep(20000);                                    // sleep 20 ms
            }
        }

        // KeyT* h_vec = new KeyT[size];
        // cudaMemcpy(h_vec, d_vec, size * sizeof(KeyT), cudaMemcpyDeviceToHost);
        // h_vec[0] = (KeyT)(-1034);
        // cudaMemcpy(d_vec, h_vec, size * sizeof(KeyT), cudaMemcpyHostToDevice);
        // delete[] h_vec;

        // copy the vector to d_vec_copy, which will be used to restore it later
        cudaMemcpy(d_vec_copy, d_vec, size * sizeof(KeyT), cudaMemcpyDeviceToDevice);

        winnerArray[i] = 0;
        float currentWinningTime = INFINITY;
        // run the various timing functions
        for (x = 0; x < NUMBEROFALGORITHMS; x++) {
            j = runOrder[x];
            if (algorithmsToTest[j]) {
                // run timing function j
                TIME_FUNC(arrayOfTimingFunctions[j](d_vec_copy, size, k, d_res, g_allocator), runtime);
                // 我猜测 GPU 释放空间与函数返回是异步的，上一次测试申请的空间还没有释放结束，下一次测试函数就开始了
                // 由于我的 GPU 显存只有 8GB，如果原始数据大小为 2GB，因为 GPU 没有更多的 2GB 空间用来分配（d_vec_copy, d_vec 已经使用了 4GB）
                // 下一次测试必须等待，导致除第一个上 GPU 的测试外，其余测试都有 30 ~ 50 ms 不等的延时
                if (j == 0) {                                      // 如果是 sort top-k 算法（需要申请临时空间）
                    if (size == (uint)(2 << 30) / sizeof(KeyT)) {  // 2GB
                        // printf("sleep 100\n");
                        usleep(50000);                                    // sleep 50 ms
                    } else if (size == (uint)(1 << 30) / sizeof(KeyT)) {  // 1GB
                        usleep(20000);                                    // sleep 20 ms
                    }
                }
#if NEED_PRINT_EVERY_TESTING
                printf("\tTESTING: %-2u %-20s runtime: %f ms\n", j, namesOfTimingFunctions[j], runtime);
#endif

                // check for error
                cudaError_t error = cudaGetLastError();
                if (error != cudaSuccess) {
                    // print the CUDA error message and exit
                    printf("CUDA error: %s\n", cudaGetErrorString(error));
                    exit(-1);
                }

                // record the time result
                timeArray[j][i] = runtime;

                // record the value returned
                cudaMemcpy(resultsArray[j][i], d_res, k * sizeof(KeyT), cudaMemcpyDeviceToHost);
                std::sort(resultsArray[j][i], resultsArray[j][i] + k, std::greater<KeyT>());

                // update the current "winner" if necessary
                if (timeArray[j][i] < currentWinningTime) {
                    currentWinningTime = runtime;
                    winnerArray[i] = j;
                }

                // perform clean up
                cudaMemcpy(d_vec_copy, d_vec, size * sizeof(KeyT), cudaMemcpyDeviceToDevice);
                cudaMemset(d_res, 0, k * sizeof(KeyT));
            }
        }

        curandDestroyGenerator(generator);

#if NEED_ANALYSIS
        // analyze
        sort(d_vec_copy, size, g_allocator);
        cudaMemcpy(h_sort_vec, d_vec_copy, sizeof(KeyT) * size, cudaMemcpyDeviceToHost);
        // 前提：找到的结果数组已经按从大到小排列
        for (j = 0; j < NUMBEROFALGORITHMS; j++) {
            if (!res_error[j] && algorithmsToTest[j]) {
                for (uint res_idx = 0, sort_idx = 0; res_idx < k; res_idx++) {
                    while (sort_idx < size && h_sort_vec[sort_idx] != resultsArray[j][i][res_idx]) sort_idx++;
                    if (sort_idx == size) {
                        res_error[j] = true;
                        break;
                    } else {
                        sum_noWeight_analyze_1[j] += (int)(tolerance * k) - (int)sort_idx;
                        if (sort_idx < k) total_topk_times[j]++;
                    }
                }
            }
        }
#endif
    }

#if NEED_ANALYSIS
    for (j = 0; j < NUMBEROFALGORITHMS; j++) {
        if (!res_error[j] && algorithmsToTest[j]) {
            avg_analyze_1[j] = sum_noWeight_analyze_1[j] / (weight * numTests);
            avg_topk_rate[j] = total_topk_times[j] / (numTests * k);
        }
    }
    free(h_sort_vec);
#endif

    // calculate the statistical data
    fill_n(standardPerAlgorithm, NUMBEROFALGORITHMS, 0);
    for (j = 0; j < NUMBEROFALGORITHMS; j++) {
        maxTimesPerAlgorithm[j] = *max_element(timeArray[j], timeArray[j] + numTests);
        minTimesPerAlgorithm[j] = *min_element(timeArray[j], timeArray[j] + numTests);
        totalTimesPerAlgorithm[j] = accumulate(timeArray[j], timeArray[j] + numTests, 0.0);
        // 计算均值
        averageTimesPerAlgorithm[j] = totalTimesPerAlgorithm[j] / numTests;
        // 计算方差
        if (numTests > 1) {
            for (i = 0; i < numTests; i++) {
                standardPerAlgorithm[j] += pow(timeArray[j][i] - averageTimesPerAlgorithm[j], 2);
            }
            standardPerAlgorithm[j] = sqrt(standardPerAlgorithm[j] / (numTests - 1));
        }
    }

    // count the number of times each algorithm won.
    for (i = 0; i < numTests; i++) {
        timesWon[winnerArray[i]]++;
    }

#if NEED_PRINT_EVERY_TESTING
    printf("\n\n");
#endif

    // print out the statistical data
    int total_algorithms = accumulate(algorithmsToTest, algorithmsToTest + NUMBEROFALGORITHMS, 0);
    // print out header of the table
    printf("%-20s %-15s %-15s %-15s", "algorithm", "minimum (ms)", "maximum (ms)", "average (ms)");
    if (numTests > 1) printf(" %-15s", "std dev");
    if (total_algorithms > 1) printf(" %-15s", "won times");
#if NEED_ANALYSIS
    printf(" %-15s %-15s", "top-k rate (%)", "analyze 1");
#endif
    printf("\n");
    // print out data
    for (i = 0; i < NUMBEROFALGORITHMS; i++) {
        if (algorithmsToTest[i]) {
            printf("%-20s %-15f %-15f %-15f", namesOfTimingFunctions[i], minTimesPerAlgorithm[i],
                   maxTimesPerAlgorithm[i], averageTimesPerAlgorithm[i]);
            if (numTests > 1) printf(" %-15f", standardPerAlgorithm[i]);
            if (total_algorithms > 1) printf(" %-15d", timesWon[i]);
#if NEED_ANALYSIS
            if (res_error[i])
                printf(" %-15s %-15s", "ERROR", "ERROR");
            else
                printf(" %-15.2f %-15.3f", avg_topk_rate[i] * 100, avg_analyze_1[i]);
#endif
            printf("\n");
        }
    }
    printf("\n");

#if NEED_PRINT_DIFF
    if (algorithmsToTest[0]) {
        for (i = 0; i < numTests; i++) {
            for (j = 1; j < NUMBEROFALGORITHMS; j++) {
                if (algorithmsToTest[j]) {
                    for (int m = 0; m < k; m++)
                        if (resultsArray[j][i][m] != resultsArray[0][i][m]) {
                            std::cout << namesOfTimingFunctions[j] << " did not return the correct answer on test" << i + 1 << std::endl;
                            std::cout << "Method:\t";
                            // PrintFunctions::printArray<KeyT>(resultsArray[j][i], k);
                            std::cout << "Sort:\t";
                            // PrintFunctions::printArray<KeyT>(resultsArray[0][i], k);
                            std::cout << "\n";
                            for (int l = 0; l < k; l++) {
                                std::cout << (KeyT)resultsArray[j][i][l] << "  " << (KeyT)resultsArray[0][i][l] << std::endl;
                            }
                            break;
                        }
                }
            }
        }
    }
#endif

    // free memory
    for (j = 0; j < NUMBEROFALGORITHMS; j++)
        for (i = 0; i < numTests; i++)
            delete[] resultsArray[j][i];
    cudaFree(d_vec);
    cudaFree(d_vec_copy);
    cudaFree(d_res);
}

template <typename KeyT>
void runTests(uint generateType, int K, uint startPower, uint stopPower, uint timesToTestEachK = 3) {
    // Algorithms To Run
    // timeSort, timeRadixSelect, timeBitonicTopK
    uint algorithmsToRun[NUMBEROFALGORITHMS];
    SET_ALGORITHMS_RUN();
    for (uint power = startPower; power <= stopPower; power++) {
        uint size = 1 << power;
        printf("NOW STARTING A NEW TOP-K [size: 2^%u (%u), k: %d]\n", power, size, K);
        compareAlgorithms<KeyT>(size, K, timesToTestEachK, algorithmsToRun, generateType);
    }
}

int main(int argc, char** argv) {
    uint testCount;
    int K;
    uint type, distributionType, startPower, stopPower;
    if (argc == 7) {
        type = atoi(argv[1]);
        distributionType = atoi(argv[2]);
        K = atoi(argv[3]);
        testCount = atoi(argv[4]);
        startPower = atoi(argv[5]);
        stopPower = atoi(argv[6]);
    } else {
        printf("Please enter the type of value you want to test:\n1-float\n2-double\n3-uint\n");
        cin >> type;
        printf("Please enter distribution type: ");
        cin >> distributionType;
        printf("Please enter K: ");
        cin >> K;
        printf("Please enter number of tests to run per K: ");
        cin >> testCount;
        printf("Please enter start power (dataset size starts at 2^start)(max val: 29): ");
        cin >> startPower;
        printf("Please enter stop power (dataset size stops at 2^stop)(max val: 29): ");
        cin >> stopPower;
    }

    switch (type) {
        case 1:
            runTests<float>(distributionType, K, startPower, stopPower, testCount);
            break;
        case 2:
            runTests<double>(distributionType, K, startPower, stopPower, testCount);
            break;
        case 3:
            runTests<unsigned int>(distributionType, K, startPower, stopPower, testCount);
            break;
        default:
            printf("You entered and invalid option, now exiting\n");
            break;
    }

    return 0;
}
