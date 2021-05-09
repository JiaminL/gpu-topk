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
#include <getopt.h>
#include <string.h>

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
        algorithmsToRun[4] = 0;                         \
    }

using namespace std;

CachingDeviceAllocator g_allocator(true);  // Caching allocator for device memory

template <typename KeyT, typename FuncType>
void compareAlgorithms(uint size, uint k, uint numTests, uint* algorithmsToTest, double* parameter,
                       char* generateName, FuncType generateFunc, bool genNeedSort) {
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

    // these are the functions that can be called
    INIT_FUNCTIONS()

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

    uint tolerance = 2;                                                   // 评价指标容忍度
    long int weight = ((long int)(tolerance * k * 2 - (k - 1)) * k) / 2;  // 评价指标标准化权重
    long int sum_noWeight_analyze_1[NUMBEROFALGORITHMS];                  // 所有 numTests 个无权评价指标之和
    double avg_analyze_1[NUMBEROFALGORITHMS];                             // 所有 numTests 个有权评价指标的均值
    fill_n(sum_noWeight_analyze_1, NUMBEROFALGORITHMS, 0);

    bool res_error[NUMBEROFALGORITHMS];  // 结果中出现了原数据中没有的数，或者出现次数大于原数据中出现的次数，判定该算法出错
    fill_n(res_error, NUMBEROFALGORITHMS, false);
#endif

    printf("The distribution is: %s\n", generateName);
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
        generateFunc(d_vec, size, generator, g_allocator, parameter);

        if (genNeedSort) {                                 // 如果是升序或降序（需要申请临时空间用来排序）
            if (size == (uint)(2 << 30) / sizeof(KeyT)) {  // 2GB
                // printf("sleep 100\n");
                usleep(50000);                                    // sleep 50 ms
            } else if (size == (uint)(1 << 30) / sizeof(KeyT)) {  // 1GB
                usleep(20000);                                    // sleep 20 ms
            }
        }

        // KeyT* h_vec = new KeyT[size];
        // cudaMemcpy(h_vec, d_vec, size * sizeof(KeyT), cudaMemcpyDeviceToHost);
        // cout << h_vec[0] << " " << h_vec[1] << " " << h_vec[2] << " " << h_vec[3] << " " << h_vec[4] << " " << endl;
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
                    if (h_sort_vec[sort_idx] != resultsArray[j][i][res_idx] && sort_idx >= 1 && h_sort_vec[sort_idx - 1] == resultsArray[j][i][res_idx]) continue;
                    while (sort_idx < size && h_sort_vec[sort_idx] != resultsArray[j][i][res_idx]) sort_idx++;
                    if (sort_idx == size) {
                        res_error[j] = true;
                        break;
                    } else {
                        sum_noWeight_analyze_1[j] += (int)(tolerance * k) - (int)sort_idx;
                        if (sort_idx < k) total_topk_times[j]++;
                    }
                    sort_idx++;
                }
            }
        }
#endif
    }

#if NEED_ANALYSIS
    for (j = 0; j < NUMBEROFALGORITHMS; j++) {
        if (!res_error[j] && algorithmsToTest[j]) {
            avg_analyze_1[j] = sum_noWeight_analyze_1[j] / (double)(weight * numTests);
            avg_topk_rate[j] = total_topk_times[j] / (double)(numTests * k);
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
                printf(" %-15.4f %-15f", avg_topk_rate[i] * 100, avg_analyze_1[i]);
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
void runTests(uint k, uint startPower, uint stopPower, uint testCount, double* parameter,
              DataType type, Distribution distribution, SortType sort) {
    // Algorithms To Run
    uint algorithmsToRun[NUMBEROFALGORITHMS];
    SET_ALGORITHMS_RUN();

    // 获得对应的生成随机数的函数
    typedef void (*ptrToGeneratingFunction)(KeyT*, uint, curandGenerator_t, CachingDeviceAllocator&, double*);
    ptrToGeneratingFunction generateFunc = (ptrToGeneratingFunction)returnGenFunction<KeyT>(distribution, sort);
    char* generateName = returnNameOfGenerators(type, distribution, sort);
    bool genNeedSort = (sort != NO);

    for (uint power = startPower; power <= stopPower; power++) {
        uint size = 1 << power;
        printf("NOW STARTING A NEW TOP-K [size: 2^%u (%u), k: %d]\n", power, size, k);
        // compareAlgorithms<KeyT>(size, k, testCount, algorithmsToRun, 0);
        // compareAlgorithms<KeyT>(size, k, testCount, algorithmsToRun, parameter, generateName, genNeedSort);
        compareAlgorithms<KeyT, ptrToGeneratingFunction>(size, k, testCount, algorithmsToRun, parameter, generateName, generateFunc, genNeedSort);
    }

    free(generateName);
}

void getParameters(int argc, char** argv,
                   uint& u_k, uint& u_start_power, uint& u_stop_power, uint& u_test_count, double* parameter,
                   DataType& type, Distribution& distribution, SortType& sort) {
    const int max_4b_power = 29;
    const int max_8b_power = 28;

    int k = -1, start_power = -1, stop_power = -1;
    int test_count = 3;
    parameter[0] = 0;
    parameter[1] = 1;
    type = UINT;
    distribution = UNIFORM;
    sort = NO;

    enum Error { NO_ERROR,
                 TYPE_ERROR,
                 DISTRIBUTION_ERROR,
                 SORT_ERROR,
                 TESTCOUNT_ERROR,
                 K_ERROR,
                 N_ERROR } error = NO_ERROR;

    static struct option long_options[] =
        {
            {"type", 1, NULL, 't'},
            {"distribution", 1, NULL, 'd'},
            {"testcount", 1, NULL, 'c'},
            {"p1", 1, NULL, '1'},
            {"p2", 1, NULL, '2'},
            {"sort", 1, NULL, 's'},
            {"startpower", 1, NULL, 'a'},
            {"stoppower", 1, NULL, 'b'},
            {NULL, 0, NULL, 0},
        };
    int ch;
    while ((ch = getopt_long(argc, argv, "t:d:s:k:a:b:c:1:2:", long_options, NULL)) != -1) {
        if (error != NO_ERROR) break;
        switch (ch) {
            case 't':  // type
                if (strcmp(optarg, "uint") == 0)
                    type = UINT;
                else if (strcmp(optarg, "ulong") == 0)
                    type = ULONG;
                else if (strcmp(optarg, "int") == 0)
                    type = INT;
                else if (strcmp(optarg, "long") == 0)
                    type = LONG;
                else if (strcmp(optarg, "float") == 0)
                    type = FLOAT;
                else if (strcmp(optarg, "double") == 0)
                    type = DOUBLE;
                else
                    error = TYPE_ERROR;
                break;
            case 'd':  // distribution
                if (strcmp(optarg, "uniform") == 0)
                    distribution = UNIFORM;
                else if (strcmp(optarg, "poisson") == 0)
                    distribution = POISSON;
                else if (strcmp(optarg, "normal") == 0)
                    distribution = NORMAL;
                else if (strcmp(optarg, "lognormal") == 0)
                    distribution = LOG_NORMAL;
                else
                    error = DISTRIBUTION_ERROR;
                break;
            case 's':  // sort
                if (strcmp(optarg, "inc") == 0)
                    sort = INC;
                else if (strcmp(optarg, "dec") == 0)
                    sort = DEC;
                else
                    error = SORT_ERROR;
                break;
            case 'k':  // k
                k = atoi(optarg);
                if (k <= 0)
                    error = K_ERROR;
                break;
            case 'c':  // test_count
                test_count = atoi(optarg);
                if (test_count <= 0)
                    error = TESTCOUNT_ERROR;
                break;
            case 'a':  // log_n begin
                start_power = atoi(optarg);
                if (start_power < 0)
                    error = N_ERROR;
                break;
            case 'b':  // log_n end
                stop_power = atoi(optarg);
                if (stop_power < 0)
                    error = N_ERROR;
                break;
            case '1':  // parameter 1
                parameter[0] = atof(optarg);
                break;
            case '2':  // parameter 2
                parameter[1] = atof(optarg);
            case '?':  // 未定义的选项
                printf("unknown option \n");
                break;
            default:
                printf("default \n");
        }
    }

    // deal error
    if (error == TYPE_ERROR) {
        cerr << "error: 仅支持类型：int, long, uint, ulong, float, double\n";
        exit(1);
    }
    if (error != DISTRIBUTION_ERROR &&
        ((type == UINT && distribution != UNIFORM && distribution != POISSON) ||
         (type == ULONG && distribution != UNIFORM) ||
         (type == INT && distribution != UNIFORM) ||
         (type == LONG && distribution != UNIFORM) ||
         (type == FLOAT && distribution != UNIFORM && distribution != NORMAL && distribution != LOG_NORMAL) ||
         (type == DOUBLE && distribution != UNIFORM && distribution != NORMAL && distribution != LOG_NORMAL))) {
        error = DISTRIBUTION_ERROR;
    }
    if (error == DISTRIBUTION_ERROR) {
        if (type == UINT) cerr << "error: uint（unsigned int）类型仅支持均匀分布 uniform, 泊松分布 possion\n";
        if (type == ULONG) cerr << "error: ulong（unsigned long long）类型仅支持均匀分布 uniform\n";
        if (type == INT) cerr << "error: int 类型仅支持均匀分布 uniform\n";
        if (type == LONG) cerr << "error: long（long long）类型仅支持均匀分布 uniform\n";
        if (type == FLOAT) cerr << "error: float 类型仅支持正态分布 normal, 对数正态分布 lognormal, U(0,1)均匀分布 uniform\n";
        if (type == DOUBLE) cerr << "error: double 类型仅支持正态分布 normal, 对数正态分布 lognormal, U(0,1)均匀分布 uniform\n";
        exit(1);
    }
    if (error == SORT_ERROR) {
        cerr << "error: 有两种排序：增序 inc，降序 dec\n";
        exit(1);
    }
    if (k == -1) {
        cerr << "error: 请输入 k 值，如: -k 32\n";
        exit(1);
    }
    if (error == K_ERROR) {
        cerr << "error: k 值必须为正整数\n";
        exit(1);
    }
    if (error == TESTCOUNT_ERROR) {
        cerr << "error: testcount 必须为正整数\n";
        exit(1);
    }

    if (error == N_ERROR) {
        cerr << "error: start_power 与 stop_power 须为正整数\n";
        exit(1);
    }
    if (start_power == -1) start_power = log2_32(k) + 1;  // 未设置 start_power
    if (start_power <= log2_32(k)) {
        cerr << "error: 2^startpower 必须大于 k\n";
        exit(1);
    }
    int max_power = (type == UINT || type == FLOAT || type == INT) ? max_4b_power : max_8b_power;  // 未设置 stop_power
    if (stop_power == -1) stop_power = max_power;
    if (stop_power > max_power) {
        cerr << "error: 2^stoppower 个该类型数据，已超过 GPU 能承载的最大数量\n";
        exit(1);
    }
    if (start_power > stop_power) {
        if (start_power == log2_32(k) + 1 && stop_power == max_power)
            cerr << "error: k 值过大\n";
        else if (stop_power <= log2_32(k))
            cerr << "error: 2^stoppower 必须大于 k\n";
        else
            cerr << "error: startpower 不应大于 stoppower\n";
        exit(1);
    }

    switch (distribution) {
        case POISSON:
            if (parameter[0] <= 0) {
                cerr << "error: POISSON 分布参数 p1 (lambda) 须大于 0\n";
                exit(1);
            }
            break;
        case NORMAL:
            if (parameter[1] < 0) {
                cerr << "error: NORMAL 分布参数 p2 (std dev) 须大于 0\n";
                exit(1);
            }
    }
    u_k = (uint)k;
    u_start_power = (uint)start_power;
    u_stop_power = (uint)stop_power;
    u_test_count = (uint)test_count;
}

int main(int argc, char** argv) {
    uint k, startPower, stopPower, testCount;
    double parameter[2];
    DataType type;
    Distribution distribution;
    SortType sort;
    getParameters(argc, argv, k, startPower, stopPower, testCount, parameter, type, distribution, sort);

    switch (type) {
        case UINT:
            runTests<uint>(k, startPower, stopPower, testCount, parameter, type, distribution, sort);
            break;
        case ULONG:
            // 不知道为什么，类型写成 unsigned long long 就编译报错
            runTests<unsigned long>(k, startPower, stopPower, testCount, parameter, type, distribution, sort);
            break;
        case INT:
            runTests<int>(k, startPower, stopPower, testCount, parameter, type, distribution, sort);
            break;
        case LONG:
            runTests<long>(k, startPower, stopPower, testCount, parameter, type, distribution, sort);
            break;
        case FLOAT:
            runTests<float>(k, startPower, stopPower, testCount, parameter, type, distribution, sort);
            break;
        case DOUBLE:
            runTests<double>(k, startPower, stopPower, testCount, parameter, type, distribution, sort);
            break;
    }

    return 0;
}
