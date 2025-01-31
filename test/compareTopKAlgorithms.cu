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
#include "localMaxTopK.cuh"
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
#define INIT_FUNCTIONS()                                                                                  \
    typedef cudaError_t (*ptrToTimingFunction)(KeyT*, uint, uint, KeyT*, uint&, CachingDeviceAllocator&); \
    const char* namesOfTimingFunctions[NUMBEROFALGORITHMS] = {                                            \
        "Sort TopK",                                                                                      \
        "Radix Select",                                                                                   \
        "Bitonic TopK",                                                                                   \
        "Threshold TopK",                                                                                 \
        "Local Max TopK",                                                                                 \
    };                                                                                                    \
    ptrToTimingFunction arrayOfTimingFunctions[NUMBEROFALGORITHMS] = {                                    \
        &sortTopK<KeyT>,                                                                                  \
        &radixSelectTopK<KeyT>,                                                                           \
        &bitonicTopK<KeyT>,                                                                               \
        &thresholdTopK<KeyT>,                                                                             \
        &localMaxTopK<KeyT>,                                                                              \
    };
#define SET_ALGORITHMS_RUN()                            \
    {                                                   \
        fill_n(algorithmsToRun, NUMBEROFALGORITHMS, 1); \
        algorithmsToRun[0] = (NEED_PRINT_DIFF) ? 1 : 0; \
        algorithmsToRun[1] = 0;                         \
        algorithmsToRun[2] = 0;                         \
        algorithmsToRun[3] = 0;                         \
        algorithmsToRun[4] = 1;                         \
    }

using namespace std;

template <typename KeyT, typename FuncType>
void compareAlgorithms(uint size, uint k, uint numTests, uint* algorithmsToTest, double* parameter,
                       char* generateName, FuncType generateFunc, bool genNeedSort) {
    CachingDeviceAllocator g_allocator(2, 1, 31, (uint)(3 << 30), false);  // Caching allocator for device memory

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
    uint out_k[NUMBEROFALGORITHMS][numTests];

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

    double avg_out_k[NUMBEROFALGORITHMS];
    uint total_topk_times[NUMBEROFALGORITHMS];  // 所有 numTests 个测试的结果中正确的 top-k 出现的总次数
    double avg_topk_rate[NUMBEROFALGORITHMS];   // 所有 numTests 个测试结果中正确 top-k 的比例均值
    fill_n(total_topk_times, NUMBEROFALGORITHMS, 0);

    uint tolerance = 2;                                                     // 评价指标容忍度
    long double weight = ((long double)(tolerance * k * 2 - k) * k) / 2.0;  // 评价指标标准化权重
    long long int sum_noWeight_analyze_1[NUMBEROFALGORITHMS];               // 所有 numTests 个无权评价指标之和
    long double avg_analyze_1[NUMBEROFALGORITHMS];                          // 所有 numTests 个有权评价指标的均值
    for (int i = 0; i < NUMBEROFALGORITHMS; i++) sum_noWeight_analyze_1[i] = 0;

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
                TIME_FUNC(arrayOfTimingFunctions[j](d_vec_copy, size, k, d_res, out_k[j][i], g_allocator), runtime);
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
                std::sort(resultsArray[j][i], resultsArray[j][i] + out_k[j][i], std::greater<KeyT>());

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
                for (uint res_idx = 0, sort_idx = 0; res_idx < out_k[j][i]; res_idx++) {
                    while (sort_idx < size && h_sort_vec[sort_idx] != resultsArray[j][i][res_idx]) sort_idx++;
                    if (sort_idx == size) {
                        printf("seed: %llu", seed);
                        for (uint idx = 0; idx < out_k[j][i]; idx++) cout << resultsArray[j][i][idx] << " " << h_sort_vec[idx] << endl;
                        res_error[j] = true;
                        break;
                    } else {
                        sum_noWeight_analyze_1[j] += (long long int)(tolerance * k) - (long long int)sort_idx;
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
        if (algorithmsToTest[j]) {
            unsigned long long total_out_keys = accumulate(out_k[j], out_k[j] + numTests, 0);
            if (!res_error[j]) {
                avg_analyze_1[j] = (sum_noWeight_analyze_1[j] - 0.5 * total_out_keys) / (weight * numTests);
                avg_topk_rate[j] = total_topk_times[j] / (double)(numTests * k);
            }
            avg_out_k[j] = total_out_keys / (double)numTests;
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
    printf("%-15s %-15s %-15s", "return items", "top-k rate (%)", "analyze 1");
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
            printf("% -15.2f", avg_out_k[i]);
            if (res_error[i])
                printf(" %-15s %-15s", "ERROR", "ERROR");
            else
                printf(" %-15.4f %-15Lf", avg_topk_rate[i] * 100, avg_analyze_1[i]);
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
void runTests(uint k, uint* k_log, uint startPower, uint stopPower, uint testCount, double* parameter,
              DataType type, Distribution distribution, SortType sort) {
    // Algorithms To Run
    uint algorithmsToRun[NUMBEROFALGORITHMS];
    SET_ALGORITHMS_RUN();

    // 获得对应的生成随机数的函数
    typedef void (*ptrToGeneratingFunction)(KeyT*, uint, curandGenerator_t, CachingDeviceAllocator&, double*);
    ptrToGeneratingFunction generateFunc = (ptrToGeneratingFunction)returnGenFunction<KeyT>(distribution, sort);
    char* generateName = returnNameOfGenerators(type, distribution, sort, parameter);
    bool genNeedSort = (sort != NO);

    for (uint power = startPower; power <= stopPower; power++) {
        uint size = 1 << power;
        if (k != (~0)) {
            printf("NOW STARTING A NEW TOP-K [size: 2^%u (%u), k: %d]\n", power, size, k);
            compareAlgorithms<KeyT, ptrToGeneratingFunction>(size, k, testCount, algorithmsToRun, parameter, generateName, generateFunc, genNeedSort);
        } else {
            for (uint log = k_log[0]; log <= k_log[1]; log++) {
                k = 1 << log;
                printf("NOW STARTING A NEW TOP-K [size: 2^%u (%u), k: %d]\n", power, size, k);
                compareAlgorithms<KeyT, ptrToGeneratingFunction>(size, k, testCount, algorithmsToRun, parameter, generateName, generateFunc, genNeedSort);
            }
        }
    }

    free(generateName);
}

void getParameters(int argc, char** argv,
                   int& k, int* k_log, int& start_power, int& stop_power, int& test_count, double* parameter,
                   DataType& type, Distribution& distribution, SortType& sort) {
    const int max_4b_power = 29;
    const int max_8b_power = 28;

    k = -1;
    start_power = stop_power = -1;
    k_log[0] = k_log[1] = -1;
    test_count = 3;
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
                 KLOG_ERROR,
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
            {"klog1", 1, NULL, 'e'},
            {"klog2", 1, NULL, 'f'},
            {NULL, 0, NULL, 0},
        };
    int ch;
    while ((ch = getopt_long(argc, argv, "t:d:s:k:e:f:a:b:c:1:2:", long_options, NULL)) != -1) {
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
            case 'e':  // klog0
                k_log[0] = atoi(optarg);
                if (k_log[0] < 0)
                    error = KLOG_ERROR;
                break;
            case 'f':  // klog1
                k_log[1] = atoi(optarg);
                if (k_log[1] < 0)
                    error = KLOG_ERROR;
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
                break;
            case '?':  // 未定义的选项
                printf("unknown option: %c\n", optopt);
                exit(1);
                break;
            default:
                printf("default \n");
        }
    }

    // deal error
    int max_power = (type == UINT || type == INT || type == FLOAT) ? max_4b_power : max_8b_power;
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
        if (type == UINT) cerr << "error: uint（unsigned int）类型仅支持均匀分布 uniform, 泊松分布 poisson\n";
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

    if (error == K_ERROR) {
        cerr << "error: k 值必须为正整数\n";
        exit(1);
    }
    if (error == KLOG_ERROR) {
        cerr << "error: k log 值必须为非负整数\n";
        exit(1);
    }
    if (k == -1 && k_log[0] == -1 && k_log[1] == -1) {
        cerr << "error: 请输入 k 值\n";
        exit(1);
    }
    if (k_log[0] != -1 || k_log[1] != -1) {
        if (k_log[0] == -1)
            k_log[0] = k_log[1];
        else if (k_log[1] == -1)
            k_log[1] = k_log[0];
        else {
            int temp;
            if (k_log[0] > k_log[1]) {
                temp = k_log[0];
                k_log[0] = k_log[1];
                k_log[1] = temp;
            }
        }
        k = -1;
    }
    if ((k != -1 && log2_32(k) + 1 > max_power) || (k == -1 && k_log[1] + 1 > max_power)) {
        cerr << "error: k 值过大\n";
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

    if (start_power == -1 && stop_power == -1)  // 如果两个 power 都没有给出，默认只执行最大的数据集
        start_power = stop_power = max_power;
    else if (start_power == -1)  // 如果只给出了一个 power，那么只执行这个 power 对应的数据集
        start_power = stop_power;
    else if (stop_power == -1)
        stop_power = start_power;
    if (start_power != stop_power) {
        if ((k != -1 && start_power <= log2_32(k)) || (k == -1 && start_power <= k_log[1])) {
            cerr << "error: 2^startpower 必须大于 k\n";
            exit(1);
        }
        if (stop_power > max_power) {
            cerr << "error: 2^stoppower 个该类型数据，已超过 GPU 能承载的最大数量\n";
            exit(1);
        }
    } else {  // start_power == stop_power
        if ((k != -1 && start_power <= log2_32(k)) || (k == -1 && start_power <= k_log[1])) {
            cerr << "error: power=" << start_power << ", 2^power 必须大于 k\n";
            exit(1);
        } else if (start_power > max_power) {
            cerr << "error: power=" << start_power << ", 2^power 个该类型数据，已超过 GPU 能承载的最大数量\n";
            exit(1);
        }
    }

    if (start_power > stop_power) {
        if ((k != -1 && stop_power <= log2_32(k)) || (k == -1 && stop_power <= k_log[1]))
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
}

int main(int argc, char** argv) {
    int k, startPower, stopPower, testCount;
    double parameter[2];
    int k_log[2];
    DataType type;
    Distribution distribution;
    SortType sort;
    getParameters(argc, argv, k, k_log, startPower, stopPower, testCount, parameter, type, distribution, sort);

    switch (type) {
        case UINT:
            runTests<uint>((uint)k, (uint*)k_log, (uint)startPower, (uint)stopPower, (uint)testCount, parameter, type, distribution, sort);
            break;
        case ULONG:
            // 不知道为什么，类型写成 unsigned long long 就编译报错
            runTests<unsigned long>((uint)k, (uint*)k_log, (uint)startPower, (uint)stopPower, (uint)testCount, parameter, type, distribution, sort);
            break;
        case INT:
            runTests<int>((uint)k, (uint*)k_log, (uint)startPower, (uint)stopPower, (uint)testCount, parameter, type, distribution, sort);
            break;
        case LONG:
            runTests<long>((uint)k, (uint*)k_log, (uint)startPower, (uint)stopPower, (uint)testCount, parameter, type, distribution, sort);
            break;
        case FLOAT:
            runTests<float>((uint)k, (uint*)k_log, (uint)startPower, (uint)stopPower, (uint)testCount, parameter, type, distribution, sort);
            break;
        case DOUBLE:
            runTests<double>((uint)k, (uint*)k_log, (uint)startPower, (uint)stopPower, (uint)testCount, parameter, type, distribution, sort);
            break;
    }

    return 0;
}
