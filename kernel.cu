
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <omp.h>

#include <iostream>
#include <stdio.h>
#include <string>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <unordered_map>

#define MIN_WORD_LEN 2
#define MAX_WORD_LEN 15
#define WORD_BUFFER_LEN 16

#define HISTOGRAM_SIZE 10000

#define MAX_BLOCKS_PER_KC 10 // Max Blocks per Kernel Call
#define BLOCK_HIST_LEN 5000
//#define BLOCK_INP_LEN 1048576
//#define BLOCK_INP_LEN 524288
#define BLOCK_INP_LEN 262144
//#define BLOCK_INP_LEN 131072
//#define BLOCK_INP_LEN 65536


#define SM_OVERAL_THREADS 1024

//#define CPU_RUN
//#define GEN_FILES

using namespace std;

unsigned int tokenize(char* str, int len);
void detokenize(char* str, int token);
bool sortPairBySecondDesc(const pair<unsigned int, unsigned int>& a, const pair<unsigned int, unsigned int>& b);

bool sortPairBySecondAsc(const pair<int, unsigned int>& a, const pair<int, unsigned int>& b)
{
    return (a.second < b.second);
}

int ppLineWord(unsigned int &wordCount, string iPath = "", string oPath = "");
int ppTokenize(pair<string, unsigned int>* const arr, unsigned int arrLen, unsigned int& tokenCount, string iPath = "", string oPath = "");
int ppUniqueTokens(pair<string, unsigned int>* arr, unsigned int arrLen,
    pair<string, unsigned int>* &uniqueArr, pair<unsigned int, unsigned int>*& histogram, unsigned int& uniqueArrLen, string hPath = "");
int ppSeperatefirstKElements(const int K, pair<unsigned int, unsigned int>* histogram, unsigned int histogramLen,
    unsigned int*& keywords, unsigned int& keywordCount, string oPath = "", string hPath = "");

int ompHistogram(unsigned int histLen, unsigned int*& h_histogramTokens, unsigned int*& h_histogram, string wordsPath = "", string keywordsPath = "");
cudaError_t cudaHistogram(unsigned int histLen, unsigned int*& h_histogramWords, unsigned int*& h_histogram, string wordsPath = "", string keywordsPath = "");
cudaError_t cudaHistogram1(unsigned int histLen, unsigned int*& h_histogramTokens, unsigned int*& h_histogram, string wordsPath = "", string keywordsPath = "");
__global__ void tokenizeKernel(char* d_words, unsigned int* d_tokens, int wordCount, int blocksPerHist);
__global__ void histogramKernel(unsigned int* d_tokens, int tokenCount, unsigned int* d_histogramTokens, int histogramLen, unsigned int* d_partialHists, int blocksPerHist);

void omp_check();

int main()
{
    omp_check();

    //unsigned int* omp_histogram;
    //unsigned int* omp_histogramWords;
    //ompHistogram(HISTOGRAM_SIZE, omp_histogram, omp_histogramWords);

    //delete[] omp_histogram;
    //delete[] omp_histogramWords;
    //return 0;

#ifdef CPU_RUN
    int res;
    unsigned int wordCount = 0, tokenCount = 0;
    double startTime, endTime, extraTimeBeg, extraTimeEnd;
    res = ppLineWord(wordCount);
    startTime = omp_get_wtime();
    pair<string, unsigned int>* tokenizedWords = new pair<string, unsigned int>[wordCount];
    res = ppTokenize(tokenizedWords, wordCount, tokenCount);
    sort(tokenizedWords, tokenizedWords + wordCount);

    pair<string, unsigned int>* uniqueTokens;
    pair<unsigned int, unsigned int>* histogram;
    unsigned int uniqueTokenCount;
    res = ppUniqueTokens(tokenizedWords, tokenCount, uniqueTokens, histogram, uniqueTokenCount);
    endTime = omp_get_wtime();

    unsigned int* keywords;
    unsigned int keywordCount;
    res = ppSeperatefirstKElements(HISTOGRAM_SIZE, histogram, uniqueTokenCount, keywords, keywordCount);
    

    printf("Sizes: %d -> %d -> %d -> %d\nCPU Run Time: %f(ms)\n", wordCount, tokenCount, uniqueTokenCount, keywordCount, (endTime - startTime) * 1000);

    delete[] tokenizedWords;
    delete[] uniqueTokens;
    delete[] histogram;
    delete[] keywords;
#endif // CPU_RUN

    //ifstream kwfs;
    //char filePath[48];
    //char wordBuffer[7];
    //unsigned int* histogramTokens = new unsigned int[HISTOGRAM_SIZE];
    //sprintf(filePath, "./files/keyWords%d.txt", HISTOGRAM_SIZE);
    //kwfs.open(filePath, ios::binary);
    //if (!kwfs) {
    //    printf("cudaHistogram Couldn't open file\n");
    //    return cudaErrorFileNotFound;
    //}
    //kwfs.read((char*)histogramTokens, 10 * sizeof(int));
    //if (kwfs.bad()) {
    //    printf("cudaHistogram Couldn't read Histogram Words Correctly\n");
    //    return cudaErrorFileNotFound;
    //}
    //for (int i = 0; i < 10; i++) {
    //    detokenize(wordBuffer, histogramTokens[i]);
    //    printf("main: %s\n", wordBuffer);
    //}
    //kwfs.close();

    //printf("6\n");


    unsigned int* cuda_histogram;
    unsigned int* cuda_histogramWords;
    cudaError_t error = cudaHistogram1(HISTOGRAM_SIZE, cuda_histogramWords, cuda_histogram);
    char str[WORD_BUFFER_LEN];
    for (int i = 0; i < 15; i++) {
        detokenize(str, cuda_histogramWords[i]);
        printf("%s\t%d\n", str, cuda_histogram[i]);
    }

    cudaFreeHost(cuda_histogram);
    cudaFreeHost(cuda_histogramWords);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

unsigned int tokenize(char* str, int len) {
    unsigned int token = 0;
    unsigned char delta = 'a' - 1;
    if (len > 6)
        len = 6;
    for (int i = 0; i < 6; i++) {
        token <<= 5;
        if (i < len)
            token += str[i] - delta;
    }
    return token;
}

void detokenize(char* str, int token) {
    unsigned char delta = 'a' - 1;
    str[5] = token % 32 + delta;
    token >>= 5;
    str[4] = token % 32 + delta;
    token >>= 5;
    str[3] = token % 32 + delta;
    token >>= 5;
    str[2] = token % 32 + delta;
    token >>= 5;
    str[1] = token % 32 + delta;
    token >>= 5;
    str[0] = token % 32 + delta;
    for (int i = 0; i < 6; i++)
        if (str[i] == delta) {
            str[i] = 0;
            break;
        }
    str[6] = 0;
}

bool sortPairBySecondDesc(const pair<unsigned int, unsigned int>& a, const pair<unsigned int, unsigned int>& b)
{
    return (a.second > b.second);
}


int ppLineWord(unsigned int& wordCount, string iPath, string oPath) {
    ifstream ifs;
    if (iPath == "") {
        ifs.open("./files/input.txt");
    }
    else {
        ifs.open(iPath);
    }
    if (!ifs) {
        printf("ppLineWord Couldn't open file\n");
        return -1;
    }

    ofstream ofs;
    if (oPath == "") {
        ofs.open("./files/words.txt");
    }
    else {
        ofs.open(oPath);
    }
    if (!ofs) {
        printf("ppLineWord Couldn't create file\n");
        ifs.close();
        return -1;
    }

    int bufferLen = 8 * 1024 * 1024;
    char* buffer = new char[bufferLen];
    int outputLen = 12 * 1024 * 1024;
    char* outputBuffer = new char[outputLen];
    int capDelta = 'a' - 'A';
    unsigned int wordLen = 0;
    unsigned int lineIdx = 0;
    unsigned int wordLengths[MAX_WORD_LEN] = { 0 };
    while (ifs) {
        ifs.getline(buffer, bufferLen);
        //printf("Line %d\n", lineIdx++);
        int idx = 0;
        int outIdx = 0;
        while (buffer[idx]) {
            if (wordLen >= MAX_WORD_LEN) {
                wordLengths[MAX_WORD_LEN-1]++;
                outputBuffer[outIdx++] = '\n';
                wordLen = 0;
                while (buffer[idx] != ' ' && buffer[idx] != '/' && buffer[idx] != '.')
                    idx++;
            }else if (buffer[idx] >= 'a' && buffer[idx] <= 'z') {
                outputBuffer[outIdx++] = buffer[idx];
                wordLen++;
            } else if (buffer[idx] >= 'A' && buffer[idx] <= 'Z') {
                outputBuffer[outIdx++] = buffer[idx] + capDelta;
                wordLen++;
            } else if (buffer[idx] == ' ' || buffer[idx] == '/' || buffer[idx] == '.') {
                if (wordLen < MIN_WORD_LEN) {
                    outIdx -= wordLen;
                    wordLen = 0;
                } else {
                    wordLengths[wordLen-1]++;
                    outputBuffer[outIdx++] = '\n';
                    wordLen = 0;
                }
            }

            idx++;
        }
        if (wordLen > 0) {
            if (wordLen < MIN_WORD_LEN)
                outIdx -= wordLen;
            else {
                wordLengths[wordLen-1]++;
                outputBuffer[outIdx++] = '\n';
            }
            wordLen = 0;
        }
        ofs.write(outputBuffer, outIdx);
    }

    printf("ifs: %d, %d, %d, %d\n", ifs.good(), ifs.eof(), ifs.fail(), ifs.bad());
    printf("ofs: %d, %d, %d, %d\n", ofs.good(), ofs.eof(), ofs.fail(), ofs.bad());

    unsigned int totalWordCount = 0;
    for (int i = 0; i < MAX_WORD_LEN; i++) {
        totalWordCount += wordLengths[i];
        printf("%d\t\twords with length %d\n", wordLengths[i], i+1);
    }
    printf("Total Word Count: %d\n", totalWordCount);
    wordCount = totalWordCount;
    delete[] buffer;
    delete[] outputBuffer;
    ifs.close();
    ofs.close();
    return totalWordCount;
}

int ppTokenize(pair<string, unsigned int>* const arr, unsigned int arrLen, unsigned int& tokenCount, string iPath, string oPath) {
    if (arr == NULL)
        return -1;

    ifstream ifs;
    if (iPath == "") {
        ifs.open("./files/words.txt");
    }
    else {
        ifs.open(iPath);
    }
    if (!ifs) {
        printf("ppTokenize Couldn't open file\n");
        return -1;
    }

#ifdef GEN_FILES
    ofstream ofs;
    if (oPath == "") {
        ofs.open("./files/tokenizedWords.txt", ios::binary);
    }
    else {
        ofs.open(oPath, ios::binary);
    }
    if (!ofs) {
        printf("ppTokenize Couldn't create file\n");
        ifs.close();
        return -1;
    }
#endif // GEN_FILES
    

    char* buffer = new char[WORD_BUFFER_LEN];
    unsigned int bufferLen, len;
    unsigned int token = 0;
    unsigned char delta = 'a' - 1;
    unsigned int tokenIdx = 0;
    while (ifs) {
        ifs.getline(buffer, WORD_BUFFER_LEN);
        bufferLen = strlen(buffer);
        unsigned int token = 0;
        unsigned char delta = 'a' - 1;
        if (bufferLen > 6)
            len = 6;
        else
            len = bufferLen;
        token = 0;
        for (int i = 0; i < 6; i++) {
            token <<= 5;
            if (i < len)
                token += buffer[i] - delta;
        }
        arr[tokenIdx++] = make_pair(string(buffer), token);
#ifdef GEN_FILES
        ofs.write((char*)&token, sizeof(int));
        ofs.write(buffer, WORD_BUFFER_LEN);
#endif // GEN_FILES
        if (tokenIdx >= arrLen)
            break;
    }
    tokenCount = tokenIdx;
    delete[] buffer;
    ifs.close();
#ifdef GEN_FILES
    ofs.close();
#endif // GEN_FILES
    return tokenIdx;
}

int ppUniqueTokens(pair<string, unsigned int>* arr, unsigned int arrLen,
    pair<string, unsigned int>*& uniqueArr, pair<unsigned int, unsigned int>*& histogram, unsigned int& uniqueArrLen, string hPath) {
    unsigned int lastUniqueIdx = 0;
    unsigned int uniqueTokenCount = 1;
    for (int i = 1; i < arrLen; i++) {
        if (arr[lastUniqueIdx].second != arr[i].second) {
            lastUniqueIdx = i;
            uniqueTokenCount++;
        }
    }
    uniqueArrLen = uniqueTokenCount;
    uniqueArr = new pair<string, unsigned int>[uniqueTokenCount];
    histogram = new pair<unsigned int, unsigned int>[uniqueTokenCount];
    uniqueArr[0] = arr[0];
    unsigned int begIdx = 0;
    lastUniqueIdx = 0;
    for (int i = 1; i < arrLen && lastUniqueIdx < uniqueTokenCount; i++) {
        if (uniqueArr[lastUniqueIdx].second != arr[i].second) {
            histogram[lastUniqueIdx] = make_pair(uniqueArr[lastUniqueIdx].second, i - begIdx);
            lastUniqueIdx++;
            uniqueArr[lastUniqueIdx] = make_pair(arr[i].first, arr[i].second);
            begIdx = i;
        }
    }

#ifdef GEN_FILES
    ofstream ofs;
    if (hPath == "") {
        ofs.open("./files/histogram.txt");
    }
    else {
        ofs.open(hPath);
    }
    if (!ofs) {
        printf("ppUniqueTokens Couldn't create histogram file\n");
        return -1;
    }
    for (int i = 0; i < uniqueTokenCount; i++) {
        ofs << uniqueArr[i].first << "\t" << histogram[i].first << "\t" << histogram[i].second << endl;
    }
    ofs.close();
#endif // GEN_FILES
    return uniqueTokenCount;
}

int ppSeperatefirstKElements(const int K, pair<unsigned int, unsigned int>* histogram, unsigned int histogramLen,
    unsigned int*& keywords, unsigned int& keywordCount, string oPath, string hPath) {
    int len = K;
    if (K < histogramLen) {
        sort(histogram, histogram + histogramLen, sortPairBySecondDesc);
        sort(histogram, histogram + K);
    }
    else {
        len = histogramLen;
    }
    keywordCount = len;
    keywords = new unsigned int[len];

    for (int i = 0; i < len; i++)
        keywords[i] = histogram[i].first;

    //char str[7];
    //for (int i = 0; i < 10; i++) {
    //    detokenize(str, keywords[i]);
    //    printf("PP: %s\n", str);
    //}

#ifdef GEN_FILES
    char filePath[48];
    ofstream ofs;
    if (oPath == "") {
        sprintf(filePath, "./files/keyWords%d.txt", K);
        ofs.open(filePath, ios::binary);
    }
    else {
        ofs.open(oPath, ios::binary);
    }
    if (!ofs) {
        printf("ppSeperatefirstKElements Couldn't create file\n");
        return -1;
    }
    ofs.write((char*)keywords, len * sizeof(int));
    ofs.close();

    if (hPath == "") {
        sprintf(filePath, "./files/keyWordsHistogram%d.txt", K);
        ofs.open(filePath);
    }
    else {
        ofs.open(hPath);
    }
    if (!ofs) {
        printf("ppSeperatefirstKElements Couldn't create histogram file\n");
        return -1;
    }
    char str[7];
    for (int i = 0; i < len; i++) {
        detokenize(str, histogram[i].first);
        ofs << str << "\t" << histogram[i].first << "\t" << histogram[i].second << endl;
    }
    ofs.close();
#endif // GEN_FILES
    return len;
}

int ompHistogram(unsigned int histLen, unsigned int*& h_histogramTokens, unsigned int*& h_histogram, string wordsPath, string keywordsPath) {
    char filePath[48];
    int bufferLen, indexDelta;
    ifstream wfs, kwfs;
    
    if (wordsPath == "") {
        wfs.open("./files/words.txt");
    }
    else {
        wfs.open(wordsPath);
    }
    if (!wfs) {
        printf("cudaHistogram Couldn't open file\n");
        goto Error;
    }

    if (wordsPath == "") {
        sprintf(filePath, "./files/keyWords%d.txt", histLen);
        kwfs.open(filePath, ios::binary);
    }
    else {
        kwfs.open(wordsPath);
    }
    if (!kwfs) {
        printf("cudaHistogram Couldn't open file\n");
        goto Error;
    }

    h_histogram = new unsigned int[histLen];
    for (int i = 0; i < histLen; i++)
        h_histogram[i] = 0;

    h_histogramTokens = new unsigned int[histLen];
    kwfs.read((char*)h_histogramTokens, histLen * sizeof(int));

    int sharedIdx = 0;
    int myIdx;
    int token;
    
//#pragma omp parallel
//    {
//        unordered_map<int, int> partialHist;
//        char wordBuffer[WORD_BUFFER_LEN];
//        int token;
//        while (wfs) {
//#pragma omp critical
//            {
//                wfs.getline(wordBuffer, WORD_BUFFER_LEN);
//            }
//            token = tokenize(wordBuffer, strlen(wordBuffer));
//            partialHist[token]++;
//        }
//
//        for (auto itr : partialHist) {
//#pragma omp atomic
//            h_histogram[itr.first] += itr.second;
//        }
//    }
    wfs.close();
    kwfs.close();
    return 0;
Error:
    wfs.close();
    kwfs.close();
    return -1;
}

// Helper function for using CUDA
cudaError_t cudaHistogram(unsigned int histLen, unsigned int*& h_histogramTokens, unsigned int*& h_histogram, string wordsPath, string keywordsPath) {
    char filePath[48];
    char wordBuffer[WORD_BUFFER_LEN];
    int bufferLen, indexDelta;
    ifstream wfs, kwfs;
    cudaEvent_t startTime, endTime;

    cudaError_t cudaStatus;
    // Set Flag for zero-copy memory
    cudaStatus = cudaSetDeviceFlags(cudaDeviceMapHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to Set DeviceMapHost Flag (error code %s)!\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate CUDA events that we'll use for kernel timing
    cudaStatus = cudaEventCreate(&startTime);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to create kernel_start event (error code %s)!\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaEventCreate(&endTime);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to create kernel_stop event (error code %s)!\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Record the kernel_start event
    cudaStatus = cudaEventRecord(startTime, NULL);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to record kernel_start event (error code %s)!\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    if (wordsPath == "") {
        wfs.open("./files/words.txt");
    }
    else {
        wfs.open(wordsPath);
    }
    if (!wfs) {
        printf("cudaHistogram Couldn't open file\n");
        goto Error;
    }

    if (wordsPath == "") {
        sprintf(filePath, "./files/keyWords%d.txt", histLen);
        kwfs.open(filePath, ios::binary);
    }
    else {
        kwfs.open(wordsPath);
    }
    if (!kwfs) {
        printf("cudaHistogram Couldn't open file\n");
        wfs.close();
        goto Error;
    }

    // Allocate Host memory for final histogram
    cudaStatus = cudaHostAlloc(&h_histogram, histLen * sizeof(int), 0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to Allocate Pinned Memory (error code %s)!\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    for (int i = 0; i < histLen; i++)
        h_histogram[i] = 0;

    // Allocate Host memory for histogram words
    cudaStatus = cudaHostAlloc(&h_histogramTokens, histLen * sizeof(int), 0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to Allocate Pinned Memory (error code %s)!\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    unsigned int* d_histogramTokens, * h_partialHists, * d_partialHists, * d_tokens;
    char* h_words, * d_words;
    
    // Allocate Mapped memory for tokenizing words
    cudaStatus = cudaHostAlloc(&h_words, MAX_BLOCKS_PER_KC * BLOCK_INP_LEN * 6 * sizeof(char), cudaHostAllocMapped);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to Allocate Pinned Memory (error code %s)!\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaHostGetDevicePointer(&d_words, h_words, 0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to Map Array to Device Memory (error code %s)!\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Allocate Mapped memory for keeping per-block histogram
    cudaStatus = cudaHostAlloc(&h_partialHists, HISTOGRAM_SIZE * MAX_BLOCKS_PER_KC * sizeof(int), cudaHostAllocMapped);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to Allocate Pinned Memory (error code %s)!\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaHostGetDevicePointer(&d_partialHists, h_partialHists, 0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to Map Array to Device Memory (error code %s)!\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Allocate Device memory for keeping of histogram words
    cudaStatus = cudaMalloc(&d_histogramTokens, HISTOGRAM_SIZE * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to Allocate Device Memory (error code %s)!\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Allocate Device memory for keeping of intermediate tokens
    cudaStatus = cudaMalloc(&d_tokens, MAX_BLOCKS_PER_KC * BLOCK_INP_LEN * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to Allocate Device Memory (error code %s)!\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    kwfs.read((char*)h_histogramTokens, histLen * sizeof(int));
    if (kwfs.bad()) {
        printf("cudaHistogram Couldn't read Histogram Words Correctly\n");
        goto Error;
    }

    // Copy histogram words to Device
    cudaStatus = cudaMemcpy(d_histogramTokens, h_histogramTokens, HISTOGRAM_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    
    int kernelWordCount, totalBlockCount, histCount;
    int blocksPerHist = (histLen - 1) / BLOCK_HIST_LEN + 1;
    int blockDim = SM_OVERAL_THREADS / blocksPerHist;
    while (wfs) {
        //printf("7: %d\n", blocksPerHist);
        for (kernelWordCount = 0; kernelWordCount < MAX_BLOCKS_PER_KC * BLOCK_INP_LEN; kernelWordCount++) {
            wfs.getline(wordBuffer, WORD_BUFFER_LEN);
            if (!wfs)
                break;
            bufferLen = strlen(wordBuffer);
            indexDelta = kernelWordCount * 6;
            for (int j = 0; j < 6; j++) {
                if (j < bufferLen)
                    h_words[indexDelta + j] = wordBuffer[j];
                else
                    h_words[indexDelta + j] = 0;
            }
        }
        histCount = (kernelWordCount - 1) / BLOCK_INP_LEN + 1;
        totalBlockCount = blocksPerHist * histCount;
        //printf("8: %d, %d\n", blockDim, totalBlockCount);
        tokenizeKernel<<<totalBlockCount, blockDim>>>(d_words, d_tokens, kernelWordCount, blocksPerHist);
        histogramKernel<<<totalBlockCount, blockDim>>>(d_tokens, kernelWordCount, d_histogramTokens, histLen, d_partialHists, blocksPerHist);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }
        
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            goto Error;
        }

        for (int i = 0; i < histCount; i++) {
            int histBeg = histLen * i;
            for (int j = 0; j < histLen; j++) {
                h_histogram[j] += h_partialHists[histBeg + j];
            }
        }
    }

    // Record the stop event
    cudaStatus = cudaEventRecord(endTime, NULL);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Wait for the stop event to complete
    cudaStatus = cudaEventSynchronize(endTime);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    float total_time = 0.0f;
    cudaStatus = cudaEventElapsedTime(&total_time, startTime, endTime);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to get total time elapsed between events (error code %s)!\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    printf("CUDA Total Elapsed time = %f(ms)\n", total_time);

Error:
    cudaEventDestroy(startTime);
    cudaEventDestroy(endTime);
    cudaFreeHost(h_words);
    cudaFreeHost(h_partialHists);
    cudaFree(d_histogramTokens);
    cudaFree(d_tokens);
    wfs.close();
    kwfs.close();
    return cudaStatus;
}

cudaError_t cudaHistogram1(unsigned int histLen, unsigned int*& h_histogramTokens, unsigned int*& h_histogram, string wordsPath, string keywordsPath) {
    char filePath[48];
    char wordBuffer[WORD_BUFFER_LEN];
    int bufferLen, indexDelta;
    ifstream wfs, kwfs;
    cudaEvent_t startTime, endTime;
    unsigned int* d_histogramTokens;
    unsigned int* d_tokens[4];
    unsigned int* h_partialHists[2], * d_partialHists[2];
    char* h_words[2], * d_words[2];
    cudaStream_t tokenizingStream[2], histogramStream[2];

    cudaError_t cudaStatus;
    // Set Flag for zero-copy memory
    cudaStatus = cudaSetDeviceFlags(cudaDeviceMapHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to Set DeviceMapHost Flag (error code %s)!\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Create streams for concurrent running of kernels
    for (int i = 0; i < 2; i++) {
        cudaStatus = cudaStreamCreate(tokenizingStream + i);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaStreamCreate failed! (error code %s)!\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        cudaStatus = cudaStreamCreate(histogramStream + i);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaStreamCreate failed! (error code %s)!\n", cudaGetErrorString(cudaStatus));
            goto Error;

        }
    }
    
    // Allocate CUDA events that we'll use for kernel timing
    cudaStatus = cudaEventCreate(&startTime);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to create kernel_start event (error code %s)!\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaEventCreate(&endTime);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to create kernel_stop event (error code %s)!\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Record the kernel_start event
    cudaStatus = cudaEventRecord(startTime, NULL);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to record kernel_start event (error code %s)!\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    if (wordsPath == "") {
        wfs.open("./files/words.txt");
    }
    else {
        wfs.open(wordsPath);
    }
    if (!wfs) {
        printf("cudaHistogram Couldn't open file\n");
        goto Error;
    }

    if (wordsPath == "") {
        sprintf(filePath, "./files/keyWords%d.txt", histLen);
        kwfs.open(filePath, ios::binary);
    }
    else {
        kwfs.open(wordsPath);
    }
    if (!kwfs) {
        printf("cudaHistogram Couldn't open file\n");
        wfs.close();
        goto Error;
    }

    // Allocate Host memory for final histogram
    cudaStatus = cudaHostAlloc(&h_histogram, histLen * sizeof(int), 0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to Allocate Pinned Memory (error code %s)!\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    for (int i = 0; i < histLen; i++)
        h_histogram[i] = 0;

    // Allocate Host memory for histogram words
    cudaStatus = cudaHostAlloc(&h_histogramTokens, histLen * sizeof(int), 0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to Allocate Pinned Memory (error code %s)!\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Allocate Mapped memory
    for (int i = 0; i < 2; i++) {
        // Allocate Mapped memory for tokenizing words
        cudaStatus = cudaHostAlloc(h_words + i, MAX_BLOCKS_PER_KC * BLOCK_INP_LEN * 6 * sizeof(char), cudaHostAllocMapped);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "Failed to Allocate Pinned Memory (error code %s)!\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        cudaStatus = cudaHostGetDevicePointer(d_words + i, h_words[i], 0);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "Failed to Map Array to Device Memory (error code %s)!\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // Allocate Mapped memory for keeping per-block histogram
        cudaStatus = cudaHostAlloc(h_partialHists + i, HISTOGRAM_SIZE * MAX_BLOCKS_PER_KC * sizeof(int), cudaHostAllocMapped);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "Failed to Allocate Pinned Memory (error code %s)!\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        cudaStatus = cudaHostGetDevicePointer(d_partialHists + i, h_partialHists[i], 0);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "Failed to Map Array to Device Memory (error code %s)!\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

    }

    // Allocate Device memory for keeping of histogram words
    cudaStatus = cudaMalloc(&d_histogramTokens, HISTOGRAM_SIZE * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to Allocate Device Memory (error code %s)!\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    
    // Allocate Device memory for keeping of intermediate tokens
    for (int i = 0; i < 4; i++) {
        cudaStatus = cudaMalloc(d_tokens + i, MAX_BLOCKS_PER_KC * BLOCK_INP_LEN * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to Allocate Device Memory (error code %s)!\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        cudaStatus = cudaMalloc(d_tokens + i, MAX_BLOCKS_PER_KC * BLOCK_INP_LEN * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to Allocate Device Memory (error code %s)!\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }
    }

    kwfs.read((char*)h_histogramTokens, histLen * sizeof(int));
    if (kwfs.bad()) {
        printf("cudaHistogram Couldn't read Histogram Words Correctly\n");
        goto Error;
    }

    // Copy histogram words to Device
    cudaStatus = cudaMemcpyAsync(d_histogramTokens, h_histogramTokens, HISTOGRAM_SIZE * sizeof(int), cudaMemcpyHostToDevice, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    int kernelWordCount, prevKWC, totalBlockCount, prevTBC, histCount, prevHC[3];
    int blocksPerHist = (histLen - 1) / BLOCK_HIST_LEN + 1;
    int blockDim = SM_OVERAL_THREADS / blocksPerHist;
    int idx = 0;
    int state = 0;
    int prevState = 4;
    int nextState = 1;
    bool binState = false;
    int extra = 0;
    while (extra < 3) {
        if (wfs) {
            for (kernelWordCount = 0; kernelWordCount < MAX_BLOCKS_PER_KC * BLOCK_INP_LEN; kernelWordCount++) {
                wfs.getline(wordBuffer, WORD_BUFFER_LEN);
                if (!wfs)
                    break;
                bufferLen = strlen(wordBuffer);
                indexDelta = kernelWordCount * 6;
                for (int j = 0; j < 6; j++) {
                    if (j < bufferLen)
                        h_words[binState][indexDelta + j] = wordBuffer[j];
                    else
                        h_words[binState][indexDelta + j] = 0;
                }
            }
            histCount = (kernelWordCount - 1) / BLOCK_INP_LEN + 1;
            totalBlockCount = blocksPerHist * histCount;
            tokenizeKernel<<<totalBlockCount * 0.8, 768, 0, tokenizingStream[state]>>>(d_words[binState], d_tokens[state], kernelWordCount, blocksPerHist);
            
        } else {
            extra++;
        }

        if (idx > 2) {
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
                goto Error;
            }

            cudaStatus = cudaStreamSynchronize(histogramStream[!binState]);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
                goto Error;
            }
            for (int i = 0; i < prevHC[2]; i++) {
                int histBeg = histLen * i;
                for (int j = 0; j < histLen; j++) {
                    h_histogram[j] += h_partialHists[!binState][histBeg + j];
                }
            }
        }

        if (idx > 0 && extra < 2) {
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
                goto Error;
            }

            cudaStatus = cudaStreamSynchronize(tokenizingStream[!binState]);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
                goto Error;
            }
            histogramKernel<<<prevTBC, 256, 0, histogramStream[!binState] >> >(d_tokens[prevState], prevKWC, d_histogramTokens, histLen, d_partialHists[!binState], blocksPerHist);
        }

        prevKWC = kernelWordCount;
        prevHC[2] = prevHC[1];
        prevHC[1] = prevHC[0];
        prevHC[0] = histCount;
        prevTBC = totalBlockCount;
        idx++;
        prevState = state;
        state = nextState;
        nextState = (nextState + 1) % 4;
        binState = !binState;
    }

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Record the stop event
    cudaStatus = cudaEventRecord(endTime, NULL);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Wait for the stop event to complete
    cudaStatus = cudaEventSynchronize(endTime);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    float total_time = 0.0f;
    cudaStatus = cudaEventElapsedTime(&total_time, startTime, endTime);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to get total time elapsed between events (error code %s)!\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    printf("CUDA Total Elapsed time = %f(ms)\n", total_time);

Error:
    cudaEventDestroy(startTime);
    cudaEventDestroy(endTime);
    for (int i = 0; i < 1; i++) {
        cudaStreamDestroy(tokenizingStream[i]);
        cudaStreamDestroy(histogramStream[i]);
        cudaFreeHost(h_words[i]);
        cudaFreeHost(h_partialHists[i]);
    }
    cudaFree(d_histogramTokens);
    for (int i = 0; i < 4; i++)
        cudaFree(d_tokens[i]);
    wfs.close();
    kwfs.close();
    return cudaStatus;
}

__global__ void tokenizeKernel(char* d_words, unsigned int* d_tokens, int wordCount, int blocksPerHist) {
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    char wordBuffer[6];
    unsigned int wordAdr;
    unsigned int token;
    unsigned char delta = 'a' - 1;
    for (unsigned int i = threadID; i < wordCount; i += gridDim.x * blockDim.x) {
        wordAdr = 6 * i;
        token = 0;
        memcpy(wordBuffer, d_words + wordAdr, 6 * sizeof(char));
        for (int j = 0; j < 6; j++) {
            token <<= 5;
            if (wordBuffer[j] >= 'a' && wordBuffer[j] <= 'z')
                token += wordBuffer[j] - delta;
        }
        d_tokens[i] = token;
    }
}

__global__ void histogramKernel(unsigned int* d_tokens, int tokenCount, unsigned int* d_histogramTokens, int histogramLen, unsigned int* d_partialHists, int blocksPerHist) {
    __shared__ unsigned int blockHist[BLOCK_HIST_LEN];
    __shared__ unsigned int blockHistTokens[BLOCK_HIST_LEN];
    int blockID = blockIdx.x;
    int histogramCount = gridDim.x / blocksPerHist;
    int histogramId = blockID / blocksPerHist;
    int threadID1 = blockID * blockDim.x + threadIdx.x;
    int threadID = histogramId * blockDim.x + threadIdx.x;
    int histogramWordBeg = (histogramLen * (blockID % blocksPerHist)) / blocksPerHist;
    int histogramWordEnd = (histogramLen * ((blockID % blocksPerHist) + 1)) / blocksPerHist - 1; // Inclusive range
    int partialHistBeg = histogramId * histogramLen + histogramWordBeg;
    int blockHistLen = histogramWordEnd - histogramWordBeg + 1;
    for (int i = threadIdx.x; i < blockHistLen; i += blockDim.x) {
        blockHist[i] = 0;
        blockHistTokens[i] = d_histogramTokens[histogramWordBeg + i];
    }
    __syncthreads();
    unsigned int token, midValue;
    int beg, end, mid;
    for (int i = threadID; i < tokenCount; i += histogramCount * blockDim.x) {
        token = d_tokens[i];
        beg = 0;
        end = blockHistLen;
        if (token < blockHistTokens[beg] || token > blockHistTokens[end-1])
            continue;

        //if (token == 35684352) {
        //    printf("Hist: aba: %d\n", i);
        //}
        
        mid = (beg + end) / 2;
        midValue = blockHistTokens[mid];
        if (token == midValue) {
            atomicAdd(blockHist + mid, 1);
            continue;
        }

        while (mid > beg) {
            if (token > midValue) {
                beg = mid + 1;
            } else if (token < midValue) {
                end = mid;
            }
            mid = (beg + end) / 2;
            midValue = blockHistTokens[mid];
            if (token == midValue) {
                atomicAdd(blockHist + mid, 1);
                break;
            }
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < blockHistLen; i += blockDim.x) {
        d_partialHists[partialHistBeg + i] = blockHist[i];
    }
    
}

void omp_check() {
    printf("------------ Info -------------\n");
#ifdef _DEBUG
    printf("[!] Configuration: Debug.\n");
#pragma message ("Change configuration to Release for a fast execution.")
#else
    printf("[-] Configuration: Release.\n");
#endif // _DEBUG
#ifdef _M_X64
    printf("[-] Platform: x64\n");
#elif _M_IX86 
    printf("[-] Platform: x86\n");
#pragma message ("Change platform to x64 for more memory.")
#endif // _M_IX86 
#ifdef _OPENMP
    printf("[-] OpenMP is on.\n");
    printf("[-] OpenMP version: %d\n", _OPENMP);
#else
    printf("[!] OpenMP is off.\n");
    printf("[#] Enable OpenMP.\n");
#endif // _OPENMP
    printf("[-] Maximum threads: %d\n", omp_get_max_threads());
    printf("[-] Nested Parallelism: %s\n", omp_get_nested() ? "On" : "Off");
#pragma message("Enable nested parallelism if you wish to have parallel region within parallel region.")
    printf("===============================\n");
}
