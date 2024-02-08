# CUDA Word Frequency Histogram Generator
This program uses CUDA api to generate histogram of the most common words from the given corpora. It is designed with the following features:
- Preprocessing
- Scalable Corpora size
- Scalable Histogram size

## Preprocessing
Due to this project mainly focusing on Efficiency and Performance, this step is realized very strictly and only valid English words with a limited length are passed through.
- Only small letters are passed through and capital letters are normalized
- Space, Slash and dots are used as separators
- Words are limited to a limit of 15 characters and longer words are cut short.

For tokenizing each word is converted to a single uint_32 data type. each English character is represented using 5 bits and overall 6 characters have been stored for each word. This mean after tokenizing and detokenizing each word will be at most 6 characters long. This is done due to better parallelization in the GPU and reduction in bandwidth needed for data transfer between DRAM and VRAM. This could also be considered a crude version of stemming for English words. When considering many examples it can bee seen that words whose first 6 characters are equivalent are mostly form the same root. the only issue is with words that have prefixes. Although no English prefix is 6 characters long this reduces the valuable 6 characters that each stemmed word can contain. If prefixes are removed before stemming, the effect of this method would become mostly ideal for a wast range of common English words and only very few technical words that are inherently large would be affected by this method. These words only represent a small portion of overall English corpora and in many cases are cumulatively categorized as "other".
Due to being compute intensive, this step of preprocessing is done at the GPU kernel.

Since the main intentions of this work is to count number of appearances of each word, stop word have not been removed.

## Data Transfer
Words are first preprocessed and cut to 6 characters, placed in an array with constant length for each word. This array is linked to the device memory using ***zero-copy*** method and while input chunk is being processed by the CPU it's also being transfered to VRAM.
Also The overall Length of the array is set by input chuck size. One of the feature of this process is being scalable regarding corpora size. This is by dividing the corpora into constant sized chunks (constant number of words) and processing it one chunk at a time.
After processing each chunk of corpora it's respective histogram is returned to the host and it's accumulated with the overall histogram using CPU parallelization.

## CUDA Operation
There are two CUDA kernels implemented in this program. One kernel simply tokenizes the words and places tokens in global memory. The other kernel counts the number of each token. Each kernel is described in detail below:

### Tokenize Kernel:
As describe earlier this kernel reads buffer of 6 characters from input char-array and turns it into a uint_32 and saves it inside the global memory of the GPU.
This might seem trivial but this is kernel has a similar run time to the second kernel due to large amounts of data being transfered.

### Histogram Kernel:
This kernel holds two arrays in device shared memory
- Array of valid tokens
- Number of each valid token

Due to being used repeatedly these arrays need to reside on the chip therefor they have to be of a limited size. Therefor calculating the histogram for each chunk of corpora requires multiple thread blocks. This is also what makes the program scalable regarding histogram size. Increasing histogram size only requires more blocks to be calculated.
For each token the list of valid tokens is searched using binary search and if a valid token is found, it's index is used to atomically increase the number of that token in the second array.
These two arrays have the same length and because both of them are from uint_32 type, they're also the same size. Their size should be calculated based on SMEM available in each GPU SM unit. Due to being tested on Pascal Generation of GPUs which have maximum of 48KB of SMEM available for each block this value is set to 5000 which results in 40KB of SMEM being used.

### Effective Scheduling:
Even though it has been made efficient data transfer is still a very time consuming process in this program, therefor it is critical to interleave these two kernels and make sure they are running at the same time on different chunks of data.
This is done using kernel streaming. It is scheduled so that while the Histogram Kernel for one chunk of data is running, Tokenize Kernel for the next chunk would be running as well. This makes sure CUDA-cores are not being stopped due to zero-copy method of data transfer and Histogram Kernel operations are always ready to be processed.
Processes done in each stream would look like the following:
```
T stands for "Tokenize Kernel" and H stands for "Histogram Kernel".
Each number after T and H that also indicates the chunk of data that is being processed.

Stream 1: T0H0T2H2T4H4
Stream 2:   T1H1T3H3T5H5
```

Kernels are designed so that one Tokenize Kernel and one Histogram Kernel can run at the same time in one SM. Therefor the displayed schedule is only for a single SM and two streams will be needed for each SM which should be automatically created based on GPU type.
