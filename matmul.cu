#include <omp.h>
#include <immintrin.h>
#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cstdint>
#include <typeinfo>
#include <chrono>
#include <cstdlib>
#include <sstream>
#include <cuda.h>
#include <algorithm>
#include <array>
#include <mutex>
#include <atomic>
#include <condition_variable>

using std::cout;
using std::cerr;
using std::endl;
using std::vector;
using std::chrono::nanoseconds;
using std::string;

// some common definitions
typedef std::chrono::time_point<std::chrono::steady_clock> TimePoint;
typedef int64_t index_t;   // 64-bit signed index type
typedef void (single_gpu_mm_fun_t)(const uint4*,  const uint4*, uint4*, uint4*, int);
typedef void (data_arrangement_fun_t)(const uint4*, uint4*, int64_t);



// helper functions for dealing with timings
static const int TIME_STACK_SIZE = 256;
static TimePoint TIME_STACK[TIME_STACK_SIZE];
static int TIME_STACK_TOP = -1;
static bool TIMINGS_ENABLED = false;
static index_t MEMOPS = 0; // counter for memops

static inline TimePoint now() {
  return std::chrono::steady_clock::now();
}

static void enableTimings() {
  TIMINGS_ENABLED = true;
}

static void disableTimings() {
  TIMINGS_ENABLED = false;
}
  
static void pushTime() {
  if (TIMINGS_ENABLED)
    TIME_STACK[++TIME_STACK_TOP] = now();
}

static TimePoint popTime() {
  if (TIMINGS_ENABLED)
    return TIME_STACK[TIME_STACK_TOP--];
  else
    return TimePoint();
}

static int64_t getDiffNanos() {
  auto end = popTime();
  auto start = popTime();
  std::chrono::nanoseconds diff = end - start;
  return diff.count();
}

static double computeEffectiveTbops(int64_t n, int64_t nanos) {
  return ((2*n*n*n-n*n)/1e12)/(nanos/1e9);
}

static std::string nanosToString(double nanos) {
  double micros = nanos/1000.0;
  double millis = micros/1000.;
  double seconds = millis/1000.;
  std::ostringstream oss;
  if (seconds > 1)
    oss << seconds << " s";
  else if (millis > 1)
    oss << millis << " ms";
  else if (micros > 1)
    oss << micros << " µs";
  else
    oss << nanos << " ns";
  return oss.str();
}

// some magick constants
const int MAX_BLOCK_SIZE = 256;
const unsigned FULL_MASK = 0xffffffff;
static const int NUMBER_OF_CORES = 40;
static const index_t GIGABYTE_WORDS = (1L<<30)/32L;
static_assert(GIGABYTE_WORDS == 33554432LL, "wrong number of words in a gigabyte");
static const index_t WORDS_PER_THREAD = GIGABYTE_WORDS/NUMBER_OF_CORES;
static_assert(WORDS_PER_THREAD == 838860, "wrong number of words per thread");

// 4-long vectors corresponding to which submatrices of A are XORed for each T
static const int STRASSEN_ALTERNATE_BASIS_T_VECTORS[7][4] = {
 // A00, A01, A10, A11
  { 1,   0,   1,   0 }, // T0
  { 0,   0,   1,   0 },
  { 1,   1,   0,   0 },
  { 1,   0,   0,   0 },
  { 1,   0,   0,   1 },
  { 0,   1,   0,   0 },
  { 0,   0,   0,   1 }  // T6
};

static const int SIX_ALTERNATE_BASIS_T_VECTORS[6][4] = {
 // A00, A01, A10, A11
  { 1,   0,   1,   0 }, // T0
  { 0,   0,   1,   0 },
  { 1,   1,   0,   0 },
  { 1,   0,   0,   0 },
  { 1,   0,   0,   1 },
  { 0,   1,   0,   0 }  // T5
};

// 4-long vectors corresponding to which submatrices of B are XORed for each S
static const int STRASSEN_ALTERNATE_BASIS_S_VECTORS[7][4] = {
 // B00, B10, B01, B11
  { 1,   1,   0,   0 }, // S0
  { 0,   0,   1,   0 },
  { 1,   0,   1,   0 },
  { 1,   0,   0,   0 },
  { 0,   1,   0,   0 },
  { 1,   0,   0,   1 },
  { 0,   0,   0,   1 }  // S6
};

static const int SIX_ALTERNATE_BASIS_S_VECTORS[6][4] = {
 // B00, B10, B01, B11
  { 1,   1,   0,   0 }, // S0
  { 0,   0,   1,   0 },
  { 1,   0,   1,   0 },
  { 1,   0,   0,   0 },
  { 0,   1,   0,   0 },
  { 1,   0,   0,   1 }  // S5
};

// 4-long vectors corresponding to which submatrices of C are XORed for each Q
static const int STRASSEN_ALTERNATE_BASIS_Q_VECTORS[7][4] = {
 // C00, C01, C10, C11
  { 1,   0,   1,   0 }, // Q0
  { 1,   0,   0,   1 },
  { 1,   1,   0,   0 },
  { 1,   0,   0,   0 },
  { 0,   1,   0,   0 },
  { 0,   0,   1,   0 },
  { 0,   0,   0,   1 }  // Q6
};

static const int SIX_ALTERNATE_BASIS_Q_VECTORS[6][4] = {
 // C00, C01, C10, C11
  { 1,   0,   1,   0 }, // Q0
  { 1,   0,   0,   1 },
  { 1,   1,   0,   0 },
  { 1,   0,   0,   0 },
  { 0,   1,   0,   0 },
  { 0,   0,   1,   0 }  // Q5
};



// helper functions for dealing with uint4s
__host__ __device__
inline static constexpr uint4 operator^(const uint4& x, const uint4& y) {
  return uint4 { x.x^y.x, x.y^y.y, x.z^y.z, x.w^y.w };
}

__host__ __device__
inline static constexpr uint4 operator|(const uint4& x, const uint4& y) {
  return uint4 { x.x|y.x, x.y|y.y, x.z|y.z, x.w|y.w };
}

__device__
inline static void shfl(uint4& local, const uint4& remote, int lane) {
  local.x = __shfl_sync(FULL_MASK, remote.x, lane);
  local.y = __shfl_sync(FULL_MASK, remote.y, lane);
  local.z = __shfl_sync(FULL_MASK, remote.z, lane);
  local.w = __shfl_sync(FULL_MASK, remote.w, lane);
}


// some cuda specific helper macros
#define CUDA_WRAP(err) (error_wrap(err,__FILE__,__LINE__))
#define CUDA_SYNC                                                     \
    cudaDeviceSynchronize();                                          \
    error_wrap(cudaGetLastError(), __FILE__, __LINE__);               \


#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#error Per thread default stream must be enabled!
#endif

static int GPU_COUNT = 1;



// miscellaneous helper functions
static uint4* allocateAlignedMemory(int64_t nWords) {
  uint4* p;
  int rv = posix_memalign(reinterpret_cast<void**>(&p), 32, sizeof(uint4)*nWords);
  assert(rv == 0 && "Memory allocation failed");
  return p;
}

static bool isPowerOfTwo(index_t x) {
  return (x & (x - 1)) == 0;
}

static constexpr index_t intpow(index_t x, index_t y) {
  return y == 0 ? 1 : x*intpow(x,y-1);
}


static constexpr index_t intlog2Recurrence(index_t x, index_t d) {
  return (1L<<d) < x ? intlog2Recurrence(x, d+1) : d;
}

// returns the least integer y such that 2^y >= x
static constexpr index_t intlog2(index_t x) {
  return intlog2Recurrence(x, 0);
}


// avx helper functions
static void avxSet(uint4* A, int64_t n, uint64_t v) {
  __v4du* Av = reinterpret_cast<__v4du*>(A);
  n >>= 1;

  __v4du vv = {v,v,v,v};
  
  // perform in chunks of 1 GB, one v4du is 32 bytes so one chunk is
  // 33554432 v4du's
  static_assert(NUMBER_OF_CORES == 40, "wrong number of cores");
 
#pragma omp parallel num_threads(NUMBER_OF_CORES)
  {
    for (int64_t j = 0; j < n; j += GIGABYTE_WORDS) {
#pragma omp for
      for (int64_t t = 0; t < NUMBER_OF_CORES; ++t) { // one iteration per thread
        int64_t firstWord = j + WORDS_PER_THREAD*t; // inclusive
        int64_t lastWord = j + WORDS_PER_THREAD*(t+1); // exclusive
        // the last thread takes care of the modulo members
        if (omp_get_thread_num() == NUMBER_OF_CORES-1)
          lastWord += GIGABYTE_WORDS % NUMBER_OF_CORES;
        lastWord = std::min(lastWord,n);
        for (int64_t i = firstWord; i < lastWord; i += 2) {
          Av[i]   = vv;
          Av[i+1] = vv;
        }
      }
    }
  }
}


// n is here the number of 128 bit words
// we'll operate on 512 bits (64 bytes) at a time
// sizeof(__v4di) == 32, so we need two avx2 operations
static void avxXor(const uint4* A, const uint4* B, uint4* C, int64_t n) {
  const __v4du* Av = reinterpret_cast<const __v4du*>(A);
  const __v4du* Bv = reinterpret_cast<const __v4du*>(B);
  __v4du* Cv = reinterpret_cast<__v4du*>(C);
  n >>= 1;
  
  // perform in chunks of 1 GB, one v4du is 32 bytes so one chunk is
  // 33554432 v4du's

  static_assert(NUMBER_OF_CORES == 40, "wrong number of cores");

#pragma omp parallel num_threads(NUMBER_OF_CORES)
  {
    for (int64_t j = 0; j < n; j += GIGABYTE_WORDS) {
#pragma omp for
      for (int64_t t = 0; t < NUMBER_OF_CORES; ++t) { // one iteration per thread
        int64_t firstWord = j + WORDS_PER_THREAD*t; // inclusive
        int64_t lastWord = j + WORDS_PER_THREAD*(t+1); // exclusive
        // the last thread takes care of the modulo members
        if (omp_get_thread_num() == NUMBER_OF_CORES-1)
          lastWord += GIGABYTE_WORDS % NUMBER_OF_CORES;
        lastWord = std::min(lastWord,n);
        for (int64_t i = firstWord; i < lastWord; i += 2) {
          Cv[i]   = Av[i]   ^ Bv[i];
          Cv[i+1] = Av[i+1] ^ Bv[i+1];
        }
      }
    }
  }
}



// same as above but performe 4-ways
static void avxXor4(const uint4* A, const uint4* B, const uint4* C, const uint4* D,
             uint4* E, int64_t n) {
  const __v4du* Av = reinterpret_cast<const __v4du*>(A);
  const __v4du* Bv = reinterpret_cast<const __v4du*>(B);
  const __v4du* Cv = reinterpret_cast<const __v4du*>(C);
  const __v4du* Dv = reinterpret_cast<const __v4du*>(D);
  __v4du* Ev = reinterpret_cast<__v4du*>(E);
  n >>= 1;
  
  // perform in chunks of 1 GB, one v4du is 32 bytes so one chunk is
  // 33554432 v4du's

  static_assert(NUMBER_OF_CORES == 40, "wrong number of cores");
  
#pragma omp parallel num_threads(NUMBER_OF_CORES)
  {
    for (int64_t j = 0; j < n; j += GIGABYTE_WORDS) {
#pragma omp for
      for (int64_t t = 0; t < NUMBER_OF_CORES; ++t) { // one iteration per thread
        int64_t firstWord = j + WORDS_PER_THREAD*t; // inclusive
        int64_t lastWord = j + WORDS_PER_THREAD*(t+1); // exclusive
        // the last thread takes care of the modulo members
        if (omp_get_thread_num() == NUMBER_OF_CORES-1)
          lastWord += GIGABYTE_WORDS % NUMBER_OF_CORES;
        lastWord = std::min(lastWord,n);
        for (int64_t i = firstWord; i < lastWord; i += 2) {
          Ev[i]   = Av[i]   ^ Bv[i]   ^ Cv[i]   ^ Dv[i];
          Ev[i+1] = Av[i+1] ^ Bv[i+1] ^ Cv[i+1] ^ Dv[i+1];
        }
      }
    }
  }
}



// n is here the number of 128 bit words
// we'll operate on 512 bits (64 bytes) at a time
// sizeof(__v4di) == 32, so we need two avx2 operations
void avxMov(const uint4* A, uint4* B, int64_t n) {
  const __v4du* Av = reinterpret_cast<const __v4du*>(A);
  __v4du* Bv = reinterpret_cast<__v4du*>(B);
  n >>= 1;
  
  // perform in chunks of 1 GB, one v4du is 32 bytes so one chunk is
  // 33554432 v4du's
  static_assert(NUMBER_OF_CORES == 40, "wrong number of cores");
 
#pragma omp parallel num_threads(NUMBER_OF_CORES)
  {
    for (int64_t j = 0; j < n; j += GIGABYTE_WORDS) {
#pragma omp for
      for (int64_t t = 0; t < NUMBER_OF_CORES; ++t) { // one iteration per thread
        int64_t firstWord = j + WORDS_PER_THREAD*t; // inclusive
        int64_t lastWord = j + WORDS_PER_THREAD*(t+1); // exclusive
        // the last thread takes care of the modulo members
        if (omp_get_thread_num() == NUMBER_OF_CORES-1)
          lastWord += GIGABYTE_WORDS % NUMBER_OF_CORES;
        lastWord = std::min(lastWord,n);
        for (int64_t i = firstWord; i < lastWord; i += 2) {
          Bv[i]   = Av[i];
          Bv[i+1] = Av[i+1];
        }
      }
    }
  }
}



// computes popcount
template<typename T>
inline static int popcount(T x);

template<>
inline int popcount(unsigned int x) {
  return __builtin_popcount(x);
}

template<>
inline int popcount(unsigned long long x) {
  return __builtin_popcountll(x);
}


void cpuMultiplicationKernel(const uint4* A, const uint4* B,
                             uint4* C, int blockIdx,
                             int blockDim, int threadIdx) {
  assert(threadIdx >= 0);
  int it = blockIdx*blockDim + threadIdx;
  // j = (w0,jb)
  uint4 cij = make_uint4(0,0,0,0);
  uint4 aik = A[it];
  for (int jb = 0; jb < 32; ++jb) { // 32; ++jb) {
    // cpu "synchronization"
    uint4 bjk = B[(it/32)*32 + jb];
    cij.x |= (popcount((aik.x&bjk.x)^(aik.y&bjk.y))&1) << jb;
    cij.y |= (popcount((aik.x&bjk.z)^(aik.y&bjk.w))&1) << jb;
    cij.z |= (popcount((aik.z&bjk.x)^(aik.w&bjk.y))&1) << jb;
    cij.w |= (popcount((aik.z&bjk.z)^(aik.w&bjk.w))&1) << jb;
  }
  C[it] = cij;
}



// a helper class for dealing with binary matrices
class BinaryMatrix {
public:
  BinaryMatrix(index_t n, index_t m) : n(n), m(m)  {
    assert(n % 64 == 0);
    assert(m % 64 == 0);
    size_t dataSize = n*m/128*sizeof(uint4);
    int rv = posix_memalign(reinterpret_cast<void**>(&data), 32, dataSize);
    assert(rv == 0 && "Memory allocation failed");
    static_assert(sizeof(uint4)*2 == sizeof(__v4du), "data type size assumption mismatch");
    avxSet(data, dataSize/sizeof(uint4), 0);
  }

  BinaryMatrix(const BinaryMatrix& that) :
    n(that.n), m(that.m), data(allocateAlignedMemory(n*m/128))
  {
    size_t dataSize = n*m/128*sizeof(uint4);
    memcpy(data, that.data, dataSize);
  }
  
  BinaryMatrix(BinaryMatrix&& that) : n(that.n), m(that.m), data(nullptr) {
    std::swap(data,that.data);
  }
  
  BinaryMatrix& operator=(const BinaryMatrix& that) {
    if (this != &that) {
      BinaryMatrix M(that);
      std::swap(n, M.n);
      std::swap(m, M.m);
      std::swap(data, M.data);
    }
    return *this;
  }
  
  BinaryMatrix& operator=(BinaryMatrix&& that) {
    if (this != &that) {
      std::swap(n, that.n);
      std::swap(m, that.m);
      std::swap(data, that.data);
    }
    return *this;
  }

  ~BinaryMatrix() {
    free(data);
  }

  // get by coordinate
  int get(index_t i, index_t j) const {
    index_t k = bitNumber(i,j);
    return (wordForBit(k) >> (k%32)) & 1;
  }

  // get by raw bit
  int get(int64_t bp) const {
    const uint32_t& w = wordForBit(bp);
    return (w >> (bp%32)) & 1;
  }

  void set(index_t i, index_t j, int v) {
    index_t k = bitNumber(i,j);
    if (v)
      wordForBit(k) |= 1<<(k%32);
    else
      wordForBit(k) &= ~(1<<(k%32));
  }

  // generate such a matrix that if two of such matrices were to be
  // multiplied assuming and/or semantics, the resulting matrix would
  // have approximately one half of its entries nonzero
  static BinaryMatrix booleanRandom(index_t nDim) {
    BinaryMatrix M(nDim,nDim);
    assert(typeid(M.data[0].x) == typeid(uint32_t));
    static int seedCtr = 0;
    if (seedCtr == 0)
      seedCtr = time(nullptr);

    index_t d = 0;
    while ((1LL<<d)*64LL < nDim)
      ++d;
    assert((1LL<<d)*64LL == nDim);
    
    // value at index i corresponds to nDim == 2^i * 64
    static const int64_t numNonzerosPerRow[] = {
      7, 10, 14, 19, 27, 38, 54, 76, 107, 151, 214, 302, 427, 603, 853
    };
   
#pragma omp parallel
    {
#pragma omp critical
      ++seedCtr;
      
      std::mt19937 rng(seedCtr);
      std::uniform_int_distribution<index_t> dist(0,nDim-1);
#pragma omp for
      for (index_t i = 0; i < nDim; ++i) {
        for (int r = 0; r < numNonzerosPerRow[d]; ++r) {
          index_t j = dist(rng);
          while (M.get(i,j)) {
            j = dist(rng);
          }
          M.set(i,j,1);
        }
      }
    }

    return M;
  }
  
  static BinaryMatrix random(int64_t nDim, int64_t mDim = -1, double zeroWordChance = 0.0) {
    if (mDim == -1)
      mDim = nDim;
    BinaryMatrix M(nDim,mDim);
    assert(typeid(M.data[0].x) == typeid(uint32_t));
    static int seedCtr = 0;
    if (seedCtr == 0)
      seedCtr = time(nullptr);
    index_t nWords = nDim*mDim/128;
    
#pragma omp parallel
    {
#pragma omp critical
      ++seedCtr;
      
      std::mt19937 rng(seedCtr);
      std::uniform_int_distribution<uint32_t> dist(0,UINT32_MAX);
      std::uniform_real_distribution<double> zeroD(0,1);
#pragma omp for
      for (index_t i = 0; i < nWords; ++i) 
        M.data[i] = make_uint4(zeroD(rng) > zeroWordChance ? dist(rng) : 0,
                               zeroD(rng) > zeroWordChance ? dist(rng) : 0,
                               zeroD(rng) > zeroWordChance ? dist(rng) : 0,
                               zeroD(rng) > zeroWordChance ? dist(rng) : 0);
    }

    return M;
  }

  void print(std::ostream& os = std::cout) {
    for (int i = 0; i < n; ++i) {
      os << "[";
      for (int j = 0; j < m; ++j) 
        os << get(i,j);
      os << "]" << endl;
    }
  }

  void printMatlab(std::ostream& os = std::cout) {
    os << "[";
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < m; ++j) {
        if (j != 0)
          os << ",";
        os << get(i,j);
      }
      os << ((i == (n-1)) ? "]" : ";");
    }
  }

  BinaryMatrix dot(const BinaryMatrix& that) const {
    // compute naive cubic multiplication
    // obs. that is assumed to be transposed
    assert(m == that.m);
    BinaryMatrix res(n,that.n);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < that.n; ++j) {
        int v = 0;
        for (int k = 0; k < m; ++k) {
          v ^= get(i,k)&that.get(j,k);
        }
        res.set(i,j,v);
      }
    }
    return res;
  }

  

  BinaryMatrix booleanDot(const BinaryMatrix& that) const {
    // compute naive cubic multiplication
    // obs. that is assumed to be transposed
    assert(m == that.m);
    BinaryMatrix res(n,that.n);
    for (int64_t i = 0; i < n; ++i) {
      for (int64_t j = 0; j < that.n; ++j) {
        int64_t v = 0;
        for (int k = 0; k < m; ++k) {
          v |= get(i,k)&that.get(j,k);
        }
        res.set(i,j,v);
      }
    }
    return res;
  }



  // compute ONE bit from the mm
  int dotOneBit(const BinaryMatrix& that, int64_t i, int64_t j) const {
    // compute naive cubic multiplication
    // obs. that is assumed to be transposed
    assert(m == that.m);
    assert(n == that.n);

    int v = 0;
    if (n == 64) {
      if (i % 2 == 0 && j % 2 == 0) {
        v = popcount((data[i>>1].x & that.data[j>>1].x) ^
                     (data[i>>1].y & that.data[j>>1].y)) % 2;
      }
      else if (i % 2 == 0 && j % 2 == 1) {
        v = popcount((data[i>>1].x & that.data[j>>1].z) ^
                     (data[i>>1].y & that.data[j>>1].w)) % 2;
      }
      else if (i % 2 == 1 && j % 2 == 0) {
        v = popcount((data[i>>1].z & that.data[j>>1].x) ^
                     (data[i>>1].w & that.data[j>>1].y)) % 2;
      }
      else if (i % 2 == 1 && j % 2 == 1) {
        v = popcount((data[i>>1].z & that.data[j>>1].z) ^
                     (data[i>>1].w & that.data[j>>1].w)) % 2;
      }
      else
        assert(false);
    }
    else {
      int64_t rowLength = n/128;
      assert(rowLength*128 == n);
      for (int64_t k = 0; k < rowLength; ++k) {
        v ^= (((data[i*rowLength+k].x & that.data[j*rowLength+k].x) ^
               (data[i*rowLength+k].y & that.data[j*rowLength+k].y) ^
               (data[i*rowLength+k].z & that.data[j*rowLength+k].z) ^
               (data[i*rowLength+k].w & that.data[j*rowLength+k].w)));
      }
      v = popcount(static_cast<unsigned int>(v)) % 2;
    }
    return v&1;
  }

  

  BinaryMatrix dotWords(const BinaryMatrix& that) const {
    // compute naive cubic multiplication
    // obs. that is assumed to be transposed
    assert(m == that.m);
    assert(n == that.n);
    BinaryMatrix res(n,n);
    dotWordsInternal(data, that.data, res.data, n, m);
    return res;
  }



  BinaryMatrix booleanDotWords(const BinaryMatrix& that) const {
    // compute naive cubic multiplication
    // obs. that is assumed to be transposed
    assert(m == that.m);
    assert(n == that.n);
    BinaryMatrix res(n,n);
    booleanDotWordsInternal(data, that.data, res.data, n, m);
    return res;
  }



  int booleanDotOneBit(const BinaryMatrix& that, int64_t i, int64_t j) const {
    assert(m == that.m);
    assert(n == that.n);
    int64_t v = 0;
    if (n == 64) {
      if (i % 2 == 0 && j % 2 == 0) {
        v = ((data[i>>1].x & that.data[j>>1].x) |
             (data[i>>1].y & that.data[j>>1].y)) != 0;
      }
      else if (i % 2 == 0 && j % 2 == 1) {
        v = ((data[i>>1].x & that.data[j>>1].z) |
             (data[i>>1].y & that.data[j>>1].w)) != 0;
      }
      else if (i % 2 == 1 && j % 2 == 0) {
        v = ((data[i>>1].z & that.data[j>>1].x) |
             (data[i>>1].w & that.data[j>>1].y)) != 0;
      }
      else if (i % 2 == 1 && j % 2 == 1) {
        v = ((data[i>>1].z & that.data[j>>1].z) |
             (data[i>>1].w & that.data[j>>1].w)) != 0;
      }
      else
        assert(false);
    }
    else {
      int64_t rowLength = n/128;
      assert(rowLength*128 == n);
      for (int64_t k = 0; k < rowLength; ++k) {
        v = v || (((data[i*rowLength+k].x & that.data[j*rowLength+k].x) |
                   (data[i*rowLength+k].y & that.data[j*rowLength+k].y) |
                   (data[i*rowLength+k].z & that.data[j*rowLength+k].z) |
                   (data[i*rowLength+k].w & that.data[j*rowLength+k].w)) != 0);
      }
    }
    return v&1;
  }



  BinaryMatrix transpose() const {
    BinaryMatrix res(m,n);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        res.set(j,i,get(i,j));
    return res;
  }

  inline const uint4* getDataPointer() const {
    return data;
  }
                                      
  inline uint4* getDataPointer() {
    return data;
  }

  // p[i] = the coordinate that the index i travels to
  BinaryMatrix applyBitPermutation(const vector<int>& p) const {
    assert(n*m == 1 << p.size());
    BinaryMatrix A(n,m);
    for (int i = 0; i < (1 << p.size()); ++i) {
      int j = 0;
      for (size_t k = 0; k < p.size(); ++k)
        j |= ((i >> k) & 1) << p[k];
      A.set(j/m, j%m, get(i/m,i%m));
    }
    return A;
  }

  bool operator==(const BinaryMatrix& that) const {
    if (n != that.n)
      return false;
    if (m != that.m)
      return false;
    return memcmp(data, that.data, n*m/128) == 0;
  }


  
  BinaryMatrix furHatSix(const BinaryMatrix& that) const {
    assert(n == that.n);
    assert(m == that.m);
    assert(n == m);
    assert(isPowerOfTwo(n));
    int64_t d = intlog2(n);
    assert(d >= 6);
    BinaryMatrix res(n,n);
    furHatSixInternal(data, that.data, res.data, n);
    return res;
  }
  
private:
  static void furHatSixInternal(const uint4* A, const uint4* B, uint4* C, int64_t n) {
    if (n == 64) {
      for (int threadIdx = 0; threadIdx < 32; ++threadIdx)
        cpuMultiplicationKernel(A, B, C, 0, 32, threadIdx);
    }
    else {
      int64_t d = intlog2(n) - 6;
      int64_t submatrixSize = (1L<<2*(d-1))*32;
      const uint4* A00 = A;
      const uint4* A01 = A+submatrixSize;
      const uint4* A10 = A+2*submatrixSize;
      const uint4* A11 = A+3*submatrixSize;
      const uint4* B00 = B;
      const uint4* B10 = B+submatrixSize;
      const uint4* B01 = B+2*submatrixSize;
      const uint4* B11 = B+3*submatrixSize;
      uint4* C00 = C;
      uint4* C01 = C + submatrixSize;
      uint4* C10 = C + 2*submatrixSize;
      uint4* C11 = C + 3*submatrixSize;
      uint4* mem = allocateAlignedMemory(14*submatrixSize);

      uint4* T0 = mem;
      avxXor(A10, A11, T0, submatrixSize);

      const uint4* T1 = A01;

      uint4* T2 = mem+submatrixSize;
      avxXor(A01, A11, T2, submatrixSize);

      uint4* T3 = mem+2*submatrixSize;
      avxXor(A10, T2, T3, submatrixSize);

      uint4* T4 = mem+3*submatrixSize;
      avxXor(A00, T3, T4, submatrixSize);

      const uint4* T5 = A10;

      uint4* S0 = mem+4*submatrixSize;
      avxXor(B10, B11, S0, submatrixSize);

      const uint4* S1 = B10;

      uint4* S2 = mem+5*submatrixSize;
      avxXor(B01, B11, S2, submatrixSize);

      uint4* S3 = mem+6*submatrixSize;
      avxXor(B10, S2, S3, submatrixSize);

      const uint4* S4 = B01;
      
      uint4* S5 = mem+7*submatrixSize;
      avxXor(B00, S3, S5, submatrixSize);

      uint4* Q0 = mem+8*submatrixSize;
      uint4* Q1 = mem+9*submatrixSize;
      uint4* Q2 = mem+10*submatrixSize;
      uint4* Q3 = mem+11*submatrixSize;
      uint4* Q4 = mem+12*submatrixSize;
      uint4* Q5 = mem+13*submatrixSize;

      furHatSixInternal(T0,S0,Q0,n>>1);
      furHatSixInternal(T1,S1,Q1,n>>1);
      furHatSixInternal(T2,S2,Q2,n>>1);
      furHatSixInternal(T3,S3,Q3,n>>1);
      furHatSixInternal(T4,S4,Q4,n>>1);
      furHatSixInternal(T5,S5,Q5,n>>1);

      avxMov(Q1, C00, submatrixSize);
      avxXor4(Q0, Q4, Q1, Q3, C01, submatrixSize);
      avxXor4(Q5, Q2, Q1, Q3, C10, submatrixSize);
      avxXor4(Q0, Q2, Q1, Q3, C11, submatrixSize);

      free(mem);
    }
  }


  
  inline static uint32_t& wordForBit(uint4* data, int64_t k) {
    int64_t idx = k/128;
    k -= idx*128;
    return k < 32 ? data[idx].x :
      k < 64 ? data[idx].y :
      k < 96 ? data[idx].z :
      data[idx].w;
  }


  
  static void set(uint4* data, int64_t m, int64_t i, int64_t j, int v) {
    int64_t k = bitNumber(m, i,j);
    wordForBit(data, k) |= (v&1) << (k%32);
  }


  
  static void dotWordsInternal(const uint4* data, const uint4* thatData, uint4* resData, int64_t n, int64_t m) {
    assert(isPowerOfTwo(n));
    assert(n == m);
    if (n == 64) {
      for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j) {
          int64_t v;
          if (i % 2 == 0 && j % 2 == 0) {
            v = popcount((data[i>>1].x & thatData[j>>1].x) ^
                         (data[i>>1].y & thatData[j>>1].y)) % 2;
          }
          else if (i % 2 == 0 && j % 2 == 1) {
            v = popcount((data[i>>1].x & thatData[j>>1].z) ^
                         (data[i>>1].y & thatData[j>>1].w)) % 2;
          }
          else if (i % 2 == 1 && j % 2 == 0) {
            v = popcount((data[i>>1].z & thatData[j>>1].x) ^
                         (data[i>>1].w & thatData[j>>1].y)) % 2;
          }
          else if (i % 2 == 1 && j % 2 == 1) {
            v = popcount((data[i>>1].z & thatData[j>>1].z) ^
                         (data[i>>1].w & thatData[j>>1].w)) % 2;
          }
          else
            assert(false);
          set(resData,m,i,j,v);
        }
      }
    }
    else if (n==128) {
      int64_t rowLength = n/128;
      for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j) {
          unsigned int v = 0;
          for (int64_t k = 0; k < rowLength; ++k) {
            v ^= (((data[i*rowLength+k].x & thatData[j*rowLength+k].x) ^
                   (data[i*rowLength+k].y & thatData[j*rowLength+k].y) ^
                   (data[i*rowLength+k].z & thatData[j*rowLength+k].z) ^
                   (data[i*rowLength+k].w & thatData[j*rowLength+k].w)));
          }
          set(resData,m,i,j,popcount(v) % 2);
        }
      }
    }
    else {
      index_t rowLength = n/256; // in v4dus
      const __v4du* lhv = reinterpret_cast<const __v4du*>(data);
      const __v4du* rhv = reinterpret_cast<const __v4du*>(thatData);
#pragma omp parallel for
      for (index_t i = 0; i < n; ++i) {
        for (index_t j = 0; j < n; ++j) {
          __v4du v = {0,0,0,0};
          for (int64_t k = 0; k < rowLength; ++k) 
            v ^= lhv[i*rowLength+k] & rhv[j*rowLength+k];
          set(resData,m,i,j,popcount(v[0]^v[1]^v[2]^v[3]) % 2);
        }
      }
    }
  }


  static void booleanDotWordsInternal(const uint4* data, const uint4* thatData,
                                      uint4* resData, int64_t n, int64_t m) {
    assert(isPowerOfTwo(n));
    assert(n == m);
    if (n == 64) {
      for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j) {
          int64_t v;
          if (i % 2 == 0 && j % 2 == 0) {
            v = popcount((data[i>>1].x & thatData[j>>1].x) |
                         (data[i>>1].y & thatData[j>>1].y)) > 0;
          }
          else if (i % 2 == 0 && j % 2 == 1) {
            v = popcount((data[i>>1].x & thatData[j>>1].z) |
                         (data[i>>1].y & thatData[j>>1].w)) > 0;
          }
          else if (i % 2 == 1 && j % 2 == 0) {
            v = popcount((data[i>>1].z & thatData[j>>1].x) |
                         (data[i>>1].w & thatData[j>>1].y)) > 0;
          }
          else if (i % 2 == 1 && j % 2 == 1) {
            v = popcount((data[i>>1].z & thatData[j>>1].z) |
                         (data[i>>1].w & thatData[j>>1].w)) > 0;
          }
          else
            assert(false);
          set(resData,m,i,j,v);
        }
      }
    }
    else if (n==128) {
      int64_t rowLength = n/128;
      for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j) {
          unsigned int v = 0;
          for (int64_t k = 0; k < rowLength; ++k) {
            v |= (((data[i*rowLength+k].x & thatData[j*rowLength+k].x) |
                   (data[i*rowLength+k].y & thatData[j*rowLength+k].y) |
                   (data[i*rowLength+k].z & thatData[j*rowLength+k].z) |
                   (data[i*rowLength+k].w & thatData[j*rowLength+k].w)));
          }
          set(resData,m,i,j,popcount(v) > 0);
        }
      }
    }
    else {
      index_t rowLength = n/256; // in v4dus
      const __v4du* lhv = reinterpret_cast<const __v4du*>(data);
      const __v4du* rhv = reinterpret_cast<const __v4du*>(thatData);
#pragma omp parallel for
      for (index_t i = 0; i < n; ++i) {
        for (index_t j = 0; j < n; ++j) {
          __v4du v = {0,0,0,0};
          for (int64_t k = 0; k < rowLength; ++k) 
            v |= lhv[i*rowLength+k] & rhv[j*rowLength+k];
          set(resData,m,i,j,popcount(v[0]|v[1]|v[2]|v[3]) > 0);
        }
      }
    }
  }


  
  inline index_t bitNumber(index_t i, index_t j) const {
    return bitNumber(m,i,j);
  }



  inline static index_t bitNumber(index_t m, index_t i, index_t j) {
    return i*m + j;
  }


  
  // k = bit number
  inline const uint32_t& wordForBit(index_t k) const {
    index_t idx = k/128;
    assert(idx >= 0);
    k -= idx*128;
    assert(0 <= k && k < 128);
    return k < 32 ? data[idx].x :
      k < 64 ? data[idx].y :
      k < 96 ? data[idx].z :
      data[idx].w;
  }

  // k = bit number
  inline uint32_t& wordForBit(index_t k) {
    index_t idx = k/128;
    k -= idx*128;
    assert(0 <= k && k < 128);
    return k < 32 ? data[idx].x :
      k < 64 ? data[idx].y :
      k < 96 ? data[idx].z :
      data[idx].w;
  }

  index_t n, m; // shape
  uint4* data;
};



static void error_wrap(cudaError_t err,
                       const char *fn,
                       int line) {
  if(err != cudaSuccess) {
    std::printf("CUDA error [%s, line %d]: %s\n",
                fn,
                line,
                cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

vector<uint4> generateRandomArray(int nWords) {
  vector<uint4> A(nWords);
  assert(typeid(A[0].x) == typeid(uint32_t));
  static std::mt19937 rng(time(nullptr));
  std::uniform_int_distribution<uint32_t> dist(0,UINT32_MAX);
  for (int i = 0; i < nWords; ++i) {
    A[i] = make_uint4(dist(rng),dist(rng),dist(rng),dist(rng));
  }
  return A;
}

int getArrayElem(const uint4* A, int i, int j) {
  assert(i>>1 == i/2);
  assert(i >= 0 && i < 64);
  assert(j >= 0 && j < 64);
  uint4 word = A[i>>1];
  if (i%2 == 0 && j < 32)
   return (word.x >> j) & 1;
  else if (i%2 == 0 && j >= 32)
    return ((word.y >> (j-32)) & 1);
  else if (i%2 == 1 && j < 32)
    return ((word.z >> j) & 1);
  else if (i%2 == 1 && j >= 32)
    return ((word.w >> (j-32)) & 1);
  else
    assert(false);
  return -1;
}

void setArrayElem(uint4* A, int i, int j, int v) {
  assert(v == 0 || v == 1);
  assert(i>>1 == i/2);
  assert(i >= 0 && i < 64);
  assert(j >= 0 && j < 64);
  if (i%2 == 0 && j < 32)
    A[i>>1].x |= ((v & 1) << j);
  else if (i%2 == 0 && j >= 32)
    A[i>>1].y |= ((v & 1) << (j-32));
  else if (i%2 == 1 && j < 32)
    A[i>>1].z |= ((v & 1) << j);
  else if (i%2 == 1 && j >= 32)
    A[i>>1].w |= ((v & 1) << (j-32));
  else
    assert(false);
}

// compute C = AB' for 64x64 multiplication with xor/and operations
void computeBooleanMmHost(const uint4* A, const uint4* B, uint4* C) {
  for (int i = 0; i < 64; ++i) {
    for (int j = 0; j < 64; ++j) {
      int value = 0;
      for (int k = 0; k < 64; ++k)
        value ^= getArrayElem(A,i,k)&getArrayElem(B,j,k);
      setArrayElem(C,i,j,value);
    }
  }
}



void determineBlockSizeAndNumBlocks(int totalThreads, int& numBlocks,
                                    int& blockSize) {
  assert(totalThreads % 32 == 0);
  blockSize = (totalThreads < MAX_BLOCK_SIZE) ? totalThreads : MAX_BLOCK_SIZE;
  numBlocks = 1;
  while (blockSize*numBlocks != totalThreads) {
    if (numBlocks*blockSize < totalThreads)
      ++numBlocks;
    else
      blockSize -= 32;
  }
  assert(numBlocks*blockSize == totalThreads);
}



void printArray(const uint4* A) {
  for (int i = 0; i < 64; ++i) {
    cout << ((i == 0) ? "[" : " ");
    for (int j = 0; j < 64; ++j) {
      if (j != 0)
        cout << ", ";
      cout << getArrayElem(A,i,j);
    }
    cout << ((i == 63) ? "]" : ";") << endl;
  }
}

// allocate some device (gpu) memory
// a given number of 128 bit words
uint4* gpuAllocate(size_t nWords) {
    size_t size = sizeof(uint4)*nWords;
    uint4* p;

    CUDA_WRAP(cudaMalloc(&p, size));

    return p;
}

// free the associated pointer on GPU
void gpuFree(uint4* p) {
    CUDA_WRAP(cudaFree(p));
}

// upload data to device (GPU)
static void gpuUpload(index_t nWords, const uint4* cpu_p, uint4* gpu_p) {
    CUDA_WRAP(cudaMemcpy(gpu_p, cpu_p, nWords*sizeof(uint4),
                         cudaMemcpyHostToDevice));
}

// download data from device (GPU)
static void gpuDownload(index_t nWords, const uint4* gpu_p, uint4* cpu_p) {
    CUDA_WRAP(cudaMemcpy(cpu_p, gpu_p, nWords*sizeof(uint4),
                         cudaMemcpyDeviceToHost));
}

static void gpuMemcpy(index_t nWords, const uint4* gpu_from, uint4* gpu_to) {
  CUDA_WRAP(cudaMemcpy(gpu_to, gpu_from, nWords*sizeof(uint4),
                       cudaMemcpyDeviceToDevice));
}



static bool isPermutation(const vector<int>& p) {
  vector<int> found(p.size(),0);
  for (int v : p) {
    if (v < 0)
      return false;
    if (v >= static_cast<int>(p.size()))
      return false;
    found[v] = 1;
  }
  for (int v : found)
    if (v != 1)
      return false;
  return true;
}



static int64_t applyBitPermutation(const vector<int>& p, int64_t w) {
  assert(isPermutation(p));
  int64_t new_w = 0;
  for (size_t i = 0; i < p.size(); ++i)
    new_w |= ((w >> i)&1) << p[i];
  return new_w;
}



// same as above but use word operations (basically v bits come for free)
static vector<int> constructPwords(int d) {
  vector<int> js(d);
  for (int k = 0; k < d; ++k)
    js[k] = k;
  vector<int> us(6);
  for (int k = 0; k < 6; ++k)
    us[k] = k+d;
  vector<int> is(d);
  for (int k = 0; k < d; ++k)
    is[k] = k+6+d;

  vector<int> p(6+2*d);
  for (int k = 0; k < d; ++k)
    p[js[k]] = 2*k+6;
  for (int k = 0; k < 6; ++k)
    p[us[k]] = (k+1)%6;
  for (int k = 0; k < d; ++k) 
    p[is[k]] = 2*k+7;
  return p;
}



// for a matrix of dimension 2^p*2^d*64 x 2^p*2^d*64 where 2^p form the 'block grid',
// construct the following bit permutation of coordinates:
// input: i_{q-1} ... i_0 i'_{d-1} ... i'_0 u_5 ... u_0 j_{q-1} ... j_0 j'_{d-1} ... j'_0 v_5 ... v_0
// output:
// i_{q-1} ... i_0 j_{q-1} ... j_0 i'_{d-1} ... i'_0 j'_{d-1} ... j'_0 u_4 ... u_0 u_5 v_5 ... v_0
// that is, for multi gpu, the 'blocks' are indexed by the high bits
static vector<int> constructPCubicwords(int q, int d) {
  assert(q >= 0);
  assert(d >= 0);
  vector<int> jps(d);
  for (int k = 0; k < d; ++k)
    jps[k] = k;
  vector<int> js(q);
  for (int k = 0; k < q; ++k)
    js[k] = d+k;
  vector<int> us(6);
  for (int k = 0; k < 6; ++k)
    us[k] = k+d+q;
  vector<int> ips(d);
  for (int k = 0; k < d; ++k)
    ips[k] = k+6+d+q;
  vector<int> is(q);
  for (int k = 0; k < q; ++k)
    is[k] = k+6+2*d+q;

  vector<int> p(6+2*d+2*q);
  for (int k = 0; k < 6; ++k)
    p[us[k]] = (k+1)%6;
  for (int k = 0; k < d; ++k)
    p[jps[k]] = k+6;
  for (int k = 0; k < d; ++k) 
    p[ips[k]] = k+d+6;
  for (int k = 0; k < q; ++k)
    p[js[k]] = k+6+2*d;
  for (int k = 0; k < q; ++k) 
    p[is[k]] = k+2*d+6+q;
  return p;
}



static vector<int> constructInvPCubicwords(int q, int d) {
  assert(q >= 0);
  assert(d >= 0);
  vector<int> jps(d);
  for (int k = 0; k < d; ++k)
    jps[k] = k;
  vector<int> js(q);
  for (int k = 0; k < q; ++k)
    js[k] = d+k;
  vector<int> us(6);
  for (int k = 0; k < 6; ++k)
    us[k] = k+d+q;
  vector<int> ips(d);
  for (int k = 0; k < d; ++k)
    ips[k] = k+6+d+q;
  vector<int> is(q);
  for (int k = 0; k < q; ++k)
    is[k] = k+6+2*d+q;

  
  vector<int> p(6+2*d+2*q);
  for (int k = 0; k < 6; ++k)
    p[(k+1)%6] = us[k];
  for (int k = 0; k < d; ++k)
    p[k+6] = jps[k];
  for (int k = 0; k < d; ++k)
    p[k+6+d] = ips[k];
  for (int k = 0; k < q; ++k) 
    p[k+2*d+6] = js[k];
  for (int k = 0; k < q; ++k) 
    p[k+2*d+6+q] = is[k];
  return p;
}



static vector<int> constructInvPwords(int d) {
  // construct the inverse of permutation p
  vector<int> js(d);
  for (int k = 0; k < d; ++k)
    js[k] = k;
  vector<int> us(6);
  for (int k = 0; k < 6; ++k)
    us[k] = k+d;
  vector<int> is(d);
  for (int k = 0; k < d; ++k)
    is[k] = k+6+d;

  vector<int> p(6+2*d);
  for (int k = 0; k < d; ++k)
    p[2*k+6] = js[k];
  for (int k = 0; k < 6; ++k)
    p[(k+1)%6] = us[k];
  for (int k = 0; k < d; ++k) 
    p[2*k+7] = is[k];
  return p;
}



// compute the following change of basis:
// xx[0][0] <-- x[0][1] + x[1][0] + x[1][1]
// xx[0][1] <-- x[1][0]
// xx[1][0] <-- x[0][1]
// xx[1][1] <-- x[0][0]
static void changeOfBasisStrassen(const uint4* x, uint4* xx, int64_t n) {
  assert(n >= 64);
  assert(isPowerOfTwo(n));
  int64_t d = intlog2(n) - 6;
  assert(n == (1L<<d)*64);

  if (n == 64)
    avxMov(x, xx, 32);

  for (int64_t l = d-1; l >= 0; --l) {
    int64_t logM = (2*d-2*l+3);
    int64_t M = 1L << logM;
    for (int64_t t = 0; t < (1L << 2*d+3); ++t) {
      int64_t u = t >> logM;
      int64_t v = t & ~(0xffffffffffffffff << logM);
      uint4 x00 = x[u*4*M +       v];
      uint4 x01 = x[u*4*M + M   + v];
      uint4 x10 = x[u*4*M + 2*M + v];
      uint4 x11 = x[u*4*M + 3*M + v];
      xx[u*4*M +       v] = x01 ^ x10 ^ x11;
      xx[u*4*M + M   + v] = x10;
      xx[u*4*M + 2*M + v] = x01;
      xx[u*4*M + 3*M + v] = x00;
    }
    x = xx;
  }
}



static void changeOfBasisStrassenAvxParallel(const uint4* x, uint4* xx, int64_t n) {
  assert(n >= 64);
  assert(isPowerOfTwo(n));
  int64_t d = intlog2(n) - 6;
  assert(n == (1L<<d)*64);

  if (n == 64)
    avxMov(x, xx, 32);

  for (int64_t l = d-1; l >= 0; --l) {
    int64_t logM = (2*d-2*l+3);
    // int64_t logM = (2*d-2*l+1);
    int64_t M = 1L << logM;
#pragma omp parallel for
    for (int64_t t = 0; t < (1L << 2*d+3); t += 4) {
      // for (int64_t t = 0; t < (1L << 2*d+1); ++t) {
      int64_t u = t >> logM;
      int64_t v = t & ~(0xffffffffffffffff << logM);
      __v4du x000 = *reinterpret_cast<const __v4du*>(x+u*4*M+v);
      __v4du x001 = *reinterpret_cast<const __v4du*>(x+u*4*M+v+2);
      __v4du x010 = *reinterpret_cast<const __v4du*>(x+u*4*M+M+v);
      __v4du x011 = *reinterpret_cast<const __v4du*>(x+u*4*M+M+v+2);
      __v4du x100 = *reinterpret_cast<const __v4du*>(x+u*4*M+2*M+v);
      __v4du x101 = *reinterpret_cast<const __v4du*>(x+u*4*M+2*M+v+2);
      __v4du x110 = *reinterpret_cast<const __v4du*>(x+u*4*M+3*M+v);
      __v4du x111 = *reinterpret_cast<const __v4du*>(x+u*4*M+3*M+v+2);
      *reinterpret_cast<__v4du*>(xx + u*4*M + v) = x010 ^ x100 ^ x110;
      *reinterpret_cast<__v4du*>(xx + u*4*M + v+2) = x011 ^ x101 ^ x111;
      *reinterpret_cast<__v4du*>(xx + u*4*M + M + v) = x100;
      *reinterpret_cast<__v4du*>(xx + u*4*M + M + v+2) = x101;
      *reinterpret_cast<__v4du*>(xx + u*4*M + 2*M + v) = x010;
      *reinterpret_cast<__v4du*>(xx + u*4*M + 2*M + v+2) = x011;
      *reinterpret_cast<__v4du*>(xx + u*4*M + 3*M + v) = x000;
      *reinterpret_cast<__v4du*>(xx + u*4*M + 3*M + v+2) = x001;
    }
    x = xx;
  }
}



// compute the following change of basis:
// z[0][0] <-- zz[1][1]
// z[0][1] <-- zz[0][0] + zz[0][1]
// z[1][0] <-- zz[0][0] + zz[1][0]
// z[1][1] <-- zz[0][0]
static void inverseChangeOfBasisStrassen(const uint4* zz, uint4* z, int64_t n) {
  assert(n >= 64);
  assert(isPowerOfTwo(n));
  int64_t d = intlog2(n) - 6;
  assert(n == (1L<<d)*64);

  if (n == 64)
    avxMov(zz, z, 32);

  for (int64_t l = d-1; l >= 0; --l) {
    int64_t logM = (2*d-2*l+3);
    int64_t M = 1L << logM;
    for (int64_t t = 0; t < (1L << 2*d+3); ++t) {
      int64_t u = t >> logM;
      int64_t v = t & ~(0xffffffffffffffff << logM);
      uint4 zz00 = zz[u*4*M +       v];
      uint4 zz01 = zz[u*4*M + M   + v];
      uint4 zz10 = zz[u*4*M + 2*M + v];
      uint4 zz11 = zz[u*4*M + 3*M + v];
      z[u*4*M +       v] = zz11;
      z[u*4*M + M   + v] = zz00^zz01;
      z[u*4*M + 2*M + v] = zz00^zz10;
      z[u*4*M + 3*M + v] = zz00;
    }
    zz = z;
  }
}



static void arrangeDataCpuWords(const uint4* A, uint4* B, int64_t N) {
  assert(N % 4096 == 0);
  // determine d
  int d = 0;
  while ((1LL << 2*d) * 4096 < N) {
    ++d;
  }
  assert((1LL<<2*d)*4096 == N);

  vector<int> p = constructPwords(d);
  assert(isPermutation(p));

#pragma omp parallel for
  for (int64_t w = 0; w < N/64; ++w) {
    int64_t new_w = applyBitPermutation(p, w);
    if (new_w % 2 == 0 && w % 2 == 0) {
      B[new_w>>1].x = A[w>>1].x;
      B[new_w>>1].y = A[w>>1].y;
    }
    else if (new_w % 2 == 0 && w % 2 == 1) {
      B[new_w>>1].x = A[w>>1].z;
      B[new_w>>1].y = A[w>>1].w;
    }
    else if (new_w % 2 == 1 && w % 2 == 0) {
      B[new_w>>1].z = A[w>>1].x;
      B[new_w>>1].w = A[w>>1].y;
    }
    else  {
      B[new_w>>1].z = A[w>>1].z;
      B[new_w>>1].w = A[w>>1].w;
    }
  }
}



// p == block dimension, d == inner block dimension
// ie. side length n = 2^q * 2^d * 64
static void arrangeDataForMultiGpuCubicCpu(const uint4* A, uint4* B, int64_t q, int64_t d) {
  vector<int> p = constructPCubicwords(q,d);
  assert(isPermutation(p));

  int64_t N = 4096LL*(1LL<<2LL*(q+d));

  #pragma omp parallel for
  for (int64_t w = 0; w < N/64; ++w) {
    int64_t new_w = applyBitPermutation(p, w);
    if (new_w % 2 == 0 && w % 2 == 0) {
      B[new_w>>1].x = A[w>>1].x;
      B[new_w>>1].y = A[w>>1].y;
    }
    else if (new_w % 2 == 0 && w % 2 == 1) {
      B[new_w>>1].x = A[w>>1].z;
      B[new_w>>1].y = A[w>>1].w;
    }
    else if (new_w % 2 == 1 && w % 2 == 0) {
      B[new_w>>1].z = A[w>>1].x;
      B[new_w>>1].w = A[w>>1].y;
    }
    else  {
      B[new_w>>1].z = A[w>>1].z;
      B[new_w>>1].w = A[w>>1].w;
    }
  }
}



static void arrangeDataForCubicCpu(const uint4* A, uint4* B, int64_t N) {
  assert(N % 4096 == 0);
  // determine d
  int64_t d = 0;
  while ((1LL << 2*d) * 4096 < N) {
    ++d;
  }
  assert((1LL << 2*d) * 4096 == N);
  arrangeDataForMultiGpuCubicCpu(A, B, 0, d);
}



static void dearrangeDataCpuWords(const uint4* A, uint4* B, int64_t N) {
  assert(N % 4096 == 0);
  // determine d
  int d = 0;
  while ((1LL << 2*d) * 4096 < N) {
    ++d;
  }
  assert((1LL<<2*d)*4096 == N);

  vector<int> p = constructInvPwords(d);
  assert(isPermutation(p));

  #pragma omp parallel for
  for (int64_t w = 0; w < N/64; ++w) {
    int64_t new_w = applyBitPermutation(p, w);
    if (new_w % 2 == 0 && w % 2 == 0) {
      B[new_w>>1].x = A[w>>1].x;
      B[new_w>>1].y = A[w>>1].y;
    }
    else if (new_w % 2 == 0 && w % 2 == 1) {
      B[new_w>>1].x = A[w>>1].z;
      B[new_w>>1].y = A[w>>1].w;
    }
    else if (new_w % 2 == 1 && w % 2 == 0) {
      B[new_w>>1].z = A[w>>1].x;
      B[new_w>>1].w = A[w>>1].y;
    }
    else  {
      B[new_w>>1].z = A[w>>1].z;
      B[new_w>>1].w = A[w>>1].w;
    }
  }
}



static void dearrangeDataForMultiGpuCubicCpu(const uint4* A, uint4* B, int64_t q, int64_t d) {
  vector<int> p = constructInvPCubicwords(q,d);
  assert(isPermutation(p));

  int64_t N = 4096LL*(1LL<<2LL*(q+d));

#pragma omp parallel for
  for (int64_t w = 0; w < N/64; ++w) {
    int64_t new_w = applyBitPermutation(p, w);
    if (new_w % 2 == 0 && w % 2 == 0) {
      B[new_w>>1].x = A[w>>1].x;
      B[new_w>>1].y = A[w>>1].y;
    }
    else if (new_w % 2 == 0 && w % 2 == 1) {
      B[new_w>>1].x = A[w>>1].z;
      B[new_w>>1].y = A[w>>1].w;
    }
    else if (new_w % 2 == 1 && w % 2 == 0) {
      B[new_w>>1].z = A[w>>1].x;
      B[new_w>>1].w = A[w>>1].y;
    }
    else  {
      B[new_w>>1].z = A[w>>1].z;
      B[new_w>>1].w = A[w>>1].w;
    }
  }
}



static void dearrangeDataForCubicCpu(const uint4* A, uint4* B, int64_t N) {
  assert(N % 4096 == 0);
  // determine d
  int64_t d = 0;
  while ((1LL << 2*d) * 4096 < N) {
    ++d;
  }
  assert((1LL << 2*d) * 4096 == N);
  dearrangeDataForMultiGpuCubicCpu(A, B, 0, d);
}


// performs a xor on nwords between elements of A and B, storing result in C
// the number of threads == the length of A and B
__global__
void cudaUint4XorKernel(const uint4* A, const uint4* B, uint4* C) {
  int t = blockIdx.x*blockDim.x+threadIdx.x;
  uint4 a = A[t];
  uint4 b = B[t];
  uint4 c = a^b;
  C[t] = c;
}



// performs a xor on nwords between elements three operands A,B,C, storing result in D
// the number of threads == the length of A,B,C
__global__
void cudaUint4XorKernel3(const uint4* A, const uint4* B, const uint4* C,
                         uint4* D) {
  int t = blockIdx.x*blockDim.x+threadIdx.x;
  uint4 a = A[t];
  uint4 b = B[t];
  uint4 c = C[t];
  uint4 d = make_uint4(a.x^b.x^c.x,
                       a.y^b.y^c.y,
                       a.z^b.z^c.z,
                       a.w^b.w^c.w);
  D[t] = d;
}



// performs a xor on nwords between elements four operands A,B,C,D storing result in E
// the number of threads == the length of A,B,C,D
__global__
void cudaUint4XorKernel4(const uint4* A, const uint4* B, const uint4* C,
                         const uint4* D, uint4* E) {
  int t = blockIdx.x*blockDim.x+threadIdx.x;
  uint4 a = A[t];
  uint4 b = B[t];
  uint4 c = C[t];
  uint4 d = D[t];
  uint4 e = make_uint4(a.x^b.x^c.x^d.x,
                       a.y^b.y^c.y^d.y,
                       a.z^b.z^c.z^d.z,
                       a.w^b.w^c.w^d.w);
  E[t] = e;
}



__global__
void cudaMultiplicationKernel(const uint4* A, const uint4* B, uint4* C) {
  uint4 aik = A[blockIdx.x*blockDim.x + threadIdx.x];
  uint4 bjk = B[blockIdx.x*blockDim.x + threadIdx.x];
  uint4 cij = make_uint4(0,0,0,0);
  for (int jb = 0; jb < 32; ++jb) {
    // synchronize the values B at from thread jb
#if __CUDACC_VER_MAJOR__ < 8
    uint32_t bjkx = __shfl(bjk.x, jb);
    uint32_t bjky = __shfl(bjk.y, jb);
    uint32_t bjkz = __shfl(bjk.z, jb);
    uint32_t bjkw = __shfl(bjk.w, jb);
#else
    uint32_t bjkx = __shfl_sync(FULL_MASK, bjk.x, jb);
    uint32_t bjky = __shfl_sync(FULL_MASK, bjk.y, jb);
    uint32_t bjkz = __shfl_sync(FULL_MASK, bjk.z, jb);
    uint32_t bjkw = __shfl_sync(FULL_MASK, bjk.w, jb);
#endif
    cij.x |= (__popc((aik.x&bjkx)^(aik.y&bjky))&1) << jb;
    cij.y |= (__popc((aik.x&bjkz)^(aik.y&bjkw))&1) << jb;
    cij.z |= (__popc((aik.z&bjkx)^(aik.w&bjky))&1) << jb;
    cij.w |= (__popc((aik.z&bjkz)^(aik.w&bjkw))&1) << jb;
  }
  C[blockIdx.x*blockDim.x + threadIdx.x] = cij;
}



// same as above but with AND/OR instead of AND/XOR
__global__
void cudaBooleanMultiplicationKernel(const uint4* A, const uint4* B, uint4* C) {
  uint4 aik = A[blockIdx.x*blockDim.x + threadIdx.x];
  uint4 bjk = B[blockIdx.x*blockDim.x + threadIdx.x];
  uint4 cij = make_uint4(0,0,0,0);
  for (int jb = 0; jb < 32; ++jb) {
    // synchronize the values B at from thread jb
#if __CUDACC_VER_MAJOR__ < 8
    uint32_t bjkx = __shfl(bjk.x, jb);
    uint32_t bjky = __shfl(bjk.y, jb);
    uint32_t bjkz = __shfl(bjk.z, jb);
    uint32_t bjkw = __shfl(bjk.w, jb);
#else
    uint32_t bjkx = __shfl_sync(FULL_MASK, bjk.x, jb);
    uint32_t bjky = __shfl_sync(FULL_MASK, bjk.y, jb);
    uint32_t bjkz = __shfl_sync(FULL_MASK, bjk.z, jb);
    uint32_t bjkw = __shfl_sync(FULL_MASK, bjk.w, jb);
#endif
    cij.x |= (((aik.x&bjkx)|(aik.y&bjky))!=0) << jb;
    cij.y |= (((aik.x&bjkz)|(aik.y&bjkw))!=0) << jb;
    cij.z |= (((aik.z&bjkx)|(aik.w&bjky))!=0) << jb;
    cij.w |= (((aik.z&bjkz)|(aik.w&bjkw))!=0) << jb;
  }
  C[blockIdx.x*blockDim.x + threadIdx.x] = cij;
}



// perform cubic multiplication in 64x64 blocks
// assume A and B have shape (2^d * 64) * (2^d * 64)
__global__
void cudaCubicMultiplicationKernel(const uint4* A, const uint4* B,
                                   uint4* C, int d) {
  int t = blockIdx.x*blockDim.x + threadIdx.x;
  int aFibIdx = t >> (5+d);
  int bFibIdx = (t >> 5) & ~(0xffffffff << d);
  // the length of a fiber is 32 * 2^d words
  int fiberLength = 32*(1<<d);

  uint4 cij = make_uint4(0,0,0,0);
  for (int idx = 0; idx < (1<<d); ++idx) {
    int it = aFibIdx*fiberLength + 32*idx + threadIdx.x%32;
    int jt = bFibIdx*fiberLength + 32*idx + threadIdx.x%32;
    uint4 aik = A[it];
    uint4 bjk = B[jt];
    for (int jb = 0; jb < 32; ++jb) {
      uint32_t bjkx = __shfl_sync(FULL_MASK, bjk.x, jb);
      uint32_t bjky = __shfl_sync(FULL_MASK, bjk.y, jb);
      uint32_t bjkz = __shfl_sync(FULL_MASK, bjk.z, jb);
      uint32_t bjkw = __shfl_sync(FULL_MASK, bjk.w, jb);
      cij.x ^= (__popc((aik.x&bjkx)^(aik.y&bjky))&1) << jb;
      cij.y ^= (__popc((aik.x&bjkz)^(aik.y&bjkw))&1) << jb;
      cij.z ^= (__popc((aik.z&bjkx)^(aik.w&bjky))&1) << jb;
      cij.w ^= (__popc((aik.z&bjkz)^(aik.w&bjkw))&1) << jb;
    }
  }
  C[aFibIdx*fiberLength + bFibIdx*32 + threadIdx.x%32] = cij;
}



// perform cubic multiplication in 64x64 blocks
// assume A and B have shape (2^d * 64) * (2^d * 64)
__global__
void cudaBooleanCubicMultiplicationKernel(const uint4* A, const uint4* B,
                                          uint4* C, int d) {
  int64_t t = blockIdx.x*blockDim.x + threadIdx.x;
  int64_t aFibIdx = t >> (5+d);
  int64_t bFibIdx = (t >> 5) & ~(0xffffffff << d);
  // the length of a fiber is 32 * 2^d words
  int64_t fiberLength = 32*(1<<d);

  uint4 cij = make_uint4(0,0,0,0);
  for (int64_t idx = 0; idx < (1<<d); ++idx) {
    int64_t it = aFibIdx*fiberLength + 32*idx + threadIdx.x%32;
    int64_t jt = bFibIdx*fiberLength + 32*idx + threadIdx.x%32;
    uint4 aik = A[it];
    uint4 bjk = B[jt];
    for (int jb = 0; jb < 32; ++jb) {
      uint32_t bjkx = __shfl_sync(FULL_MASK, bjk.x, jb);
      uint32_t bjky = __shfl_sync(FULL_MASK, bjk.y, jb);
      uint32_t bjkz = __shfl_sync(FULL_MASK, bjk.z, jb);
      uint32_t bjkw = __shfl_sync(FULL_MASK, bjk.w, jb);
      cij.x |= (((aik.x&bjkx)|(aik.y&bjky))!=0) << jb;
      cij.y |= (((aik.x&bjkz)|(aik.y&bjkw))!=0) << jb;
      cij.z |= (((aik.z&bjkx)|(aik.w&bjky))!=0) << jb;
      cij.w |= (((aik.z&bjkz)|(aik.w&bjkw))!=0) << jb;
    }
  }
  C[aFibIdx*fiberLength + bFibIdx*32 + threadIdx.x%32] = cij;
}



static void cudaCubicMultiplication(const uint4* A_d, const uint4* B_d, uint4* C_d, uint4*, int n) {
  assert(n % 64 == 0);
  int d = 0;
  while (64*(1<<d) < n)
    ++d;
  assert(64*(1<<d) == n);

  int totalThreads = 32*(1<<2*d);
  int blockSize, numBlocks;
  determineBlockSizeAndNumBlocks(totalThreads, numBlocks, blockSize);
  cudaCubicMultiplicationKernel<<<numBlocks,blockSize>>>(A_d, B_d, C_d, d);
  CUDA_SYNC;

  // 3 * 2^d memops / thread
  MEMOPS += totalThreads*3*(1LL<<d);
}



static void cudaBooleanCubicMultiplication(const uint4* A_d, const uint4* B_d, uint4* C_d, uint4*, int n) {
  assert(n % 64 == 0);
  int d = 0;
  while (64*(1<<d) < n)
    ++d;
  assert(64*(1<<d) == n);

  int totalThreads = 32*(1<<2*d);
  int blockSize, numBlocks;
  determineBlockSizeAndNumBlocks(totalThreads, numBlocks, blockSize);
  cudaBooleanCubicMultiplicationKernel<<<numBlocks,blockSize>>>(A_d, B_d, C_d, d);
  CUDA_SYNC;

  // 3 * 2^d memops / thread
  MEMOPS += totalThreads*3*(1LL<<d);
}






// gpu version of above
// input: A
// output: T
// assume A has dimension 7^l * 4^(d-l) * 32
// d = number of "outer dimensions" (outside the 64x64 base block)
// l = 'level' (iterated from 0,...,d-1)
// output must have dimension 7^(l+1) * 4^(d-l-1) * 32
__global__
void cudaStrassenKernelA(const uint4* A, uint4* T, int d, int l) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int logM = 2*d-2*l+3;
  int M = 1 << logM;
  int u = t >> logM;
  int v = t & ~(0xffffffff << logM);
  
  uint4 A00 = A[u*4*M       + v];
  uint4 A01 = A[u*4*M +   M + v];
  uint4 A10 = A[u*4*M + 2*M + v];
  uint4 A11 = A[u*4*M + 3*M + v];
  
  const uint4& T0 = A10^A11;
  const uint4& T1 = A01;
  const uint4& T2 = A01^A11;
  const uint4& T3 = A10^T2;
  const uint4& T4 = A00^T3;
  const uint4& T5 = A10;
  const uint4& T6 = A00;

  T[u*7*M + 0*M + v] = T0;
  T[u*7*M + 1*M + v] = T1;
  T[u*7*M + 2*M + v] = T2;
  T[u*7*M + 3*M + v] = T3;
  T[u*7*M + 4*M + v] = T4;
  T[u*7*M + 5*M + v] = T5;
  T[u*7*M + 6*M + v] = T6;
}



// gpu version of above
// input: A
// output: T
// assume A has dimension 7^l * 4^(d-l) * 32
// d = number of "outer dimensions" (outside the 64x64 base block)
// l = 'level' (iterated from 0,...,d-1)
// output must have dimension 7^(l+1) * 4^(d-l-1) * 32
__global__
void cudaAlternateBasisStrassenKernelA(const uint4* A, uint4* T, int d, int l) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int logM = 2*d-2*l+3;
  int M = 1 << logM;
  int u = t >> logM;
  int v = t & ~(0xffffffff << logM);
  
  uint4 A00 = A[u*4*M       + v];
  uint4 A01 = A[u*4*M +   M + v];
  uint4 A10 = A[u*4*M + 2*M + v];
  uint4 A11 = A[u*4*M + 3*M + v];
  
  const uint4& T0 = A00^A10;
  const uint4& T1 = A10;
  const uint4& T2 = A00^A01;
  const uint4& T3 = A00;
  const uint4& T4 = A00^A11;
  const uint4& T5 = A01;
  const uint4& T6 = A11;

  T[u*7*M + 0*M + v] = T0;
  T[u*7*M + 1*M + v] = T1;
  T[u*7*M + 2*M + v] = T2;
  T[u*7*M + 3*M + v] = T3;
  T[u*7*M + 4*M + v] = T4;
  T[u*7*M + 5*M + v] = T5;
  T[u*7*M + 6*M + v] = T6;
}



__global__
void cudaAlternateBasisSixKernelA(const uint4* A, uint4* T, int d, int l) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int logM = 2*d-2*l+3;
  int M = 1 << logM;
  int u = t >> logM;
  int v = t & ~(0xffffffff << logM);
  
  uint4 A00 = A[u*4*M       + v];
  uint4 A01 = A[u*4*M +   M + v];
  uint4 A10 = A[u*4*M + 2*M + v];
  uint4 A11 = A[u*4*M + 3*M + v];
  
  const uint4& T0 = A00^A10;
  const uint4& T1 = A10;
  const uint4& T2 = A00^A01;
  const uint4& T3 = A00;
  const uint4& T4 = A00^A11;
  const uint4& T5 = A01;

  T[u*6*M + 0*M + v] = T0;
  T[u*6*M + 1*M + v] = T1;
  T[u*6*M + 2*M + v] = T2;
  T[u*6*M + 3*M + v] = T3;
  T[u*6*M + 4*M + v] = T4;
  T[u*6*M + 5*M + v] = T5;
}



// gpu version of above
// input: A
// output: T
// assume A has dimension 7^l * 4^(d-l) * 32
// d = number of "outer dimensions" (outside the 64x64 base block)
// l = 'level' (iterated from 0,2,...,d-2)
// will assume that d is odd and l is even
// output must have dimension 7^(l+2) * 4^(d-l-2) * 32
__global__
void cudaDoubleStrassenKernelA(const uint4* A, uint4* T, int d, int l) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int logM = 2*d-2*l+1;
  int M = 1 << logM;
  int u = t >> logM;
  int v = t & ~(0xffffffff << logM);

  uint4 Ts[4][7];
  for (int i = 0; i < 4; ++i) {
    const uint4& A00 = A[u*16*M +      i*M + v];
    const uint4& A01 = A[u*16*M +  (i+4)*M + v];
    const uint4& A10 = A[u*16*M +  (i+8)*M + v];
    const uint4& A11 = A[u*16*M + (i+12)*M + v];
    const uint4& T0 = A10^A11;
    const uint4& T1 = A01;
    const uint4& T2 = A01^A11;
    const uint4& T3 = A10^T2;
    const uint4& T4 = A00^T3;
    const uint4& T5 = A10;
    const uint4& T6 = A00;
    Ts[i][0] = T0;
    Ts[i][1] = T1;
    Ts[i][2] = T2;
    Ts[i][3] = T3;
    Ts[i][4] = T4;
    Ts[i][5] = T5;
    Ts[i][6] = T6;
  }

  for (int i = 0; i < 7; ++i) {
    const uint4& A00 = Ts[0][i];
    const uint4& A01 = Ts[1][i];
    const uint4& A10 = Ts[2][i];
    const uint4& A11 = Ts[3][i];
    const uint4& T0 = A10^A11;
    const uint4& T1 = A01;
    const uint4& T2 = A01^A11;
    const uint4& T3 = A10^T2;
    const uint4& T4 = A00^T3;
    const uint4& T5 = A10;
    const uint4& T6 = A00;
    T[u*49*M + i*7*M + 0*M + v] = T0;
    T[u*49*M + i*7*M + 1*M + v] = T1;
    T[u*49*M + i*7*M + 2*M + v] = T2;
    T[u*49*M + i*7*M + 3*M + v] = T3;
    T[u*49*M + i*7*M + 4*M + v] = T4;
    T[u*49*M + i*7*M + 5*M + v] = T5;
    T[u*49*M + i*7*M + 6*M + v] = T6;
  }
}



// gpu version of above
// input: A
// output: T
// assume A has dimension 7^l * 4^(d-l) * 32
// d = number of "outer dimensions" (outside the 64x64 base block)
// l = 'level' (iterated from 0,2,...,d-2)
// will assume that d is odd and l is even
// output must have dimension 7^(l+2) * 4^(d-l-2) * 32
__global__
void cudaAlternateBasisDoubleStrassenKernelA(const uint4* A, uint4* T, int d, int l) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int logM = 2*d-2*l+1;
  int M = 1 << logM;
  int u = t >> logM;
  int v = t & ~(0xffffffff << logM);

  uint4 Ts[4][7];
  for (int i = 0; i < 4; ++i) {
    const uint4& A00 = A[u*16*M +      i*M + v];
    const uint4& A01 = A[u*16*M +  (i+4)*M + v];
    const uint4& A10 = A[u*16*M +  (i+8)*M + v];
    const uint4& A11 = A[u*16*M + (i+12)*M + v];
    const uint4& T0 = A00^A10;
    const uint4& T1 = A10;
    const uint4& T2 = A00^A01;
    const uint4& T3 = A00;
    const uint4& T4 = A00^A11;
    const uint4& T5 = A01;
    const uint4& T6 = A11;
    Ts[i][0] = T0;
    Ts[i][1] = T1;
    Ts[i][2] = T2;
    Ts[i][3] = T3;
    Ts[i][4] = T4;
    Ts[i][5] = T5;
    Ts[i][6] = T6;
  }

  for (int i = 0; i < 7; ++i) {
    const uint4& A00 = Ts[0][i];
    const uint4& A01 = Ts[1][i];
    const uint4& A10 = Ts[2][i];
    const uint4& A11 = Ts[3][i];
    const uint4& T0 = A00^A10;
    const uint4& T1 = A10;
    const uint4& T2 = A00^A01;
    const uint4& T3 = A00;
    const uint4& T4 = A00^A11;
    const uint4& T5 = A01;
    const uint4& T6 = A11;
    T[u*49*M + i*7*M + 0*M + v] = T0;
    T[u*49*M + i*7*M + 1*M + v] = T1;
    T[u*49*M + i*7*M + 2*M + v] = T2;
    T[u*49*M + i*7*M + 3*M + v] = T3;
    T[u*49*M + i*7*M + 4*M + v] = T4;
    T[u*49*M + i*7*M + 5*M + v] = T5;
    T[u*49*M + i*7*M + 6*M + v] = T6;
  }
}



__global__
void cudaAlternateBasisDoubleSixKernelA(const uint4* A, uint4* T, int d, int l) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int logM = 2*d-2*l+1;
  int M = 1 << logM;
  int u = t >> logM;
  int v = t & ~(0xffffffff << logM);

  uint4 Ts[4][6];
  for (int i = 0; i < 4; ++i) {
    const uint4& A00 = A[u*16*M +      i*M + v];
    const uint4& A01 = A[u*16*M +  (i+4)*M + v];
    const uint4& A10 = A[u*16*M +  (i+8)*M + v];
    const uint4& A11 = A[u*16*M + (i+12)*M + v];
    const uint4& T0 = A00^A10;
    const uint4& T1 = A10;
    const uint4& T2 = A00^A01;
    const uint4& T3 = A00;
    const uint4& T4 = A00^A11;
    const uint4& T5 = A01;
    Ts[i][0] = T0;
    Ts[i][1] = T1;
    Ts[i][2] = T2;
    Ts[i][3] = T3;
    Ts[i][4] = T4;
    Ts[i][5] = T5;
  }

  for (int i = 0; i < 6; ++i) {
    const uint4& A00 = Ts[0][i];
    const uint4& A01 = Ts[1][i];
    const uint4& A10 = Ts[2][i];
    const uint4& A11 = Ts[3][i];
    const uint4& T0 = A00^A10;
    const uint4& T1 = A10;
    const uint4& T2 = A00^A01;
    const uint4& T3 = A00;
    const uint4& T4 = A00^A11;
    const uint4& T5 = A01;
    T[u*36*M + i*6*M + 0*M + v] = T0;
    T[u*36*M + i*6*M + 1*M + v] = T1;
    T[u*36*M + i*6*M + 2*M + v] = T2;
    T[u*36*M + i*6*M + 3*M + v] = T3;
    T[u*36*M + i*6*M + 4*M + v] = T4;
    T[u*36*M + i*6*M + 5*M + v] = T5;
  }
}



// gpu version of above
// input: B
// output: S
// assume B has dimension 7^l * 4^(d-l) * 32
// d = number of "outer dimensions" (outside the 64x64 base block)
// l = 'level' (iterated from 0,...,d-1)
// output must have dimension 7^(l+1) * 4^(d-l-1) * 32
__global__
void cudaStrassenKernelB(const uint4* B, uint4* S, int d, int l) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int logM = 2*d-2*l+3;
  int M = 1 << logM;
  int u = t >> logM;
  int v = t & ~(0xffffffff << logM);
  
  uint4 B00 = B[u*4*M       + v];
  uint4 B10 = B[u*4*M +   M + v];
  uint4 B01 = B[u*4*M + 2*M + v];
  uint4 B11 = B[u*4*M + 3*M + v];
 
  const uint4& S0 = B10^B11;
  const uint4& S1 = B10;
  const uint4& S2 = B01^B11;
  const uint4& S3 = B10^S2;
  const uint4& S4 = B01;
  const uint4& S5 = B00^S3;
  const uint4& S6 = B00;

  S[u*7*M + 0*M + v] = S0;
  S[u*7*M + 1*M + v] = S1;
  S[u*7*M + 2*M + v] = S2;
  S[u*7*M + 3*M + v] = S3;
  S[u*7*M + 4*M + v] = S4;
  S[u*7*M + 5*M + v] = S5;
  S[u*7*M + 6*M + v] = S6;
}



// gpu version of above
// input: B
// output: S
// assume B has dimension 7^l * 4^(d-l) * 32
// d = number of "outer dimensions" (outside the 64x64 base block)
// l = 'level' (iterated from 0,...,d-1)
// output must have dimension 7^(l+1) * 4^(d-l-1) * 32
__global__
void cudaAlternateBasisStrassenKernelB(const uint4* B, uint4* S, int d, int l) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int logM = 2*d-2*l+3;
  int M = 1 << logM;
  int u = t >> logM;
  int v = t & ~(0xffffffff << logM);
  
  uint4 B00 = B[u*4*M       + v];
  uint4 B10 = B[u*4*M +   M + v];
  uint4 B01 = B[u*4*M + 2*M + v];
  uint4 B11 = B[u*4*M + 3*M + v];
 
  const uint4& S0 = B00^B10;
  const uint4& S1 = B01;
  const uint4& S2 = B00^B01;
  const uint4& S3 = B00;
  const uint4& S4 = B10;
  const uint4& S5 = B00^B11;
  const uint4& S6 = B11;

  S[u*7*M + 0*M + v] = S0;
  S[u*7*M + 1*M + v] = S1;
  S[u*7*M + 2*M + v] = S2;
  S[u*7*M + 3*M + v] = S3;
  S[u*7*M + 4*M + v] = S4;
  S[u*7*M + 5*M + v] = S5;
  S[u*7*M + 6*M + v] = S6;
}



__global__
void cudaAlternateBasisSixKernelB(const uint4* B, uint4* S, int d, int l) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int logM = 2*d-2*l+3;
  int M = 1 << logM;
  int u = t >> logM;
  int v = t & ~(0xffffffff << logM);
  
  uint4 B00 = B[u*4*M       + v];
  uint4 B10 = B[u*4*M +   M + v];
  uint4 B01 = B[u*4*M + 2*M + v];
  uint4 B11 = B[u*4*M + 3*M + v];
 
  const uint4& S0 = B00^B10;
  const uint4& S1 = B01;
  const uint4& S2 = B00^B01;
  const uint4& S3 = B00;
  const uint4& S4 = B10;
  const uint4& S5 = B00^B11;

  S[u*6*M + 0*M + v] = S0;
  S[u*6*M + 1*M + v] = S1;
  S[u*6*M + 2*M + v] = S2;
  S[u*6*M + 3*M + v] = S3;
  S[u*6*M + 4*M + v] = S4;
  S[u*6*M + 5*M + v] = S5;
}



// gpu version of above
// input: B
// output: S
// assume A has dimension 7^l * 4^(d-l) * 32
// d = number of "outer dimensions" (outside the 64x64 base block)
// l = 'level' (iterated from 0,2,...,d-2)
// will assume that d is odd and l is even
// output must have dimension 7^(l+2) * 4^(d-l-2) * 32
__global__
void cudaDoubleStrassenKernelB(const uint4* B, uint4* S, int d, int l) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int logM = 2*d-2*l+1;
  int M = 1 << logM;
  int u = t >> logM;
  int v = t & ~(0xffffffff << logM);

  uint4 Ss[4][7];
  for (int i = 0; i < 4; ++i) {
    const uint4& B00 = B[u*16*M +      i*M + v];
    const uint4& B10 = B[u*16*M +  (i+4)*M + v];
    const uint4& B01 = B[u*16*M +  (i+8)*M + v];
    const uint4& B11 = B[u*16*M + (i+12)*M + v];
    const uint4& S0 = B10^B11;
    const uint4& S1 = B10;
    const uint4& S2 = B01^B11;
    const uint4& S3 = B10^S2;
    const uint4& S4 = B01;
    const uint4& S5 = B00^S3;
    const uint4& S6 = B00;

    Ss[i][0] = S0;
    Ss[i][1] = S1;
    Ss[i][2] = S2;
    Ss[i][3] = S3;
    Ss[i][4] = S4;
    Ss[i][5] = S5;
    Ss[i][6] = S6;
  }

  for (int i = 0; i < 7; ++i) {
    const uint4& B00 = Ss[0][i];
    const uint4& B10 = Ss[1][i];
    const uint4& B01 = Ss[2][i];
    const uint4& B11 = Ss[3][i];
    const uint4& S0 = B10^B11;
    const uint4& S1 = B10;
    const uint4& S2 = B01^B11;
    const uint4& S3 = B10^S2;
    const uint4& S4 = B01;
    const uint4& S5 = B00^S3;
    const uint4& S6 = B00;
    S[u*49*M + i*7*M + 0*M + v] = S0;
    S[u*49*M + i*7*M + 1*M + v] = S1;
    S[u*49*M + i*7*M + 2*M + v] = S2;
    S[u*49*M + i*7*M + 3*M + v] = S3;
    S[u*49*M + i*7*M + 4*M + v] = S4;
    S[u*49*M + i*7*M + 5*M + v] = S5;
    S[u*49*M + i*7*M + 6*M + v] = S6;
  }
}



// gpu version of above
// input: B
// output: S
// assume A has dimension 7^l * 4^(d-l) * 32
// d = number of "outer dimensions" (outside the 64x64 base block)
// l = 'level' (iterated from 0,2,...,d-2)
// will assume that d is odd and l is even
// output must have dimension 7^(l+2) * 4^(d-l-2) * 32
__global__
void cudaAlternateBasisDoubleStrassenKernelB(const uint4* B, uint4* S, int d, int l) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int logM = 2*d-2*l+1;
  int M = 1 << logM;
  int u = t >> logM;
  int v = t & ~(0xffffffff << logM);

  uint4 Ss[4][7];
  for (int i = 0; i < 4; ++i) {
    const uint4& B00 = B[u*16*M +      i*M + v];
    const uint4& B10 = B[u*16*M +  (i+4)*M + v];
    const uint4& B01 = B[u*16*M +  (i+8)*M + v];
    const uint4& B11 = B[u*16*M + (i+12)*M + v];
    const uint4& S0 = B00^B10;
    const uint4& S1 = B01;
    const uint4& S2 = B00^B01;
    const uint4& S3 = B00;
    const uint4& S4 = B10;
    const uint4& S5 = B00^B11;
    const uint4& S6 = B11;

    Ss[i][0] = S0;
    Ss[i][1] = S1;
    Ss[i][2] = S2;
    Ss[i][3] = S3;
    Ss[i][4] = S4;
    Ss[i][5] = S5;
    Ss[i][6] = S6;
  }

  for (int i = 0; i < 7; ++i) {
    const uint4& B00 = Ss[0][i];
    const uint4& B10 = Ss[1][i];
    const uint4& B01 = Ss[2][i];
    const uint4& B11 = Ss[3][i];
    const uint4& S0 = B00^B10;
    const uint4& S1 = B01;
    const uint4& S2 = B00^B01;
    const uint4& S3 = B00;
    const uint4& S4 = B10;
    const uint4& S5 = B00^B11;
    const uint4& S6 = B11;
    S[u*49*M + i*7*M + 0*M + v] = S0;
    S[u*49*M + i*7*M + 1*M + v] = S1;
    S[u*49*M + i*7*M + 2*M + v] = S2;
    S[u*49*M + i*7*M + 3*M + v] = S3;
    S[u*49*M + i*7*M + 4*M + v] = S4;
    S[u*49*M + i*7*M + 5*M + v] = S5;
    S[u*49*M + i*7*M + 6*M + v] = S6;
  }
}



__global__
void cudaAlternateBasisDoubleSixKernelB(const uint4* B, uint4* S, int d, int l) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int logM = 2*d-2*l+1;
  int M = 1 << logM;
  int u = t >> logM;
  int v = t & ~(0xffffffff << logM);

  uint4 Ss[4][6];
  for (int i = 0; i < 4; ++i) {
    const uint4& B00 = B[u*16*M +      i*M + v];
    const uint4& B10 = B[u*16*M +  (i+4)*M + v];
    const uint4& B01 = B[u*16*M +  (i+8)*M + v];
    const uint4& B11 = B[u*16*M + (i+12)*M + v];
    const uint4& S0 = B00^B10;
    const uint4& S1 = B01;
    const uint4& S2 = B00^B01;
    const uint4& S3 = B00;
    const uint4& S4 = B10;
    const uint4& S5 = B00^B11;

    Ss[i][0] = S0;
    Ss[i][1] = S1;
    Ss[i][2] = S2;
    Ss[i][3] = S3;
    Ss[i][4] = S4;
    Ss[i][5] = S5;
  }

  for (int i = 0; i < 6; ++i) {
    const uint4& B00 = Ss[0][i];
    const uint4& B10 = Ss[1][i];
    const uint4& B01 = Ss[2][i];
    const uint4& B11 = Ss[3][i];
    const uint4& S0 = B00^B10;
    const uint4& S1 = B01;
    const uint4& S2 = B00^B01;
    const uint4& S3 = B00;
    const uint4& S4 = B10;
    const uint4& S5 = B00^B11;
    S[u*36*M + i*6*M + 0*M + v] = S0;
    S[u*36*M + i*6*M + 1*M + v] = S1;
    S[u*36*M + i*6*M + 2*M + v] = S2;
    S[u*36*M + i*6*M + 3*M + v] = S3;
    S[u*36*M + i*6*M + 4*M + v] = S4;
    S[u*36*M + i*6*M + 5*M + v] = S5;
  }
}



// gpu version of above
// input: Ca
// output: Cb
// assume Ca has dimension 7^l * 4^(d-l) * 32
// d = number of "outer dimensions" (outside the 64x64 base block)
// l = 'level' (iterated from d-1,...,0)
// output must have dimension 7^(l-1) * 4^(d-l+1) * 32
__global__
void cudaStrassenKernelC(const uint4* Ca, uint4* Cb, int d, int l) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int logM = 2*d-2*l+3;
  int M = 1 << logM;
  int u = t >> logM;
  int v = t & ~(0xffffffff << logM);

  uint4 Q0  = Ca[u*7*M+0*M+v];
  uint4 Q1  = Ca[u*7*M+1*M+v];
  uint4 Q2  = Ca[u*7*M+2*M+v];
  uint4 Q3  = Ca[u*7*M+3*M+v];
  uint4 Q4  = Ca[u*7*M+4*M+v];
  uint4 Q5  = Ca[u*7*M+5*M+v];
  uint4 Q6  = Ca[u*7*M+6*M+v];
  uint4 U0  = Q1^Q3;
  uint4 U1  = U0^Q2;
  uint4 U2  = Q4^U0;
  uint4 C00 = Q1^Q6;
  uint4 C01 = U2^Q0;
  uint4 C10 = Q5^U1;
  uint4 C11 = Q0^U1;
  Cb[u*4*M       + v] = C00;
  Cb[u*4*M +   M + v] = C01;
  Cb[u*4*M + 2*M + v] = C10;
  Cb[u*4*M + 3*M + v] = C11;
}



// gpu version of above
// input: Ca
// output: Cb
// assume Ca has dimension 7^l * 4^(d-l) * 32
// d = number of "outer dimensions" (outside the 64x64 base block)
// l = 'level' (iterated from d-1,...,0)
// output must have dimension 7^(l-1) * 4^(d-l+1) * 32
__global__
void cudaAlternateBasisStrassenKernelC(const uint4* Ca, uint4* Cb, int d, int l) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int logM = 2*d-2*l+3;
  int M = 1 << logM;
  int u = t >> logM;
  int v = t & ~(0xffffffff << logM);

  uint4 Q0  = Ca[u*7*M+0*M+v];
  uint4 Q1  = Ca[u*7*M+1*M+v];
  uint4 Q2  = Ca[u*7*M+2*M+v];
  uint4 Q3  = Ca[u*7*M+3*M+v];
  uint4 Q4  = Ca[u*7*M+4*M+v];
  uint4 Q5  = Ca[u*7*M+5*M+v];
  uint4 Q6  = Ca[u*7*M+6*M+v];
  uint4 U0  = Q0^Q1;
  uint4 U1  = Q2^Q3;
  uint4 C00 = U0^U1;
  uint4 C01 = Q2^Q4;
  uint4 C10 = Q0^Q5;
  uint4 C11 = Q1^Q6;
  Cb[u*4*M       + v] = C00;
  Cb[u*4*M +   M + v] = C01;
  Cb[u*4*M + 2*M + v] = C10;
  Cb[u*4*M + 3*M + v] = C11;
}



__global__
void cudaAlternateBasisSixKernelC(const uint4* Ca, uint4* Cb, int d, int l) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int logM = 2*d-2*l+3;
  int M = 1 << logM;
  int u = t >> logM;
  int v = t & ~(0xffffffff << logM);

  uint4 Q0  = Ca[u*6*M+0*M+v];
  uint4 Q1  = Ca[u*6*M+1*M+v];
  uint4 Q2  = Ca[u*6*M+2*M+v];
  uint4 Q3  = Ca[u*6*M+3*M+v];
  uint4 Q4  = Ca[u*6*M+4*M+v];
  uint4 Q5  = Ca[u*6*M+5*M+v];
  uint4 U0  = Q0^Q1;
  uint4 U1  = Q2^Q3;
  uint4 C00 = U0^U1;
  uint4 C01 = Q2^Q4;
  uint4 C10 = Q0^Q5;
  uint4 C11 = Q1;
  Cb[u*4*M       + v] = C00;
  Cb[u*4*M +   M + v] = C01;
  Cb[u*4*M + 2*M + v] = C10;
  Cb[u*4*M + 3*M + v] = C11;
}



// gpu version of above
// input: Ca
// output: Cb
// assume Ca has dimension 7^l * 4^(d-l) * 32
// d = number of "outer dimensions" (outside the 64x64 base block)
// l = 'level' (iterated from 0,2,...,d-2)
// will assume that d is odd and l is even
// output must have dimension 7^(l-2) * 4^(d-l+2) * 32
__global__
void cudaDoubleStrassenKernelC(const uint4* Ca, uint4* Cb, int d, int l) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int logM = 2*d-2*l+1;
  int M = 1 << logM;
  int u = t >> logM;
  int v = t & ~(0xffffffff << logM);

  uint4 Cs[7][4];
  for (int i = 0; i < 7; ++i) {
    const uint4& Q0 = Ca[u*49*M + i*7*M       + v];
    const uint4& Q1 = Ca[u*49*M + i*7*M + 1*M + v];
    const uint4& Q2 = Ca[u*49*M + i*7*M + 2*M + v];
    const uint4& Q3 = Ca[u*49*M + i*7*M + 3*M + v];
    const uint4& Q4 = Ca[u*49*M + i*7*M + 4*M + v];
    const uint4& Q5 = Ca[u*49*M + i*7*M + 5*M + v];
    const uint4& Q6 = Ca[u*49*M + i*7*M + 6*M + v];
    uint4 U0  = Q1^Q3;
    uint4 U1  = U0^Q2;
    uint4 U2  = Q4^U0;
    Cs[i][0] = Q1^Q6;
    Cs[i][1] = U2^Q0;
    Cs[i][2] = Q5^U1;
    Cs[i][3] = Q0^U1;
  }

  for (int i = 0; i < 4; ++i) {
    const uint4& Q0 = Cs[0][i];
    const uint4& Q1 = Cs[1][i];
    const uint4& Q2 = Cs[2][i];
    const uint4& Q3 = Cs[3][i];
    const uint4& Q4 = Cs[4][i];
    const uint4& Q5 = Cs[5][i];
    const uint4& Q6 = Cs[6][i];
    uint4 U0  = Q1^Q3;
    uint4 U1  = U0^Q2;
    uint4 U2  = Q4^U0;
    Cb[u*16*M +      i*M + v] = Q1^Q6;
    Cb[u*16*M +  (i+4)*M + v] = U2^Q0;
    Cb[u*16*M +  (i+8)*M + v] = Q5^U1;
    Cb[u*16*M + (i+12)*M + v] = Q0^U1;
  }
}



// gpu version of above
// input: Ca
// output: Cb
// assume Ca has dimension 7^l * 4^(d-l) * 32
// d = number of "outer dimensions" (outside the 64x64 base block)
// l = 'level' (iterated from 0,2,...,d-2)
// will assume that d is odd and l is even
// output must have dimension 7^(l-2) * 4^(d-l+2) * 32
__global__
void cudaAlternateBasisDoubleStrassenKernelC(const uint4* Ca, uint4* Cb, int d, int l) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int logM = 2*d-2*l+1;
  int M = 1 << logM;
  int u = t >> logM;
  int v = t & ~(0xffffffff << logM);

  uint4 Cs[7][4];
  for (int i = 0; i < 7; ++i) {
    const uint4& Q0 = Ca[u*49*M + i*7*M       + v];
    const uint4& Q1 = Ca[u*49*M + i*7*M + 1*M + v];
    const uint4& Q2 = Ca[u*49*M + i*7*M + 2*M + v];
    const uint4& Q3 = Ca[u*49*M + i*7*M + 3*M + v];
    const uint4& Q4 = Ca[u*49*M + i*7*M + 4*M + v];
    const uint4& Q5 = Ca[u*49*M + i*7*M + 5*M + v];
    const uint4& Q6 = Ca[u*49*M + i*7*M + 6*M + v];
    uint4 U0  = Q0^Q1;
    uint4 U1  = Q2^Q3;
    Cs[i][0] = U0^U1;
    Cs[i][1] = Q2^Q4;
    Cs[i][2] = Q0^Q5;
    Cs[i][3] = Q1^Q6;
  }

  for (int i = 0; i < 4; ++i) {
    const uint4& Q0 = Cs[0][i];
    const uint4& Q1 = Cs[1][i];
    const uint4& Q2 = Cs[2][i];
    const uint4& Q3 = Cs[3][i];
    const uint4& Q4 = Cs[4][i];
    const uint4& Q5 = Cs[5][i];
    const uint4& Q6 = Cs[6][i];
    uint4 U0  = Q0^Q1;
    uint4 U1  = Q2^Q3;
    Cb[u*16*M +      i*M + v] = U0^U1;
    Cb[u*16*M +  (i+4)*M + v] = Q2^Q4;
    Cb[u*16*M +  (i+8)*M + v] = Q0^Q5;
    Cb[u*16*M + (i+12)*M + v] = Q1^Q6;
  }
}



__global__
void cudaAlternateBasisDoubleSixKernelC(const uint4* Ca, uint4* Cb, int d, int l) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int logM = 2*d-2*l+1;
  int M = 1 << logM;
  int u = t >> logM;
  int v = t & ~(0xffffffff << logM);

  uint4 Cs[6][4];
  for (int i = 0; i < 6; ++i) {
    const uint4& Q0 = Ca[u*36*M + i*6*M       + v];
    const uint4& Q1 = Ca[u*36*M + i*6*M + 1*M + v];
    const uint4& Q2 = Ca[u*36*M + i*6*M + 2*M + v];
    const uint4& Q3 = Ca[u*36*M + i*6*M + 3*M + v];
    const uint4& Q4 = Ca[u*36*M + i*6*M + 4*M + v];
    const uint4& Q5 = Ca[u*36*M + i*6*M + 5*M + v];
    uint4 U0  = Q0^Q1;
    uint4 U1  = Q2^Q3;
    Cs[i][0] = U0^U1;
    Cs[i][1] = Q2^Q4;
    Cs[i][2] = Q0^Q5;
    Cs[i][3] = Q1;
  }

  for (int i = 0; i < 4; ++i) {
    const uint4& Q0 = Cs[0][i];
    const uint4& Q1 = Cs[1][i];
    const uint4& Q2 = Cs[2][i];
    const uint4& Q3 = Cs[3][i];
    const uint4& Q4 = Cs[4][i];
    const uint4& Q5 = Cs[5][i];
    uint4 U0  = Q0^Q1;
    uint4 U1  = Q2^Q3;
    Cb[u*16*M +      i*M + v] = U0^U1;
    Cb[u*16*M +  (i+4)*M + v] = Q2^Q4;
    Cb[u*16*M +  (i+8)*M + v] = Q0^Q5;
    Cb[u*16*M + (i+12)*M + v] = Q1;
  }
}



// gpu version of above
// computes the final 4*32 uint4 multiplication at one go
// here we assume l = d-1 so M = 32
__global__
void cudaDoubleStrassenCoreMmKernel(const uint4* A, const uint4* B,
                                    uint4* C) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int u = t >> 5;
  int v = t & ~(0xffffffff << 5);

  uint4 A00 = A[u*128      + v];
  uint4 A01 = A[u*128 + 32 + v];
  uint4 A10 = A[u*128 + 64 + v];
  uint4 A11 = A[u*128 + 96 + v];

  // these should be shuffled
  uint4 B00 = B[u*128      + v];
  uint4 B10 = B[u*128 + 32 + v];
  uint4 B01 = B[u*128 + 64 + v];
  uint4 B11 = B[u*128 + 96 + v];

  uint4 T0 = A10^A11;
  uint4 T1 = A01;
  uint4 T2 = A01^A11;
  uint4 T3 = A10^T2;
  uint4 T4 = A00^T3;
  uint4 T5 = A10;
  uint4 T6 = A00;

  uint4 Q0 = make_uint4(0,0,0,0);
  uint4 Q1 = make_uint4(0,0,0,0);
  uint4 Q2 = make_uint4(0,0,0,0);
  uint4 Q3 = make_uint4(0,0,0,0);
  uint4 Q4 = make_uint4(0,0,0,0);
  uint4 Q5 = make_uint4(0,0,0,0);
  uint4 Q6 = make_uint4(0,0,0,0);
  for (int jb = 0; jb < 32; ++jb) {
    // these should be shuffled
    uint4 B00_s, B01_s, B10_s, B11_s;
    shfl(B00_s, B00, jb);
    shfl(B01_s, B01, jb);
    shfl(B10_s, B10, jb);
    shfl(B11_s, B11, jb);
    uint4 S0 = B10_s^B11_s;
    uint4 S1 = B10_s;
    uint4 S2 = B01_s^B11_s;
    uint4 S3 = B10_s^S2;
    uint4 S4 = B01_s;
    uint4 S5 = B00_s^S3;
    uint4 S6 = B00_s;
    
    Q0.x |= (__popc((T0.x&S0.x)^(T0.y&S0.y))&1) << jb;
    Q0.y |= (__popc((T0.x&S0.z)^(T0.y&S0.w))&1) << jb;
    Q0.z |= (__popc((T0.z&S0.x)^(T0.w&S0.y))&1) << jb;
    Q0.w |= (__popc((T0.z&S0.z)^(T0.w&S0.w))&1) << jb;

    Q1.x |= (__popc((T1.x&S1.x)^(T1.y&S1.y))&1) << jb;
    Q1.y |= (__popc((T1.x&S1.z)^(T1.y&S1.w))&1) << jb;
    Q1.z |= (__popc((T1.z&S1.x)^(T1.w&S1.y))&1) << jb;
    Q1.w |= (__popc((T1.z&S1.z)^(T1.w&S1.w))&1) << jb;
 
    Q2.x |= (__popc((T2.x&S2.x)^(T2.y&S2.y))&1) << jb;
    Q2.y |= (__popc((T2.x&S2.z)^(T2.y&S2.w))&1) << jb;
    Q2.z |= (__popc((T2.z&S2.x)^(T2.w&S2.y))&1) << jb;
    Q2.w |= (__popc((T2.z&S2.z)^(T2.w&S2.w))&1) << jb;

    Q3.x |= (__popc((T3.x&S3.x)^(T3.y&S3.y))&1) << jb;
    Q3.y |= (__popc((T3.x&S3.z)^(T3.y&S3.w))&1) << jb;
    Q3.z |= (__popc((T3.z&S3.x)^(T3.w&S3.y))&1) << jb;
    Q3.w |= (__popc((T3.z&S3.z)^(T3.w&S3.w))&1) << jb;

    Q4.x |= (__popc((T4.x&S4.x)^(T4.y&S4.y))&1) << jb;
    Q4.y |= (__popc((T4.x&S4.z)^(T4.y&S4.w))&1) << jb;
    Q4.z |= (__popc((T4.z&S4.x)^(T4.w&S4.y))&1) << jb;
    Q4.w |= (__popc((T4.z&S4.z)^(T4.w&S4.w))&1) << jb;

    Q5.x |= (__popc((T5.x&S5.x)^(T5.y&S5.y))&1) << jb;
    Q5.y |= (__popc((T5.x&S5.z)^(T5.y&S5.w))&1) << jb;
    Q5.z |= (__popc((T5.z&S5.x)^(T5.w&S5.y))&1) << jb;
    Q5.w |= (__popc((T5.z&S5.z)^(T5.w&S5.w))&1) << jb;

    Q6.x |= (__popc((T6.x&S6.x)^(T6.y&S6.y))&1) << jb;
    Q6.y |= (__popc((T6.x&S6.z)^(T6.y&S6.w))&1) << jb;
    Q6.z |= (__popc((T6.z&S6.x)^(T6.w&S6.y))&1) << jb;
    Q6.w |= (__popc((T6.z&S6.z)^(T6.w&S6.w))&1) << jb;
  }
  uint4 U0  = Q1^Q3;
  uint4 U1  = U0^Q2;
  uint4 U2  = Q4^U0;
  uint4 C00 = Q1^Q6;
  uint4 C01 = U2^Q0;
  uint4 C10 = Q5^U1;
  uint4 C11 = Q0^U1;
  C[u*128      + v] = C00;
  C[u*128 + 32 + v] = C01;
  C[u*128 + 64 + v] = C10; 
  C[u*128 + 96 + v] = C11;
}



// gpu version of above
// computes the final 4*32 uint4 multiplication at one go
// here we assume l = d-1 so M = 32
__global__
void cudaAlternateBasisDoubleStrassenCoreMmKernel(const uint4* A, const uint4* B,
                                                  uint4* C) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int u = t >> 5;
  int v = t & ~(0xffffffff << 5);

  uint4 A00 = A[u*128      + v];
  uint4 A01 = A[u*128 + 32 + v];
  uint4 A10 = A[u*128 + 64 + v];
  uint4 A11 = A[u*128 + 96 + v];

  // these should be shuffled
  uint4 B00 = B[u*128      + v];
  uint4 B10 = B[u*128 + 32 + v];
  uint4 B01 = B[u*128 + 64 + v];
  uint4 B11 = B[u*128 + 96 + v];

  uint4 T0 = A00^A10;
  uint4 T1 = A10;
  uint4 T2 = A00^A01;
  uint4 T3 = A00;
  uint4 T4 = A00^A11;
  uint4 T5 = A01;
  uint4 T6 = A11;

  uint4 Q0 = make_uint4(0,0,0,0);
  uint4 Q1 = make_uint4(0,0,0,0);
  uint4 Q2 = make_uint4(0,0,0,0);
  uint4 Q3 = make_uint4(0,0,0,0);
  uint4 Q4 = make_uint4(0,0,0,0);
  uint4 Q5 = make_uint4(0,0,0,0);
  uint4 Q6 = make_uint4(0,0,0,0);
  for (int jb = 0; jb < 32; ++jb) {
    // these should be shuffled
    uint4 B00_s, B01_s, B10_s, B11_s;
    shfl(B00_s, B00, jb);
    shfl(B01_s, B01, jb);
    shfl(B10_s, B10, jb);
    shfl(B11_s, B11, jb);
    uint4 S0 = B00_s^B10_s;
    uint4 S1 = B01_s;
    uint4 S2 = B00_s^B01_s;
    uint4 S3 = B00_s;
    uint4 S4 = B10_s;
    uint4 S5 = B00_s^B11_s;
    uint4 S6 = B11_s;
    
    Q0.x |= (__popc((T0.x&S0.x)^(T0.y&S0.y))&1) << jb;
    Q0.y |= (__popc((T0.x&S0.z)^(T0.y&S0.w))&1) << jb;
    Q0.z |= (__popc((T0.z&S0.x)^(T0.w&S0.y))&1) << jb;
    Q0.w |= (__popc((T0.z&S0.z)^(T0.w&S0.w))&1) << jb;

    Q1.x |= (__popc((T1.x&S1.x)^(T1.y&S1.y))&1) << jb;
    Q1.y |= (__popc((T1.x&S1.z)^(T1.y&S1.w))&1) << jb;
    Q1.z |= (__popc((T1.z&S1.x)^(T1.w&S1.y))&1) << jb;
    Q1.w |= (__popc((T1.z&S1.z)^(T1.w&S1.w))&1) << jb;
 
    Q2.x |= (__popc((T2.x&S2.x)^(T2.y&S2.y))&1) << jb;
    Q2.y |= (__popc((T2.x&S2.z)^(T2.y&S2.w))&1) << jb;
    Q2.z |= (__popc((T2.z&S2.x)^(T2.w&S2.y))&1) << jb;
    Q2.w |= (__popc((T2.z&S2.z)^(T2.w&S2.w))&1) << jb;

    Q3.x |= (__popc((T3.x&S3.x)^(T3.y&S3.y))&1) << jb;
    Q3.y |= (__popc((T3.x&S3.z)^(T3.y&S3.w))&1) << jb;
    Q3.z |= (__popc((T3.z&S3.x)^(T3.w&S3.y))&1) << jb;
    Q3.w |= (__popc((T3.z&S3.z)^(T3.w&S3.w))&1) << jb;

    Q4.x |= (__popc((T4.x&S4.x)^(T4.y&S4.y))&1) << jb;
    Q4.y |= (__popc((T4.x&S4.z)^(T4.y&S4.w))&1) << jb;
    Q4.z |= (__popc((T4.z&S4.x)^(T4.w&S4.y))&1) << jb;
    Q4.w |= (__popc((T4.z&S4.z)^(T4.w&S4.w))&1) << jb;

    Q5.x |= (__popc((T5.x&S5.x)^(T5.y&S5.y))&1) << jb;
    Q5.y |= (__popc((T5.x&S5.z)^(T5.y&S5.w))&1) << jb;
    Q5.z |= (__popc((T5.z&S5.x)^(T5.w&S5.y))&1) << jb;
    Q5.w |= (__popc((T5.z&S5.z)^(T5.w&S5.w))&1) << jb;

    Q6.x |= (__popc((T6.x&S6.x)^(T6.y&S6.y))&1) << jb;
    Q6.y |= (__popc((T6.x&S6.z)^(T6.y&S6.w))&1) << jb;
    Q6.z |= (__popc((T6.z&S6.x)^(T6.w&S6.y))&1) << jb;
    Q6.w |= (__popc((T6.z&S6.z)^(T6.w&S6.w))&1) << jb;
  }
  uint4 U0  = Q0^Q1;
  uint4 U1  = Q2^Q3;
  uint4 C00 = U0^U1;
  uint4 C01 = Q2^Q4;
  uint4 C10 = Q0^Q5;
  uint4 C11 = Q1^Q6;
  C[u*128      + v] = C00;
  C[u*128 + 32 + v] = C01;
  C[u*128 + 64 + v] = C10; 
  C[u*128 + 96 + v] = C11;
}



__global__
void cudaAlternateBasisDoubleSixCoreMmKernel(const uint4* A, const uint4* B,
                                             uint4* C) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int u = t >> 5;
  int v = t & ~(0xffffffff << 5);

  uint4 A00 = A[u*128      + v];
  uint4 A01 = A[u*128 + 32 + v];
  uint4 A10 = A[u*128 + 64 + v];
  uint4 A11 = A[u*128 + 96 + v];

  // these should be shuffled
  uint4 B00 = B[u*128      + v];
  uint4 B10 = B[u*128 + 32 + v];
  uint4 B01 = B[u*128 + 64 + v];
  uint4 B11 = B[u*128 + 96 + v];

  uint4 T0 = A00^A10;
  uint4 T1 = A10;
  uint4 T2 = A00^A01;
  uint4 T3 = A00;
  uint4 T4 = A00^A11;
  uint4 T5 = A01;

  uint4 Q0 = make_uint4(0,0,0,0);
  uint4 Q1 = make_uint4(0,0,0,0);
  uint4 Q2 = make_uint4(0,0,0,0);
  uint4 Q3 = make_uint4(0,0,0,0);
  uint4 Q4 = make_uint4(0,0,0,0);
  uint4 Q5 = make_uint4(0,0,0,0);
  for (int jb = 0; jb < 32; ++jb) {
    // these should be shuffled
    uint4 B00_s, B01_s, B10_s, B11_s;
    shfl(B00_s, B00, jb);
    shfl(B01_s, B01, jb);
    shfl(B10_s, B10, jb);
    shfl(B11_s, B11, jb);
    uint4 S0 = B00_s^B10_s;
    uint4 S1 = B01_s;
    uint4 S2 = B00_s^B01_s;
    uint4 S3 = B00_s;
    uint4 S4 = B10_s;
    uint4 S5 = B00_s^B11_s;
    
    Q0.x |= (__popc((T0.x&S0.x)^(T0.y&S0.y))&1) << jb;
    Q0.y |= (__popc((T0.x&S0.z)^(T0.y&S0.w))&1) << jb;
    Q0.z |= (__popc((T0.z&S0.x)^(T0.w&S0.y))&1) << jb;
    Q0.w |= (__popc((T0.z&S0.z)^(T0.w&S0.w))&1) << jb;

    Q1.x |= (__popc((T1.x&S1.x)^(T1.y&S1.y))&1) << jb;
    Q1.y |= (__popc((T1.x&S1.z)^(T1.y&S1.w))&1) << jb;
    Q1.z |= (__popc((T1.z&S1.x)^(T1.w&S1.y))&1) << jb;
    Q1.w |= (__popc((T1.z&S1.z)^(T1.w&S1.w))&1) << jb;
 
    Q2.x |= (__popc((T2.x&S2.x)^(T2.y&S2.y))&1) << jb;
    Q2.y |= (__popc((T2.x&S2.z)^(T2.y&S2.w))&1) << jb;
    Q2.z |= (__popc((T2.z&S2.x)^(T2.w&S2.y))&1) << jb;
    Q2.w |= (__popc((T2.z&S2.z)^(T2.w&S2.w))&1) << jb;

    Q3.x |= (__popc((T3.x&S3.x)^(T3.y&S3.y))&1) << jb;
    Q3.y |= (__popc((T3.x&S3.z)^(T3.y&S3.w))&1) << jb;
    Q3.z |= (__popc((T3.z&S3.x)^(T3.w&S3.y))&1) << jb;
    Q3.w |= (__popc((T3.z&S3.z)^(T3.w&S3.w))&1) << jb;

    Q4.x |= (__popc((T4.x&S4.x)^(T4.y&S4.y))&1) << jb;
    Q4.y |= (__popc((T4.x&S4.z)^(T4.y&S4.w))&1) << jb;
    Q4.z |= (__popc((T4.z&S4.x)^(T4.w&S4.y))&1) << jb;
    Q4.w |= (__popc((T4.z&S4.z)^(T4.w&S4.w))&1) << jb;

    Q5.x |= (__popc((T5.x&S5.x)^(T5.y&S5.y))&1) << jb;
    Q5.y |= (__popc((T5.x&S5.z)^(T5.y&S5.w))&1) << jb;
    Q5.z |= (__popc((T5.z&S5.x)^(T5.w&S5.y))&1) << jb;
    Q5.w |= (__popc((T5.z&S5.z)^(T5.w&S5.w))&1) << jb;
  }
  uint4 U0  = Q0^Q1;
  uint4 U1  = Q2^Q3;
  uint4 C00 = U0^U1;
  uint4 C01 = Q2^Q4;
  uint4 C10 = Q0^Q5;
  uint4 C11 = Q1;
  C[u*128      + v] = C00;
  C[u*128 + 32 + v] = C01;
  C[u*128 + 64 + v] = C10; 
  C[u*128 + 96 + v] = C11;
}



// let n == 64*2^d
// scratch must contain 3*32*7^d words
static void cudaDoubleStrassen(const uint4* A, const uint4* B, uint4* C,
                        uint4* scratch, int n) {
  int d = 0;
  while ((1<<d)*64 < n)
    ++d;
  assert((1<<d)*64 == n);

  if (d == 0) {
    // account for the special case of a single 64x64 mm
    cudaMultiplicationKernel<<<1,32>>>(A, B, C);
    CUDA_SYNC;
    // 3 memops / thread, 32 threads
    MEMOPS += 32*3;
    return;
  }

  int totalThreads, blockSize, numBlocks;
  int scratchSliceLength = 32*intpow(7,d);
  uint4* A_d = scratch;
  uint4* B_d = scratch + scratchSliceLength;
  uint4* C_d = scratch + 2*scratchSliceLength;
  gpuMemcpy(32*(1<<2*d), A, A_d);
  gpuMemcpy(32*(1<<2*d), B, B_d);

  int l = 0;
  if (d%2 == 0) {
    totalThreads = (1<<(2*d+3));
    determineBlockSizeAndNumBlocks(totalThreads, numBlocks, blockSize);
    cudaStrassenKernelA<<<numBlocks,blockSize>>>(A_d, C_d, d, l);
    CUDA_SYNC;
    std::swap(A_d,C_d);
    cudaStrassenKernelB<<<numBlocks,blockSize>>>(B_d, C_d, d, l);
    std::swap(B_d,C_d);
    // 7+4 memops / kernel / thread
    MEMOPS += totalThreads*22;
    ++l;
  }

  for (; l < d-1; l += 2) {
    totalThreads = intpow(7,l)*(1<<(2*d-2*l+1));
    determineBlockSizeAndNumBlocks(totalThreads, numBlocks, blockSize);
    cudaDoubleStrassenKernelA<<<numBlocks,blockSize>>>(A_d, C_d, d, l);
    CUDA_SYNC;
    std::swap(A_d,C_d);
    cudaDoubleStrassenKernelB<<<numBlocks,blockSize>>>(B_d, C_d, d, l);
    CUDA_SYNC;
    std::swap(B_d,C_d);

    // 4*4 + 7*7 memops / kernel / thread
    MEMOPS += 2*(16+49)*totalThreads;
  }
 
  assert(l == d-1);
  totalThreads = 32*intpow(7,d-1);
  determineBlockSizeAndNumBlocks(totalThreads, numBlocks, blockSize);
  cudaDoubleStrassenCoreMmKernel<<<numBlocks,blockSize>>>(A_d, B_d, C_d);
  CUDA_SYNC;
  // 12 memops / thread
  MEMOPS += totalThreads*12;

  for (l = d-3; l >= 0; l -= 2) {
    totalThreads = intpow(7,l)*(1<<(2*d-2*l+1));
    determineBlockSizeAndNumBlocks(totalThreads, numBlocks, blockSize);
    cudaDoubleStrassenKernelC<<<numBlocks,blockSize>>>(C_d, A_d, d, l);
    CUDA_SYNC;
    std::swap(C_d,A_d);
    // 7*7+4*4 / thread
    MEMOPS += (49+16)*totalThreads;
  }

  if (d % 2 == 0) {
    assert(l == -1);
    l = 0;
    totalThreads = 1 << (2*d+3);
    determineBlockSizeAndNumBlocks(totalThreads, numBlocks, blockSize);
    cudaStrassenKernelC<<<numBlocks,blockSize>>>(C_d, A_d, d, l);
    CUDA_SYNC;
    std::swap(C_d,A_d);
    // 7+4 memops / thread
    MEMOPS += 11*totalThreads;
  }

  gpuMemcpy(32*(1<<2*d), C_d, C);
}



// let n == 64*2^d
// scratch must contain 3*32*7^d words
static void cudaAlternateBasisDoubleStrassen(const uint4* A, const uint4* B, uint4* C,
                                             uint4* scratch, int n) {
  int d = 0;
  while ((1<<d)*64 < n)
    ++d;
  assert((1<<d)*64 == n);

  if (d == 0) {
    // account for the special case of a single 64x64 mm
    cudaMultiplicationKernel<<<1,32>>>(A, B, C);
    CUDA_SYNC;
    // 3 memops / thread, 32 threads
    MEMOPS += 32*3;
    return;
  }

  int totalThreads, blockSize, numBlocks;
  int scratchSliceLength = 32*intpow(7,d);
  uint4* A_d = scratch;
  uint4* B_d = scratch + scratchSliceLength;
  uint4* C_d = scratch + 2*scratchSliceLength;
  gpuMemcpy(32*(1<<2*d), A, A_d);
  gpuMemcpy(32*(1<<2*d), B, B_d);

  int l = 0;
  if (d%2 == 0) {
    totalThreads = (1<<(2*d+3));
    determineBlockSizeAndNumBlocks(totalThreads, numBlocks, blockSize);
    cudaAlternateBasisStrassenKernelA<<<numBlocks,blockSize>>>(A_d, C_d, d, l);
    CUDA_SYNC;
    std::swap(A_d,C_d);
    cudaAlternateBasisStrassenKernelB<<<numBlocks,blockSize>>>(B_d, C_d, d, l);
    CUDA_SYNC;
    std::swap(B_d,C_d);
    // 7+4 memops / kernel / thread
    MEMOPS += totalThreads*22;
    ++l;
  }

  for (; l < d-1; l += 2) {
    totalThreads = intpow(7,l)*(1<<(2*d-2*l+1));
    determineBlockSizeAndNumBlocks(totalThreads, numBlocks, blockSize);
    cudaAlternateBasisDoubleStrassenKernelA<<<numBlocks,blockSize>>>(A_d, C_d, d, l);
    CUDA_SYNC;
    std::swap(A_d,C_d);
    cudaAlternateBasisDoubleStrassenKernelB<<<numBlocks,blockSize>>>(B_d, C_d, d, l);
    CUDA_SYNC;
    std::swap(B_d,C_d);

    // 4*4 + 7*7 memops / kernel / thread
    MEMOPS += 2*(16+49)*totalThreads;
  }
 
  assert(l == d-1);
  totalThreads = 32*intpow(7,d-1);
  determineBlockSizeAndNumBlocks(totalThreads, numBlocks, blockSize);
  cudaAlternateBasisDoubleStrassenCoreMmKernel<<<numBlocks,blockSize>>>(A_d, B_d, C_d);
  CUDA_SYNC;
  // 12 memops / thread
  MEMOPS += totalThreads*12;

  for (l = d-3; l >= 0; l -= 2) {
    totalThreads = intpow(7,l)*(1<<(2*d-2*l+1));
    determineBlockSizeAndNumBlocks(totalThreads, numBlocks, blockSize);
    cudaAlternateBasisDoubleStrassenKernelC<<<numBlocks,blockSize>>>(C_d, A_d, d, l);
    CUDA_SYNC;
    std::swap(C_d,A_d);
    // 7*7+4*4 / thread
    MEMOPS += (49+16)*totalThreads;
  }

  if (d % 2 == 0) {
    assert(l == -1);
    l = 0;
    totalThreads = 1 << (2*d+3);
    determineBlockSizeAndNumBlocks(totalThreads, numBlocks, blockSize);
    cudaAlternateBasisStrassenKernelC<<<numBlocks,blockSize>>>(C_d, A_d, d, l);
    CUDA_SYNC;
    std::swap(C_d,A_d);
    // 7+4 memops / thread
    MEMOPS += 11*totalThreads;
  }

  gpuMemcpy(32*(1<<2*d), C_d, C);
}



// let n == 64*2^d
// scratch must contain 3*32*6^d words
static void cudaAlternateBasisDoubleSix(const uint4* A, const uint4* B, uint4* C,
                                        uint4* scratch, int n) {
  int d = 0;
  while ((1<<d)*64 < n)
    ++d;
  assert((1<<d)*64 == n);

  if (d == 0) {
    // account for the special case of a single 64x64 mm
    cudaMultiplicationKernel<<<1,32>>>(A, B, C);
    CUDA_SYNC;
    return;
  }

  int totalThreads, blockSize, numBlocks;
  int scratchSliceLength = 32*intpow(6,d);
  uint4* A_d = scratch;
  uint4* B_d = scratch + scratchSliceLength;
  uint4* C_d = scratch + 2*scratchSliceLength;
  gpuMemcpy(32*(1<<2*d), A, A_d);
  gpuMemcpy(32*(1<<2*d), B, B_d);

  int l = 0;
  if (d%2 == 0) {
    totalThreads = (1<<(2*d+3));
    determineBlockSizeAndNumBlocks(totalThreads, numBlocks, blockSize);
    cudaAlternateBasisSixKernelA<<<numBlocks,blockSize>>>(A_d, C_d, d, l);
    CUDA_SYNC;
    std::swap(A_d,C_d);
    cudaAlternateBasisSixKernelB<<<numBlocks,blockSize>>>(B_d, C_d, d, l);
    CUDA_SYNC;
    std::swap(B_d,C_d);
    ++l;
  }

  for (; l < d-1; l += 2) {
    totalThreads = intpow(6,l)*(1<<(2*d-2*l+1));
    determineBlockSizeAndNumBlocks(totalThreads, numBlocks, blockSize);
    cudaAlternateBasisDoubleSixKernelA<<<numBlocks,blockSize>>>(A_d, C_d, d, l);
    CUDA_SYNC;
    std::swap(A_d,C_d);
    cudaAlternateBasisDoubleSixKernelB<<<numBlocks,blockSize>>>(B_d, C_d, d, l);
    CUDA_SYNC;
    std::swap(B_d,C_d);
  }
  
  assert(l == d-1);
  totalThreads = 32*intpow(6,d-1);
  determineBlockSizeAndNumBlocks(totalThreads, numBlocks, blockSize);
  cudaAlternateBasisDoubleSixCoreMmKernel<<<numBlocks,blockSize>>>(A_d, B_d, C_d);
  CUDA_SYNC;

  for (l = d-3; l >= 0; l -= 2) {
    totalThreads = intpow(6,l)*(1<<(2*d-2*l+1));
    determineBlockSizeAndNumBlocks(totalThreads, numBlocks, blockSize);
    cudaAlternateBasisDoubleSixKernelC<<<numBlocks,blockSize>>>(C_d, A_d, d, l);
    CUDA_SYNC;
    std::swap(C_d,A_d);
  }

  if (d % 2 == 0) {
    assert(l == -1);
    l = 0;
    totalThreads = 1 << (2*d+3);
    determineBlockSizeAndNumBlocks(totalThreads, numBlocks, blockSize);
    cudaAlternateBasisSixKernelC<<<numBlocks,blockSize>>>(C_d, A_d, d, l);
    CUDA_SYNC;
    std::swap(C_d,A_d);
  }
  
  gpuMemcpy(32*(1<<2*d), C_d, C);
}



static void cudaRecursiveDoubleStrassen(const uint4* A, const uint4* B, uint4* C, uint4* scratch, int n) {
  assert(isPowerOfTwo(n));
  int d = intlog2(n) - 6;
  if (d <= 8) {
    cudaDoubleStrassen(A, B, C, scratch, n);
    return;
  }
  
  int submatrixLength = 32*(1<<2*(d-1));
  const uint4* A00 = A;
  const uint4* A01 = A + submatrixLength;
  const uint4* A10 = A + 2*submatrixLength;
  const uint4* A11 = A + 3*submatrixLength;
  const uint4* B00 = B;
  const uint4* B10 = B + submatrixLength;
  const uint4* B01 = B + 2*submatrixLength;
  const uint4* B11 = B + 3*submatrixLength;

  uint4* T[7];
  uint4* S[7];
  uint4* Q[7];
  for (int i = 0; i < 7; ++i) {
    T[i] = scratch + submatrixLength*i;
    S[i] = scratch + submatrixLength*(7+i);
    Q[i] = scratch + submatrixLength*(14+i);
  }

  int numBlocks, blockSize;
  determineBlockSizeAndNumBlocks(submatrixLength, numBlocks, blockSize);

  cudaUint4XorKernel<<<numBlocks,blockSize>>>(A10,A11,T[0]);
  CUDA_SYNC;
  gpuMemcpy(submatrixLength, A01, T[1]);
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(A01,A11,T[2]);
  CUDA_SYNC;
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(A10,T[2],T[3]);
  CUDA_SYNC;
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(A00,T[3],T[4]);
  CUDA_SYNC;
  gpuMemcpy(submatrixLength, A10, T[5]);
  gpuMemcpy(submatrixLength, A00, T[6]);

  cudaUint4XorKernel<<<numBlocks,blockSize>>>(B10,B11,S[0]);
  CUDA_SYNC;
  gpuMemcpy(submatrixLength, B10, S[1]);
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(B01,B11,S[2]);
  CUDA_SYNC;
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(B10,S[2],S[3]);
  CUDA_SYNC;
  gpuMemcpy(submatrixLength, B01, S[4]);
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(B00,S[3],S[5]);
  CUDA_SYNC;
  gpuMemcpy(submatrixLength, B00, S[6]);
  
  // each four xor kernel performs 3 * submatrix length number of memops
  MEMOPS += 3*8*submatrixLength;
  
  // operands must be reused because of memory allocation
  for (int i = 0; i < 7; ++i) 
    cudaRecursiveDoubleStrassen(T[i], S[i], Q[i], scratch + submatrixLength*21, n>>1);

  uint4* U[3] = {T[0], T[1], T[2]};
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(Q[1],Q[3],U[0]);
  CUDA_SYNC;
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(U[0],Q[2],U[1]);
  CUDA_SYNC;
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(Q[4],U[0],U[2]);
  CUDA_SYNC;

  uint4* C00 = C;
  uint4* C01 = C + submatrixLength;
  uint4* C10 = C + 2*submatrixLength;
  uint4* C11 = C + 3*submatrixLength;

  cudaUint4XorKernel<<<numBlocks,blockSize>>>(Q[1],Q[6],C00);
  CUDA_SYNC;
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(U[2],Q[0],C01);
  CUDA_SYNC;
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(Q[5],U[1],C10);
  CUDA_SYNC;
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(Q[0],U[1],C11);
  CUDA_SYNC;

  // each seven xor kernel performs 3 * submatrix length number of memops
  MEMOPS += 3*7*submatrixLength;
}



static void cudaRecursiveAlternateBasisDoubleStrassen(const uint4* A, const uint4* B,
                                                      uint4* C, uint4* scratch, int n) {
  int d = intlog2(n) - 6;
  if (d <= 8) {
    cudaAlternateBasisDoubleStrassen(A, B, C, scratch, n);
    return;
  }
  
  int submatrixLength = 32*(1<<2*(d-1));
  const uint4* A00 = A;
  const uint4* A01 = A + submatrixLength;
  const uint4* A10 = A + 2*submatrixLength;
  const uint4* A11 = A + 3*submatrixLength;
  const uint4* B00 = B;
  const uint4* B10 = B + submatrixLength;
  const uint4* B01 = B + 2*submatrixLength;
  const uint4* B11 = B + 3*submatrixLength;
 
  uint4* T[7];
  uint4* S[7];
  uint4* Q[7];
  for (int i = 0; i < 7; ++i) {
    T[i] = scratch + submatrixLength*i;
    S[i] = scratch + submatrixLength*(7+i);
    Q[i] = scratch + submatrixLength*(14+i);
  }

  int numBlocks, blockSize;
  determineBlockSizeAndNumBlocks(submatrixLength, numBlocks, blockSize);

  cudaUint4XorKernel<<<numBlocks,blockSize>>>(A00,A10,T[0]);
  gpuMemcpy(submatrixLength, A10, T[1]);
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(A00,A01,T[2]);
  gpuMemcpy(submatrixLength, A00, T[3]);
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(A00,A11,T[4]);
  gpuMemcpy(submatrixLength, A01, T[5]);
  gpuMemcpy(submatrixLength, A11, T[6]);
   
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(B00,B10,S[0]);
  gpuMemcpy(submatrixLength, B01, S[1]);
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(B00,B01,S[2]);
  gpuMemcpy(submatrixLength, B00, S[3]);
  gpuMemcpy(submatrixLength, B10, S[4]);
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(B00,B11,S[5]);
  gpuMemcpy(submatrixLength, B11, S[6]);
  CUDA_SYNC;
 
  // each four xor kernel performs 3 * submatrix length number of memops
  MEMOPS += 3*8*submatrixLength;

  // operands must be reused because of memory allocation
  for (int i = 0; i < 7; ++i) 
    cudaRecursiveAlternateBasisDoubleStrassen(T[i], S[i], Q[i], scratch + submatrixLength*21, n>>1);

  uint4* C00 = C;
  uint4* C01 = C + submatrixLength;
  uint4* C10 = C + 2*submatrixLength;
  uint4* C11 = C + 3*submatrixLength;

  cudaUint4XorKernel4<<<numBlocks,blockSize>>>(Q[0],Q[1],Q[2],Q[3],C00);
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(Q[2],Q[4],C01);
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(Q[0],Q[5],C10);
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(Q[1],Q[6],C11);
  CUDA_SYNC;

  // each seven xor kernel performs 3 * submatrix length number of memops
  MEMOPS += 3*7*submatrixLength;
}



static void cudaRecursiveAlternateBasisDoubleSix(const uint4* A, const uint4* B,
                                                 uint4* C, uint4* scratch, int n) {
  int d = intlog2(n) - 6;
  if (d <= 8) {
    cudaAlternateBasisDoubleSix(A, B, C, scratch, n);
  }
  else if (d <= 11) {
    int submatrixLength = 32*(1<<2*(d-1));
    const uint4* A00 = A;
    const uint4* A01 = A + submatrixLength;
    const uint4* A10 = A + 2*submatrixLength;
    const uint4* A11 = A + 3*submatrixLength;
    const uint4* B00 = B;
    const uint4* B10 = B + submatrixLength;
    const uint4* B01 = B + 2*submatrixLength;
    const uint4* B11 = B + 3*submatrixLength;
    
    uint4* T = scratch;
    uint4* S = scratch + submatrixLength;
    uint4* Q[6];
    for (int i = 0; i < 6; ++i) {
      Q[i] = scratch + submatrixLength*(2+i);
    }

    int numBlocks, blockSize;
    determineBlockSizeAndNumBlocks(submatrixLength, numBlocks, blockSize);

    cudaUint4XorKernel<<<numBlocks,blockSize>>>(A00,A10,T);
    cudaUint4XorKernel<<<numBlocks,blockSize>>>(B00,B10,S);
    CUDA_SYNC;
    cudaRecursiveAlternateBasisDoubleSix(T, S, Q[0], scratch + submatrixLength*8, n>>1);

    cudaRecursiveAlternateBasisDoubleSix(A10, B01, Q[1], scratch + submatrixLength*8, n>>1);

    cudaUint4XorKernel<<<numBlocks,blockSize>>>(A00,A01,T);
    cudaUint4XorKernel<<<numBlocks,blockSize>>>(B00,B01,S);
    CUDA_SYNC;
    cudaRecursiveAlternateBasisDoubleSix(T, S, Q[2], scratch + submatrixLength*8, n>>1);

    cudaRecursiveAlternateBasisDoubleSix(A00, B00, Q[3], scratch + submatrixLength*8, n>>1);

    cudaUint4XorKernel<<<numBlocks,blockSize>>>(A00,A11,T);
    CUDA_SYNC;
    cudaRecursiveAlternateBasisDoubleSix(T, B10, Q[4], scratch + submatrixLength*8, n>>1);

    cudaUint4XorKernel<<<numBlocks,blockSize>>>(B00,B11,S);
    CUDA_SYNC;
    cudaRecursiveAlternateBasisDoubleSix(A01, S, Q[5], scratch + submatrixLength*8, n>>1);
    
    uint4* C00 = C;
    uint4* C01 = C + submatrixLength;
    uint4* C10 = C + 2*submatrixLength;
    uint4* C11 = C + 3*submatrixLength;
    
    cudaUint4XorKernel4<<<numBlocks,blockSize>>>(Q[0],Q[1],Q[2],Q[3],C00);
    cudaUint4XorKernel<<<numBlocks,blockSize>>>(Q[2],Q[4],C01);
    cudaUint4XorKernel<<<numBlocks,blockSize>>>(Q[0],Q[5],C10);
    gpuMemcpy(submatrixLength,Q[1],C11);
    CUDA_SYNC;   
  }
  else {
    assert(false && "insufficient memory");
  }
}



// data is assumed to have been arranged appropriately beforehand
static void multiGpuCubic(const uint4* A, const uint4* B, uint4* C, int n,
                          int subn, uint4** A_ds, uint4** B_ds, uint4** C_ds) {  
  // assume the matrix has shape 2^q*2^d*64 x 2^q*2^d*64; that is, the
  // matrix consist of 2^q * 2^q blocks of dimension 2^d*64 x 2^d*64 each
  // where 2^d*64 == subproblem size
  
  int d = 0;
  while ((1<<d)*64 < subn)
    ++d;
  assert((1<<d)*64 == subn);
  int q = 0;
  while ((1<<q)*(1<<d)*64 < n)
    ++q;
  assert((1<<q)*(1<<d)*64 == n);

  // number of uint4s required to represent the full matrix
  int64_t nWordsFull = (1<<2*(d+q))*32;
  // number of uint4s required to represent the submatrix
  int64_t nWordsSub = (1<<2*d)*32;
  vector< vector<uint4> > prev_Cs; // previous result
  for (int i = 0; i < GPU_COUNT; ++i)
    prev_Cs.push_back(vector<uint4>(nWordsSub));

  int blockCount = 1<<2*q;

  // initially set the entire output matrix to zero
  for (int64_t i = 0; i < nWordsFull; ++i)
    C[i] = make_uint4(0,0,0,0);

  int totalThreads = 32*(1<<2*d);
  int blockSize, numBlocks;
  determineBlockSizeAndNumBlocks(totalThreads, numBlocks, blockSize);

  int64_t nanosPerThread[8];
  int64_t wordsPerThread[8];
  
  #pragma omp parallel for
  for (int gpu = 0; gpu < GPU_COUNT; ++gpu) {
    cudaSetDevice(gpu);

    uint4* A_d = A_ds[gpu];
    uint4* B_d = B_ds[gpu];
    uint4* C_d = C_ds[gpu];
    vector<uint4>& prev_C = prev_Cs[gpu];

    nanosPerThread[gpu] = 0;
    wordsPerThread[gpu] = 0;
    
    int prevCBlockIdx = -1;
    for (int cBlockIdx = gpu; cBlockIdx < blockCount+GPU_COUNT;
         cBlockIdx += GPU_COUNT) {
      int cBlockIdx_i = cBlockIdx/(1<<q);
      int cBlockIdx_j = cBlockIdx%(1<<q);
      for (int k = 0; k < (1<<q); ++k) {        
        if (cBlockIdx < blockCount) {
          gpuUpload(nWordsSub, &A[(cBlockIdx_i*(1<<q)+k)*nWordsSub], A_d);
          gpuUpload(nWordsSub, &B[(cBlockIdx_j*(1<<q)+k)*nWordsSub], B_d);
          cudaCubicMultiplicationKernel<<<numBlocks, blockSize>>>(A_d, B_d, C_d, d);
        }
        if (prevCBlockIdx >= 0) {
          // integrate the previous result
          auto start = now();
          for (int64_t w = 0; w < nWordsSub; ++w) {
            C[prevCBlockIdx*nWordsSub+w] = C[prevCBlockIdx*nWordsSub+w] ^ prev_C[w];
          }
          auto end = now();
          nanoseconds diff = end-start;
          nanosPerThread[gpu] += diff.count();
          wordsPerThread[gpu] += nWordsSub;
        }
        if (cBlockIdx < blockCount) {        
          CUDA_SYNC;
          gpuDownload(nWordsSub, C_d, &prev_C[0]);
          prevCBlockIdx = cBlockIdx;
        }
        else {
          assert(k == 0);
          break;
        }
      }
    }
  }
}



// data is assumed to have been arranged appropriately beforehand
static void multiGpuBooleanCubic(const uint4* A, const uint4* B, uint4* C, index_t n,
                                 index_t subn, uint4** A_ds, uint4** B_ds, uint4** C_ds) {
  
  // assume the matrix has shape 2^q*2^d*64 x 2^q*2^d*64; that is, the
  // matrix consist of 2^q * 2^q blocks of dimension 2^d*64 x 2^d*64 each
  // where 2^d*64 == subproblem size
  
  int d = 0;
  while ((1<<d)*64 < subn)
    ++d;
  assert((1<<d)*64 == subn);
  int q = 0;
  while ((1<<q)*(1<<d)*64 < n)
    ++q;
  assert((1<<q)*(1<<d)*64 == n);
  
  // number of uint4s required to represent the full matrix
  index_t nWordsFull = (1<<2*(d+q))*32;
  // number of uint4s required to represent the submatrix
  index_t nWordsSub = (1<<2*d)*32;

  vector< vector<uint4> > prev_Cs; // previous result
  for (int i = 0; i < GPU_COUNT; ++i)
    prev_Cs.push_back(vector<uint4>(nWordsSub));

  int blockCount = 1<<2*q;

  // initially set the entire output matrix to zero
  for (index_t i = 0; i < nWordsFull; ++i)
    C[i] = make_uint4(0,0,0,0);

  int totalThreads = 32*(1<<2*d);
  int blockSize, numBlocks;
  determineBlockSizeAndNumBlocks(totalThreads, numBlocks, blockSize);

  int64_t nanosPerThread[8];
  int64_t wordsPerThread[8];
  
  #pragma omp parallel for
  for (int gpu = 0; gpu < GPU_COUNT; ++gpu) {
    cudaSetDevice(gpu);

    uint4* A_d = A_ds[gpu];
    uint4* B_d = B_ds[gpu];
    uint4* C_d = C_ds[gpu];
    vector<uint4>& prev_C = prev_Cs[gpu];

    nanosPerThread[gpu] = 0;
    wordsPerThread[gpu] = 0;
    
    int prevCBlockIdx = -1;
    for (int cBlockIdx = gpu; cBlockIdx < blockCount+GPU_COUNT;
         cBlockIdx += GPU_COUNT) {
      int cBlockIdx_i = cBlockIdx/(1<<q);
      int cBlockIdx_j = cBlockIdx%(1<<q);
      for (int k = 0; k < (1<<q); ++k) {        
        if (cBlockIdx < blockCount) {
          gpuUpload(nWordsSub, &A[(cBlockIdx_i*(1<<q)+k)*nWordsSub], A_d);
          gpuUpload(nWordsSub, &B[(cBlockIdx_j*(1<<q)+k)*nWordsSub], B_d);
          cudaBooleanCubicMultiplicationKernel<<<numBlocks, blockSize>>>(A_d, B_d, C_d, d);
        }
        if (prevCBlockIdx >= 0) {
          // integrate the previous result
          auto start = now();
          for (int64_t w = 0; w < nWordsSub; ++w) {
            C[prevCBlockIdx*nWordsSub+w] = C[prevCBlockIdx*nWordsSub+w] | prev_C[w];
          }
          auto end = now();
          nanoseconds diff = end-start;
          nanosPerThread[gpu] += diff.count();
          wordsPerThread[gpu] += nWordsSub;
        }
        if (cBlockIdx < blockCount) {        
          CUDA_SYNC;
          gpuDownload(nWordsSub, C_d, &prev_C[0]);
          prevCBlockIdx = cBlockIdx;
        }
        else {
          assert(k == 0);
          break;
        }
      }
    }
  }
}



// simply copies data
__global__
void cudaCopyKernel(const uint4* src, uint4* dst) {
  int t = blockDim.x*blockIdx.x+threadIdx.x;
  dst[t] = src[t];
}



// assume n = 16 * 2^d * 64
// cpu scratch required: 24*(1 << 20)*32 words
static void multiGpuPipelinedDouble(int64_t n, const uint4* A,
                                    const uint4* B, uint4* C,
                                    uint4* cpuScratch, uint4** A_ds,
                                    uint4** B_ds, uint4** C_ds,
                                    uint4** scratch_ds,
                                    const int(*tVectors)[4],
                                    const int(*sVectors)[4],
                                    const int(*qVectors)[4],
                                    single_gpu_mm_fun_t singleGpuMM, int rank,
                                    int64_t subn, int numberOfLanes,
                                    int numberOfAuxiliaryMatrices) {
  const int MAXIMUM_NUMBER_OF_LANES = 32;
  const int MAXIMUM_NUMBER_OF_AUXILIARY_MATRICES = 8;
  assert(numberOfLanes <= MAXIMUM_NUMBER_OF_LANES);
  assert(numberOfAuxiliaryMatrices <= MAXIMUM_NUMBER_OF_AUXILIARY_MATRICES);
  assert(numberOfLanes % 8 == 0);
  assert(numberOfLanes/8*numberOfAuxiliaryMatrices <= 8);
  
  assert(GPU_COUNT == 8);
  assert(subn >= 64);
  assert(isPowerOfTwo(n));
  assert(isPowerOfTwo(subn));
  const int64_t d = intlog2(subn)-6;
  const int64_t q = intlog2(n)-d-6;
  const int64_t submatrixSize = (1L<<2*d)*32;
  assert(q == 3 || q == 4);

  const int EMPTY = -1;
  const int LAST_T_THREAD = numberOfLanes;
  const int LAST_S_THREAD = 2*numberOfLanes;
  const int LAST_MM_THREAD = 3*numberOfLanes;
  const int LAST_INTEGRATION_THREAD = 4*numberOfLanes;
  const int TOTAL_NUMBER_OF_THREADS = LAST_INTEGRATION_THREAD;

  uint4* Ts[MAXIMUM_NUMBER_OF_LANES][MAXIMUM_NUMBER_OF_AUXILIARY_MATRICES];
  uint4* Ss[MAXIMUM_NUMBER_OF_LANES][MAXIMUM_NUMBER_OF_AUXILIARY_MATRICES];
  uint4* Qs[MAXIMUM_NUMBER_OF_LANES][MAXIMUM_NUMBER_OF_AUXILIARY_MATRICES];
  std::atomic<int> Tstatus[MAXIMUM_NUMBER_OF_LANES][MAXIMUM_NUMBER_OF_AUXILIARY_MATRICES];
  std::atomic<int> Sstatus[MAXIMUM_NUMBER_OF_LANES][MAXIMUM_NUMBER_OF_AUXILIARY_MATRICES];
  std::atomic<int> Qstatus[MAXIMUM_NUMBER_OF_LANES][MAXIMUM_NUMBER_OF_AUXILIARY_MATRICES];
  for (int i = 0; i < MAXIMUM_NUMBER_OF_LANES; ++i) {
    for (int j = 0; j < MAXIMUM_NUMBER_OF_AUXILIARY_MATRICES; ++j) {
      Ts[i][j] = nullptr;
      Ss[i][j] = nullptr;
      Qs[i][j] = nullptr;
      Tstatus[i][j] = -2;
      Sstatus[i][j] = -2;
      Qstatus[i][j] = -2;
    }
  }

  for (int i = 0; i < numberOfLanes; ++i) {
    for (int j = 0; j < numberOfAuxiliaryMatrices; ++j) {
      Ts[i][j] = cpuScratch + (i*numberOfAuxiliaryMatrices+j)*submatrixSize;
      Ss[i][j] = cpuScratch + ((i+numberOfLanes)*numberOfAuxiliaryMatrices+j)*submatrixSize;
      Qs[i][j] = cpuScratch + ((i+2*numberOfLanes)*numberOfAuxiliaryMatrices+j)*submatrixSize;
      Tstatus[i][j] = EMPTY;
      Sstatus[i][j] = EMPTY;
      Qstatus[i][j] = EMPTY;
    }
  }

  // total number of submatrices: 256
  // iterate over 7^4 times and select the appropriate submatrices
  bool firstC[256];
  for (int i = 0; i < 256; ++i)
    firstC[i] = true;

  std::array<std::mutex,256> cMutexes;
  std::array<std::mutex,8> gpuMutexes;
  std::array<std::mutex,MAXIMUM_NUMBER_OF_LANES> laneMutexes;
  std::array<std::condition_variable,MAXIMUM_NUMBER_OF_LANES> laneCvs;

  int64_t nanosPerThread[TOTAL_NUMBER_OF_THREADS];
  int64_t wordsPerThread[TOTAL_NUMBER_OF_THREADS];

#pragma omp parallel
  {
#pragma omp for
    for (int t = 0; t < TOTAL_NUMBER_OF_THREADS; ++t) {
      // threads 0..7 create T
      // threads 8..15 create S
      // threads 16..23 perform upload & mm & download
      // threads 24..31 perform result integration

      nanosPerThread[t] = 0;
      wordsPerThread[t] = 0;
      
      int gpu = t % 8;
      int lane = t % numberOfLanes;
      cudaSetDevice(gpu);
      uint4* A_d = A_ds[gpu];
      uint4* B_d = B_ds[gpu];
      uint4* C_d = C_ds[gpu];
      uint4* scratch_d = scratch_ds[gpu];
      int rankp2 = rank*rank;
      int rankp3 = rankp2*rank;
      int max_i = (q == 3) ? rankp3 : rankp3*rank;
      for (int i = lane; i < max_i; i += numberOfLanes) {
        int auxiliaryMatrixNumber = (i/numberOfLanes)%numberOfAuxiliaryMatrices;
        uint4* T = Ts[lane][auxiliaryMatrixNumber];
        uint4* S = Ss[lane][auxiliaryMatrixNumber];
        uint4* Q = Qs[lane][auxiliaryMatrixNumber];
        
        // base r representation
        //int i3 = i/343;
        //int i2 = (i - i3*343)/49;
        //int i1 = (i - i3*343 - i2*49)/7;
        //int i0 = i%7;
        int i3 = i/rankp3;
        int i2 = (i - i3*rankp3)/rankp2;
        int i1 = (i - i3*rankp3 - i2*rankp2)/rank;
        int i0 = i%rank;

        // construct operands
        if (t < LAST_T_THREAD) {
          {
            std::unique_lock<std::mutex> lk(laneMutexes[lane]);
            laneCvs[lane].wait(lk, [&]() {
                return Tstatus[lane][auxiliaryMatrixNumber] == EMPTY;
              });
          }
          bool firstT = true;
          auto start = std::chrono::steady_clock::now();
          for (int j = 0; j < 64*(q == 4 ? 4 : 1); ++j) {
            int j3 = (j>>6)&3;
            int j2 = (j>>4)&3;
            int j1 = (j>>2)&3;
            int j0 = j&3;
            if ((q == 3 || tVectors[i3][j3]) &&
                tVectors[i2][j2] && tVectors[i1][j1] && tVectors[i0][j0]) {
              const uint4* subA = A + j*submatrixSize;
              if (firstT) {
                avxMov(subA, T, submatrixSize);
                firstT = false;
                wordsPerThread[t] += 2*submatrixSize;
              }
              else {
                avxXor(subA, T, T, submatrixSize);
                wordsPerThread[t] += 3*submatrixSize;
              }
            }
          }
          {
            std::lock_guard<std::mutex> lk(laneMutexes[lane]);
            Tstatus[lane][auxiliaryMatrixNumber] = i;
          }
          laneCvs[lane].notify_all();
          auto end = std::chrono::steady_clock::now();
          std::chrono::nanoseconds diff = end-start;
          nanosPerThread[t] += diff.count();
        }
        else if (t < LAST_S_THREAD) {
          {
            std::unique_lock<std::mutex> lk(laneMutexes[lane]);
            laneCvs[lane].wait(lk, [&]() {
                return Sstatus[lane][auxiliaryMatrixNumber] == EMPTY;
              });
          }
          bool firstS = true;
          auto start = std::chrono::steady_clock::now();
          for (int j = 0; j < 64*(q == 4 ? 4 : 1); ++j) {
            int j3 = (j>>6)&3;
            int j2 = (j>>4)&3;
            int j1 = (j>>2)&3;
            int j0 = j&3;
            assert(j0 + 4*j1 + 16*j2 + 64*j3 == j);
            
            if ((q == 3 || sVectors[i3][j3]) &&
                sVectors[i2][j2] && sVectors[i1][j1] && sVectors[i0][j0]) {
              const uint4* subB = B + j*submatrixSize;
              if (firstS) {
                avxMov(subB, S, submatrixSize);
                wordsPerThread[t] += 2*submatrixSize;
                firstS = false;
              }
              else {
                avxXor(subB, S, S, submatrixSize);
                wordsPerThread[t] += 3*submatrixSize;
              }
            }
          }
          {
            std::lock_guard<std::mutex> lk(laneMutexes[lane]);
            Sstatus[lane][auxiliaryMatrixNumber] = i;
          }
          laneCvs[lane].notify_all();
          auto end = std::chrono::steady_clock::now();
          std::chrono::nanoseconds diff = end-start;
          nanosPerThread[t] += diff.count();
        }
        else if (t < LAST_MM_THREAD) {
          // T upload
          {
            std::unique_lock<std::mutex> lk(laneMutexes[lane]);
            laneCvs[lane].wait(lk, [&]() {
                return Tstatus[lane][auxiliaryMatrixNumber] == i &&
                  Sstatus[lane][auxiliaryMatrixNumber] == i &&
                  Qstatus[lane][auxiliaryMatrixNumber] == EMPTY;
              });
          }
          {
            std::lock_guard<std::mutex> lock(gpuMutexes[gpu]); 
            auto start = std::chrono::steady_clock::now();

            gpuUpload(submatrixSize, T, A_d);
            {
              std::lock_guard<std::mutex> lk(laneMutexes[lane]);
              Tstatus[lane][auxiliaryMatrixNumber] = EMPTY;
            }
            laneCvs[lane].notify_all();            

            gpuUpload(submatrixSize, S, B_d);
            {
              std::lock_guard<std::mutex> lk(laneMutexes[lane]);
              Sstatus[lane][auxiliaryMatrixNumber] = EMPTY;
            }
            laneCvs[lane].notify_all();            

            singleGpuMM(A_d, B_d, C_d, scratch_d, subn);

            gpuDownload(submatrixSize, C_d, Q);
            {
              std::lock_guard<std::mutex> lk(laneMutexes[lane]);
              Qstatus[lane][auxiliaryMatrixNumber] = i;
            }
            laneCvs[lane].notify_all();            

            auto end = std::chrono::steady_clock::now();
            std::chrono::nanoseconds diff = end-start;
            nanosPerThread[t] += diff.count();
            wordsPerThread[t] += 3*submatrixSize;
          }
        }
        else if (t < LAST_INTEGRATION_THREAD) {
          {
            std::unique_lock<std::mutex> lk(laneMutexes[lane]);
            laneCvs[lane].wait(lk, [&]() {
                return Qstatus[lane][auxiliaryMatrixNumber] == i;
              });
          }

          auto start = std::chrono::steady_clock::now();
          for (int j = 0; j < 64*(q == 4 ? 4 : 1); ++j) {
            int j3 = (j>>6)&3;
            int j2 = (j>>4)&3;
            int j1 = (j>>2)&3;
            int j0 = j&3;
            if ((q == 3 || qVectors[i3][j3]) &&
                qVectors[i2][j2] && qVectors[i1][j1] && qVectors[i0][j0]) {
              std::lock_guard<std::mutex> lock(cMutexes[j]);
              uint4* subC = C + j*submatrixSize;
              bool first = false;
#pragma omp critical
              {
                first = firstC[j];
                if (first)
                  firstC[j] = false;
              }
              if (first) {
                avxMov(Q, subC, submatrixSize);
                wordsPerThread[t] += 2*submatrixSize;
              }
              else {
                avxXor(Q, subC, subC, submatrixSize);
                wordsPerThread[t] += 3*submatrixSize;                   
              }
            }
          }
          {
            std::lock_guard<std::mutex> lk(laneMutexes[lane]);
            Qstatus[lane][auxiliaryMatrixNumber] = EMPTY;
          }
          laneCvs[lane].notify_all();
          auto end = std::chrono::steady_clock::now();
          std::chrono::nanoseconds diff = end-start;
          nanosPerThread[t] += diff.count();
        }
        else {
          assert(false && "invalid number of threads");
        }
      }
    }
  }
}



static void multiGpuAlternateBasisDoubleStrassen(int64_t n, const uint4* A,
                                                 const uint4* B, uint4* C,
                                                 uint4* cpuScratch,
                                                 uint4** A_ds,
                                                 uint4** B_ds, uint4** C_ds,
                                                 uint4** scratch_ds,
                                                 int numberOfLanes,
                                                 int numberOfAuxiliaryMatrices) {
  multiGpuPipelinedDouble(n,A,B,C,cpuScratch,A_ds,B_ds,C_ds,scratch_ds,
                          STRASSEN_ALTERNATE_BASIS_T_VECTORS,
                          STRASSEN_ALTERNATE_BASIS_S_VECTORS,
                          STRASSEN_ALTERNATE_BASIS_Q_VECTORS,
                          cudaRecursiveAlternateBasisDoubleStrassen,7,n>>4,
                          numberOfLanes, numberOfAuxiliaryMatrices);
}



static void multiGpuAlternateBasisDoubleSix(int64_t n, const uint4* A,
                                            const uint4* B, uint4* C,
                                            uint4* cpuScratch, uint4** A_ds,
                                            uint4** B_ds, uint4** C_ds,
                                            uint4** scratch_ds,
                                            int numberOfLanes,
                                            int numberOfAuxiliaryMatrices) {
  multiGpuPipelinedDouble(n,A,B,C,cpuScratch,A_ds,B_ds,C_ds,scratch_ds,
                          SIX_ALTERNATE_BASIS_T_VECTORS,
                          SIX_ALTERNATE_BASIS_S_VECTORS,
                          SIX_ALTERNATE_BASIS_Q_VECTORS,
                          cudaRecursiveAlternateBasisDoubleSix,6,n>>3,
                          numberOfLanes, numberOfAuxiliaryMatrices);
}

  

static void test() {
  index_t cpuScratchSize = 192L*(1L<<20)*32;
  pushTime();
  uint4* cpuScratch = allocateAlignedMemory(cpuScratchSize);
  avxSet(cpuScratch, cpuScratchSize, 0);
  pushTime();
  cout << "cpu scratch allocation and zeroing took " << nanosToString(getDiffNanos()) << endl;

  pushTime();
  uint4* gpuMemory[8];
  index_t gpuMemorySize = 32L*(3*(1L<<22) + 3*intpow(6,8) + 8*((1L<<16) + (1L<<18) + (1L<<20)));
  uint4* A_ds[8];
  uint4* B_ds[8];
  uint4* C_ds[8];
  uint4* scratch_ds[8];
  for (int i = 0; i < 8; ++i) {
    cudaSetDevice(i);
    gpuMemory[i] = gpuAllocate(gpuMemorySize);
  }
  pushTime();
  cout << "gpu memory allocation and zeroing took " << nanosToString(getDiffNanos()) << endl;

  for (int64_t d = 0; d < 11; ++d) {
    int64_t n = (1<<d)*64;

    int64_t nWordsGpu = (1L<<2*10)*32;
    for (int i = 0; i < 8; ++i) {
      A_ds[i] = gpuMemory[i];
      B_ds[i] = gpuMemory[i]+nWordsGpu;
      C_ds[i] = gpuMemory[i]+2*nWordsGpu;
      scratch_ds[i] = gpuMemory[i]+3*nWordsGpu;
    }
    
    pushTime();
    BinaryMatrix A = BinaryMatrix::random(n,n);
    BinaryMatrix B = BinaryMatrix::random(n,n);
    BinaryMatrix A_bool = BinaryMatrix::booleanRandom(n);
    BinaryMatrix B_bool = BinaryMatrix::booleanRandom(n);
    BinaryMatrix C(n,n);
    BinaryMatrix A_a(n,n);
    BinaryMatrix B_a(n,n);
    BinaryMatrix C_a(n,n);
    BinaryMatrix A_c(n,n);
    BinaryMatrix B_c(n,n);
    BinaryMatrix C_c(n,n);
    BinaryMatrix A_ab(n,n);
    BinaryMatrix B_ab(n,n);
    BinaryMatrix C_ab(n,n);
    BinaryMatrix C_fh(n,n);
    pushTime();
    cout << "data generation " << n << "x" << n << " took " << nanosToString(getDiffNanos()) << endl;

    pushTime();
    arrangeDataCpuWords(A.getDataPointer(), A_a.getDataPointer(), n*n);
    arrangeDataCpuWords(B.getDataPointer(), B_a.getDataPointer(), n*n);
    pushTime();
    cout << "data arrangement " << n << "x" << n << " took " << nanosToString(getDiffNanos()) << endl;
    
    pushTime();
    changeOfBasisStrassen(A_a.getDataPointer(), A_ab.getDataPointer(), n);
    changeOfBasisStrassenAvxParallel(B_a.getDataPointer(), B_ab.getDataPointer(), n);
    pushTime();
    cout << "change of basis " << n << "x" << n << " took " << nanosToString(getDiffNanos()) << endl;

    if (d < 4) {
      pushTime();
      cudaSetDevice(0);
      gpuUpload((1<<2*d)*32, A_ab.getDataPointer(), A_ds[0]);
      gpuUpload((1<<2*d)*32, B_ab.getDataPointer(), B_ds[0]);
      cudaRecursiveAlternateBasisDoubleStrassen(A_ds[0], B_ds[0], C_ds[0], scratch_ds[0], n);
      gpuDownload((1<<2*d)*32, C_ds[0], C_ab.getDataPointer());
      pushTime();
      cout << "cudaRecursiveAlternateBasisDoubleStrassen " << n << "x" << n
           << " took " << nanosToString(getDiffNanos()) << endl;
    }
    else {
      pushTime();
      multiGpuAlternateBasisDoubleStrassen(n, A_ab.getDataPointer(),
                                           B_ab.getDataPointer(),
                                           C_ab.getDataPointer(), cpuScratch,
                                           A_ds, B_ds, C_ds, scratch_ds,
                                           8, 4);
      pushTime();
      cout << "multiGpuAlternateBasisDoubleStrassen " << n << "x" << n
           << " with 8 lanes and 4 auxmats took " << nanosToString(getDiffNanos())
           << endl;
    }
    
    
    pushTime();
    inverseChangeOfBasisStrassen(C_ab.getDataPointer(), C_a.getDataPointer(), n);
    pushTime();
    cout << "inverse change of basis " << n << "x" << n << " took " << nanosToString(getDiffNanos()) << endl;

    pushTime();
    dearrangeDataCpuWords(C_a.getDataPointer(), C.getDataPointer(), n*n);
    pushTime();
    cout << "data dearrangement " << n << "x" << n << " took " << nanosToString(getDiffNanos()) << endl;
    
    pushTime();
    C_fh = A.dotWords(B);
    pushTime();
    cout << "A.dotWords(B) " << n << "x" << n << " took "
         << nanosToString(getDiffNanos()) << endl;

    assert(C == C_fh);

    cout << "alternate basis strassen " << n << "x" << n << " ok" << endl;
    //////////////////////////////////////////

    nWordsGpu = (1L<<22)*32;
    for (int i = 0; i < 8; ++i) {
      A_ds[i] = gpuMemory[i];
      B_ds[i] = gpuMemory[i]+nWordsGpu;
      C_ds[i] = gpuMemory[i]+2*nWordsGpu;
      scratch_ds[i] = gpuMemory[i]+3*nWordsGpu;
    }

    int64_t subn = n >> 3;
    if (subn < 64) {
      pushTime();
      arrangeDataForCubicCpu(A.getDataPointer(), A_c.getDataPointer(), n*n);
      arrangeDataForCubicCpu(B.getDataPointer(), B_c.getDataPointer(), n*n);
      pushTime();
      cout << "cubic data arrangement " << n << "x" << n << " took " << nanosToString(getDiffNanos()) << endl;

      cudaSetDevice(0);
      pushTime();
      gpuUpload((1<<2*d)*32, A_c.getDataPointer(), A_ds[0]);
      gpuUpload((1<<2*d)*32, B_c.getDataPointer(), B_ds[0]);
      cudaCubicMultiplication(A_ds[0], B_ds[0], C_ds[0], nullptr, n);
      gpuDownload((1<<2*d)*32, C_ds[0], C_c.getDataPointer());
      pushTime();
      cout << "cudaCubicMultiplication " << n << "x" << n << " took " << nanosToString(getDiffNanos()) << endl;

      pushTime();
      dearrangeDataForCubicCpu(C_c.getDataPointer(), C.getDataPointer(), n*n);
      pushTime();
      cout << "cubic data dearrangement " << n << "x" << n << " took " << nanosToString(getDiffNanos()) << endl;

      assert(C_fh == C);

      cout << "cuda cubic multiplication " << n << "x" << n << " ok" << endl;
    }
    else {
      pushTime();
      arrangeDataForMultiGpuCubicCpu(A.getDataPointer(), A_c.getDataPointer(), 3, d-3);
      arrangeDataForMultiGpuCubicCpu(B.getDataPointer(), B_c.getDataPointer(), 3, d-3);
      pushTime();
      cout << "multigpu cubic data arrangement " << n << "x" << n << " took " << nanosToString(getDiffNanos()) << endl;

      pushTime();
      multiGpuCubic(A_c.getDataPointer(), B_c.getDataPointer(),
                    C_c.getDataPointer(), n, subn, A_ds, B_ds, C_ds);
      pushTime();
      cout << "multiGpuCubic " << n << "x" << n << " took " << nanosToString(getDiffNanos()) << endl;

      pushTime();
      dearrangeDataForMultiGpuCubicCpu(C_c.getDataPointer(), C.getDataPointer(), 3, d-3);
      pushTime();
      cout << "multigpu cubic data dearrangement " << n << "x" << n << " took " << nanosToString(getDiffNanos()) << endl;

      assert(C_fh == C);
      
      cout << "multigpu cubic " << n << "x" << n << " ok" << endl;
    }
    //////////////////////////////////////////

    pushTime();
    C_fh = A_bool.booleanDotWords(B_bool);
    pushTime();
    cout << "A_bool.booleanDotWords(B_bool) " << n << "x" << n << " took "
         << nanosToString(getDiffNanos()) << endl;

    if (subn < 64) {
      pushTime();
      arrangeDataForCubicCpu(A_bool.getDataPointer(), A_c.getDataPointer(), n*n);
      arrangeDataForCubicCpu(B_bool.getDataPointer(), B_c.getDataPointer(), n*n);
      pushTime();
      cout << "cubic data arrangement " << n << "x" << n << " took " << nanosToString(getDiffNanos()) << endl;

      cudaSetDevice(0);
      pushTime();
      gpuUpload((1<<2*d)*32, A_c.getDataPointer(), A_ds[0]);
      gpuUpload((1<<2*d)*32, B_c.getDataPointer(), B_ds[0]);
      cudaBooleanCubicMultiplication(A_ds[0], B_ds[0], C_ds[0], nullptr, n);
      gpuDownload((1<<2*d)*32, C_ds[0], C_c.getDataPointer());
      pushTime();
      cout << "cudaBooleanCubicMultiplication " << n << "x" << n << " took "
           << nanosToString(getDiffNanos()) << endl;

      pushTime();
      dearrangeDataForCubicCpu(C_c.getDataPointer(), C.getDataPointer(), n*n);
      pushTime();
      cout << "cubic data dearrangement " << n << "x" << n << " took "
           << nanosToString(getDiffNanos()) << endl;

      assert(C_fh == C);

      cout << "cuda boolean cubic multiplication " << n << "x" << n << " ok" << endl;
    }
    else {
      subn = n;
      pushTime();
      arrangeDataForMultiGpuCubicCpu(A_bool.getDataPointer(), A_c.getDataPointer(), 0, d);
      arrangeDataForMultiGpuCubicCpu(B_bool.getDataPointer(), B_c.getDataPointer(), 0, d);
      pushTime();
      
      cout << "multigpu cubic data arrangement " << n << "x" << n << " took " << nanosToString(getDiffNanos()) << endl;
      pushTime();
      multiGpuBooleanCubic(A_c.getDataPointer(), B_c.getDataPointer(),
                           C_c.getDataPointer(), n, subn, A_ds, B_ds, C_ds);
      pushTime();
      cout << "multiGpuBooleanCubic " << n << "x" << n << " took " << nanosToString(getDiffNanos()) << endl;

      pushTime();
      dearrangeDataForMultiGpuCubicCpu(C_c.getDataPointer(), C.getDataPointer(), 0, d);
      pushTime();
      cout << "multigpu cubic data dearrangement " << n << "x" << n << " took " << nanosToString(getDiffNanos()) << endl;

      assert(C_fh == C);
      
      cout << "multigpu boolean cubic " << n << "x" << n << " ok" << endl;
    }

    //////////////////////////////////////////
    nWordsGpu = (1L<<2*11)*32;
    for (int i = 0; i < 8; ++i) {
      A_ds[i] = gpuMemory[i];
      B_ds[i] = gpuMemory[i]+nWordsGpu;
      C_ds[i] = gpuMemory[i]+2*nWordsGpu;
      scratch_ds[i] = gpuMemory[i]+3*nWordsGpu;
    }

    if (d < 3) {
      pushTime();
      cudaSetDevice(0);
      gpuUpload((1<<2*d)*32, A_ab.getDataPointer(), A_ds[0]);
      gpuUpload((1<<2*d)*32, B_ab.getDataPointer(), B_ds[0]);
      cudaRecursiveAlternateBasisDoubleSix(A_ds[0], B_ds[0], C_ds[0], scratch_ds[0], n);
      gpuDownload((1<<2*d)*32, C_ds[0], C_ab.getDataPointer());
      pushTime();
      cout << "cudaRecursiveAlternateBasisDoubleSix " << n << "x" << n
           << " took " << nanosToString(getDiffNanos()) << endl;
  
    }
    else {
      pushTime();
      multiGpuAlternateBasisDoubleSix(n, A_ab.getDataPointer(),
                                      B_ab.getDataPointer(),
                                      C_ab.getDataPointer(), cpuScratch,
                                      A_ds, B_ds, C_ds, scratch_ds,
                                      8, 2);
      pushTime();
      cout << "multiGpuAlternateBasisDoubleSix with 8 lanes and 2 auxmat took "
           << nanosToString(getDiffNanos()) << endl;
    }

    pushTime();
    inverseChangeOfBasisStrassen(C_ab.getDataPointer(), C_a.getDataPointer(), n);
    pushTime();
    cout << "inverse change of basis " << n << "x" << n << " took " << nanosToString(getDiffNanos()) << endl;
   
    pushTime();
    C_fh = A_a.furHatSix(B_a);
    pushTime();
    cout << "A_a.furHatSix(B_a) " << n << "x" << n << " took "
         << nanosToString(getDiffNanos()) << endl;

    if (d == 0) {
      dearrangeDataCpuWords(C_fh.getDataPointer(), C.getDataPointer(), n*n);
      assert(A.dot(B) == C);
    }

    assert(C_a == C_fh);
    
    cout << "alternate basis six " << n << "x" << n << " ok" << endl;    
  }
  free(cpuScratch);
  for (int i = 0; i < 8; ++i) {
    gpuFree(gpuMemory[i]);
  }
}

static void evaluate() {
  int maxD = 15;
  index_t cpuScratchSize = 192L*(1L<<20)*32;
  pushTime();
  uint4* cpuScratch = allocateAlignedMemory(cpuScratchSize);
  avxSet(cpuScratch, cpuScratchSize, 0);
  pushTime();
  cout << "cpu scratch allocation and zeroing took " << nanosToString(getDiffNanos()) << endl;

  pushTime();
  uint4* gpuMemory[8];
  index_t gpuMemorySize = 32L*(3*(1L<<22) + 3*intpow(6,8) + 8*((1L<<16) + (1L<<18) + (1L<<20)));
  uint4* A_ds_cubic[8];
  uint4* B_ds_cubic[8];
  uint4* C_ds_cubic[8];

  uint4* A_ds_strassen[8];
  uint4* B_ds_strassen[8];
  uint4* C_ds_strassen[8];
  uint4* scratch_ds_strassen[8];

  uint4* A_ds_six[8];
  uint4* B_ds_six[8];
  uint4* C_ds_six[8];
  uint4* scratch_ds_six[8];
  
  for (int i = 0; i < 8; ++i) {
    cudaSetDevice(i);
    gpuMemory[i] = gpuAllocate(gpuMemorySize);

    index_t maxCubicOperandSize = (1L<<22)*32;
    index_t maxSixOperandSize = (1L<<22)*32;
    index_t maxStrassenOperandSize = (1L<<20)*32;
    
    A_ds_cubic[i] = gpuMemory[i];
    B_ds_cubic[i] = gpuMemory[i]+maxCubicOperandSize;
    C_ds_cubic[i] = gpuMemory[i]+2*maxCubicOperandSize;
    
    A_ds_strassen[i] = gpuMemory[i];
    B_ds_strassen[i] = gpuMemory[i]+maxStrassenOperandSize;
    C_ds_strassen[i] = gpuMemory[i]+2*maxStrassenOperandSize;
    scratch_ds_strassen[i] = gpuMemory[i]+3*maxStrassenOperandSize;

    A_ds_six[i] = gpuMemory[i];
    B_ds_six[i] = gpuMemory[i]+maxSixOperandSize;
    C_ds_six[i] = gpuMemory[i]+2*maxSixOperandSize;
    scratch_ds_six[i] = gpuMemory[i]+3*maxSixOperandSize;    
  }
  pushTime();
  cout << "gpu memory allocation and zeroing took " << nanosToString(getDiffNanos()) << endl;

  // the first one may not be ok so leaving it out gives an odd number
  // for median calculation
  int NREPEATS = 6; 

  // cubic
  for (int64_t d = 0; d < maxD; ++d) {
    int64_t n = (1<<d)*64;
    int64_t subn = n >> 3;

    pushTime();
    BinaryMatrix A = BinaryMatrix::random(n,n);
    BinaryMatrix B = BinaryMatrix::random(n,n);
    BinaryMatrix C(n,n);
    pushTime();
    cout << "data generation " << n << "x" << n << " took " << nanosToString(getDiffNanos()) << endl;

    if (d < 12) {
      for (int r = 0; r < NREPEATS; ++r) {
        cudaSetDevice(0);
        gpuUpload((1<<2*d)*32, A.getDataPointer(), A_ds_cubic[0]);
        gpuUpload((1<<2*d)*32, B.getDataPointer(), B_ds_cubic[0]);
        pushTime();
        cudaCubicMultiplication(A_ds_cubic[0], B_ds_cubic[0], C_ds_cubic[0], nullptr, n);
        pushTime();
        
        auto nanos = getDiffNanos();
        cout << "cudaCubicMultiplication " << n << "x" << n << " took " 
             << nanosToString(nanos) << " (" << computeEffectiveTbops(n,nanos)
             << " effective Tbops)" << endl;
      }
    }

    if (subn >= 64) {
      for (int r = 0; r < NREPEATS; ++r) {
        pushTime();
        multiGpuCubic(A.getDataPointer(), B.getDataPointer(), C.getDataPointer(),
                      n, subn, A_ds_cubic, B_ds_cubic, C_ds_cubic);
        pushTime();
        auto nanos = getDiffNanos();
        cout << "multiGpuCubic " << n << "x" << n << " (sub: "
             << subn << "x" << subn << ") took "
             << nanosToString(nanos) << " (" << computeEffectiveTbops(n,nanos)
             << " effective Tbops)" << endl;
      }
    }
  }

  // boolean cubic

  for (int64_t d = 0; d < maxD; ++d) {
    int64_t n = (1<<d)*64;
    int64_t subn = n >> 3;

    pushTime();
    BinaryMatrix A = BinaryMatrix::booleanRandom(n);
    BinaryMatrix B = BinaryMatrix::booleanRandom(n);
    BinaryMatrix C(n,n);
    pushTime();
    cout << "data generation " << n << "x" << n << " took " << nanosToString(getDiffNanos()) << endl;

    if (d < 12) {
      for (int r = 0; r < NREPEATS; ++r) {
        cudaSetDevice(0);
        gpuUpload((1<<2*d)*32, A.getDataPointer(), A_ds_cubic[0]);
        gpuUpload((1<<2*d)*32, B.getDataPointer(), B_ds_cubic[0]);
        pushTime();
        cudaBooleanCubicMultiplication(A_ds_cubic[0], B_ds_cubic[0], C_ds_cubic[0], nullptr, n);
        pushTime();

        auto nanos = getDiffNanos();
        cout << "cudaBooleanCubicMultiplication " << n << "x" << n << " took "
             << nanosToString(nanos) << " (" << computeEffectiveTbops(n,nanos)
             << " effective Tbops)" << endl;
      }
    }

    if (subn >= 64) {
      for (int r = 0; r < NREPEATS; ++r) {
        pushTime();
        multiGpuBooleanCubic(A.getDataPointer(), B.getDataPointer(), C.getDataPointer(),
                             n, subn, A_ds_cubic, B_ds_cubic, C_ds_cubic);
        pushTime();
        auto nanos = getDiffNanos();
        cout << "multiGpuBooleanCubic " << n << "x" << n << " (sub: "
             << subn << "x" << subn << ") took "
             << nanosToString(nanos) << " (" << computeEffectiveTbops(n,nanos)
             << " effective Tbops)" << endl;
      }
    }
  }

  // Strassen
  for (int64_t d = 0; d < maxD; ++d) {
    int64_t n = (1<<d)*64;
    int64_t subn = n >> 4;

    pushTime();
    BinaryMatrix A = BinaryMatrix::random(n);
    BinaryMatrix B = BinaryMatrix::random(n);
    BinaryMatrix C(n,n);
    pushTime();
    cout << "data generation " << n << "x" << n << " took " << nanosToString(getDiffNanos()) << endl;

    if (d < 11) {
      for (int r = 0; r < NREPEATS; ++r) {
        cudaSetDevice(0);
        gpuUpload((1<<2*d)*32, A.getDataPointer(), A_ds_cubic[0]);
        gpuUpload((1<<2*d)*32, B.getDataPointer(), B_ds_cubic[0]);
        pushTime();
        cudaRecursiveAlternateBasisDoubleStrassen(A_ds_strassen[0],
                                                  B_ds_strassen[0],
                                                  C_ds_strassen[0],
                                                  scratch_ds_strassen[0], n);
        pushTime();

        auto nanos = getDiffNanos();
        cout << "cudaRecursiveAlternateBasisDoubleStrassen " << n << "x" << n
             << " took " << nanosToString(nanos) << " ("
             << computeEffectiveTbops(n,nanos) << " effective Tbops)" << endl;
      }
    }
    
    if (subn >= 64) {
      for (int r = 0; r < NREPEATS; ++r) {
        pushTime();
        multiGpuAlternateBasisDoubleStrassen(n, A.getDataPointer(),
                                             B.getDataPointer(),
                                             C.getDataPointer(),
                                             cpuScratch, A_ds_strassen,
                                             B_ds_strassen, C_ds_strassen,
                                             scratch_ds_strassen, 8, 4);
        pushTime();
        auto nanos = getDiffNanos();
        cout << "multiGpuAlternateBasisDoubleStrassen " << n << "x" << n << " (sub: "
             << subn << "x" << subn << ", with 8 lanes and 4 auxmats) took "
             << nanosToString(nanos) << " (" << computeEffectiveTbops(n,nanos)
             << " effective Tbops)" << endl;
      }
    }
  }

  // Six

  for (int64_t d = 0; d < maxD; ++d) {
    int64_t n = (1<<d)*64;
    int64_t subn = n >> 3;

    pushTime();
    BinaryMatrix A = BinaryMatrix::random(n);
    BinaryMatrix B = BinaryMatrix::random(n);
    BinaryMatrix C(n,n);
    pushTime();
    cout << "data generation " << n << "x" << n << " took " << nanosToString(getDiffNanos()) << endl;

    if (d < 12) {
      for (int r = 0; r < NREPEATS; ++r) {
        cudaSetDevice(0);
        gpuUpload((1<<2*d)*32, A.getDataPointer(), A_ds_cubic[0]);
        gpuUpload((1<<2*d)*32, B.getDataPointer(), B_ds_cubic[0]);
        pushTime();
        cudaRecursiveAlternateBasisDoubleSix(A_ds_six[0],
                                             B_ds_six[0],
                                             C_ds_six[0],
                                             scratch_ds_six[0], n);
        pushTime();

        auto nanos = getDiffNanos();
        cout << "cudaRecursiveAlternateBasisDoubleSix " << n << "x" << n
             << " took " << nanosToString(nanos) << " ("
             << computeEffectiveTbops(n,nanos) << " effective Tbops)" << endl;
      }
    }
    
    if (subn >= 64) {
      for (int r = 0; r < NREPEATS; ++r) {
        pushTime();
        multiGpuAlternateBasisDoubleSix(n, A.getDataPointer(),
                                        B.getDataPointer(),
                                        C.getDataPointer(),
                                        cpuScratch, A_ds_six,
                                        B_ds_six, C_ds_six,
                                        scratch_ds_six, 8, 2);
        pushTime();
        auto nanos = getDiffNanos();
        cout << "multiGpuAlternateBasisDoubleSix " << n << "x" << n << " (sub: "
             << subn << "x" << subn << ", with 8 lanes and 2 auxmats) took "
             << nanosToString(nanos) << " (" << computeEffectiveTbops(n,nanos)
             << " effective Tbops)" << endl;
      }
    }
  }
  free(cpuScratch);
  for (int i = 0; i < 8; ++i) 
    gpuFree(gpuMemory[i]);
}




int main(int argc, char* argv[]) {
  cudaGetDeviceCount(&GPU_COUNT);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  if (prop.major < 7) {
    cerr << "ERROR: Compute capability 7.0 or higher is required" << endl;
    return EXIT_FAILURE;
  }

  int uvaEnabled;
  cuDeviceGetAttribute(&uvaEnabled, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, 0);
           
  if (!uvaEnabled) {
    cerr << "ERROR: UVA is disabled" << endl;
  }


  enableTimings();

  if (argc == 2 && string(argv[1]) == "test")
    test();
  else if (argc == 2 && string(argv[1]) == "evaluate")
    evaluate();
  else
    cout << "usage: " << argv[0] << " <test|evaluate>" << endl;

  disableTimings();

  return EXIT_SUCCESS;
}
