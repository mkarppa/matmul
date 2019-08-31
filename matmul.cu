#include <sys/utsname.h>
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
#include <iomanip>

using std::cout;
using std::cerr;
using std::endl;
using std::vector;
using std::chrono::nanoseconds;
using std::string;
using std::mutex;
using std::array;
using std::condition_variable;

// verification of certain assumptions
static_assert(std::is_same<unsigned int, decltype(uint4().x)>::value &&
              std::is_same<unsigned int, decltype(uint4().y)>::value &&
              std::is_same<unsigned int, decltype(uint4().z)>::value &&
              std::is_same<unsigned int, decltype(uint4().w)>::value &&
              offsetof(uint4,x) == 0 &&
              offsetof(uint4,y) == sizeof(unsigned int) &&
              offsetof(uint4,z) == 2*sizeof(unsigned int) &&
              offsetof(uint4,w) == 3*sizeof(unsigned int),
              "type assumptions do not hold: uint4 is not composed of "
              "4 consecutive unsigned ints");

// some common definitions
typedef std::chrono::time_point<std::chrono::steady_clock> TimePoint;
typedef int64_t index_t;   // 64-bit signed index type
typedef void (single_gpu_mm_fun_t)(const uint4*,  const uint4*, uint4*, uint4*,
                                   int);
typedef void (multi_gpu_mm_fun_t)(int64_t, int64_t, const uint4*,
                                  const uint4*, uint4*, uint4*, uint4**,
                                  uint4**, uint4**, uint4**, int, int);
typedef void (data_arrangement_fun_t)(const uint4*, uint4*, int64_t, int64_t);

// when evaluating, perform 5+1 repeats (the first one is discarded)
static const int NREPEATS = 6;

// define if we want a per-thread report on the amount of time worked
// and the number of words moved in multigpu pipeline
// #define DEBUG_REPORT_PER_THREAD_TIMES_AND_WORDS

static const int MAXIMUM_NUMBER_OF_LANES = 32;
static const int MAXIMUM_NUMBER_OF_AUXILIARY_MATRICES = 8;

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
static const int STRASSEN_WINOGRAD_T_VECTORS[7][4] = {
 // A00, A01, A10, A11
  { 0,   0,   1,   1 }, // T0
  { 0,   1,   0,   0 },
  { 0,   1,   0,   1 },
  { 0,   1,   1,   1 },
  { 1,   1,   1,   1 },
  { 0,   0,   1,   0 },
  { 1,   0,   0,   0 }  // T6
};

static const int ALTERNATIVE_BASIS_SELF_INVERSE_T_VECTORS[7][4] = {
 // A00, A01, A10, A11
  { 1,   0,   0,   0 }, // T0
  { 0,   1,   0,   0 },
  { 0,   0,   1,   0 },
  { 0,   0,   0,   1 },
  { 1,   0,   0,   1 },
  { 0,   1,   0,   1 },
  { 0,   0,   1,   1 }  // T6
};

static const int ALTERNATIVE_BASIS_CHAINING_T_VECTORS[7][4] = {
 // A00, A01, A10, A11
  { 1,   0,   0,   0 }, // T0
  { 0,   1,   0,   0 },
  { 0,   0,   1,   0 },
  { 0,   0,   0,   1 },
  { 1,   0,   1,   0 },
  { 0,   1,   1,   0 },
  { 0,   0,   1,   1 }  // T6
};

// 4-long vectors corresponding to which submatrices of B are XORed for each S
static const int STRASSEN_WINOGRAD_S_VECTORS[7][4] = {
 // B00, B10, B01, B11
  { 0,   1,   0,   1 }, // S0
  { 0,   1,   0,   0 },
  { 0,   0,   1,   1 },
  { 0,   1,   1,   1 },
  { 0,   0,   1,   0 },
  { 1,   1,   1,   1 },
  { 1,   0,   0,   0 }  // S6
};

static const int ALTERNATIVE_BASIS_SELF_INVERSE_S_VECTORS[7][4] = {
 // B00, B10, B01, B11
  { 1,   0,   0,   0 }, // S0
  { 0,   1,   0,   0 },
  { 1,   0,   0,   1 },
  { 0,   0,   0,   1 },
  { 0,   0,   1,   0 },
  { 0,   0,   1,   1 },
  { 0,   1,   0,   1 }  // S6
};

static const int ALTERNATIVE_BASIS_CHAINING_S_VECTORS[7][4] = {
 // B00, B10, B01, B11
  { 1,   0,   0,   0 }, // S0
  { 0,   1,   0,   1 },
  { 0,   1,   0,     },
  { 0,   0,   0,   1 },
  { 0,   0,   1,   0 },
  { 0,   1,   1,   0 },
  { 1,   1,   0,   0 }  // S6
};

// 4-long vectors corresponding to which submatrices of C are XORed for each Q
static const int STRASSEN_WINOGRAD_Q_VECTORS[7][4] = {
 // C00, C01, C10, C11
  { 0,   1,   0,   1 }, // Q0
  { 1,   1,   1,   1 },
  { 0,   0,   1,   1 },
  { 0,   1,   1,   1 },
  { 0,   1,   0,   0 },
  { 0,   0,   1,   0 },
  { 1,   0,   0,   0 }  // Q6
};

static const int ALTERNATIVE_BASIS_SELF_INVERSE_Q_VECTORS[7][4] = {
 // C00, C01, C10, C11
  { 1,   0,   0,   0 }, // Q0
  { 1,   0,   0,   1 },
  { 0,   0,   1,   0 },
  { 0,   0,   0,   1 },
  { 0,   1,   0,   0 },
  { 0,   0,   1,   1 },
  { 0,   1,   0,   1 }  // Q6
};

static const int ALTERNATIVE_BASIS_CHAINING_Q_VECTORS[7][4] = {
 // C00, C01, C10, C11
  { 1,   0,   0,   0 }, // Q0
  { 1,   1,   1,   0 },
  { 0,   1,   1,   0 },
  { 0,   0,   0,   1 },
  { 0,   1,   1,   1 },
  { 0,   1,   0,   0 },
  { 0,   0,   1,   0 }  // Q6
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



static int GPU_COUNT = -1;


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
  BinaryMatrix() : n(0), m(0), data(nullptr) {}
  
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
    size_t dataSize = n*m/128;
    avxMov(that.data, data, dataSize);
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

  bool operator!=(const BinaryMatrix& that) const {
    return !(*this == that);
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



// perform a naive transpose of a 64*64 matrix assuming the following bit order:
// u_4 ... u_0 u_5 v_5 ... v_0
// the transpose is computed naively and on a per-bit basis
static void transposeNaiveInplace64(uint4* A) {
  unsigned int* Ap = reinterpret_cast<unsigned int*>(A);
  unsigned int At[128];
  for (int i = 0; i < 128; ++i)
    At[i] = 0;
  for (int i = 0; i < 128; ++i) {
    unsigned int w = Ap[i];
    int jp = i >> 2;
    int ilow = ((i&1) << 1) | ((i >> 1)&1);
    for (int j = 0; j < 32; ++j) {
      // i = u_4 ... u_0 u_5 v_5
      // j = v_4 ... v_0
      int ip = (j << 2) | ilow;
      At[ip] |= ((w >> j) & 1) << jp;
    }
  }
  for (int i = 0; i < 128; ++i)
    Ap[i] = At[i];
}



// perform a single 64*64 transpose in-place (assuming non-rearranged data)
static void transposeInplaceNonarranged64(uint4* arr) {
    unsigned int* A = reinterpret_cast<unsigned int*>(arr);
    unsigned int B[128];
    unsigned int C[128];
    B[  0] = (A[  0] & 0x55555555) | ((A[  2] & 0x55555555) << 1);
    B[  1] = (A[  1] & 0x55555555) | ((A[  3] & 0x55555555) << 1);
    B[  2] = ((A[  0] & 0xAAAAAAAA) >> 1) | (A[  2] & 0xAAAAAAAA);
    B[  3] = ((A[  1] & 0xAAAAAAAA) >> 1) | (A[  3] & 0xAAAAAAAA);
    B[  4] = (A[  4] & 0x55555555) | ((A[  6] & 0x55555555) << 1);
    B[  5] = (A[  5] & 0x55555555) | ((A[  7] & 0x55555555) << 1);
    B[  6] = ((A[  4] & 0xAAAAAAAA) >> 1) | (A[  6] & 0xAAAAAAAA);
    B[  7] = ((A[  5] & 0xAAAAAAAA) >> 1) | (A[  7] & 0xAAAAAAAA);
    B[  8] = (A[  8] & 0x55555555) | ((A[ 10] & 0x55555555) << 1);
    B[  9] = (A[  9] & 0x55555555) | ((A[ 11] & 0x55555555) << 1);
    B[ 10] = ((A[  8] & 0xAAAAAAAA) >> 1) | (A[ 10] & 0xAAAAAAAA);
    B[ 11] = ((A[  9] & 0xAAAAAAAA) >> 1) | (A[ 11] & 0xAAAAAAAA);
    B[ 12] = (A[ 12] & 0x55555555) | ((A[ 14] & 0x55555555) << 1);
    B[ 13] = (A[ 13] & 0x55555555) | ((A[ 15] & 0x55555555) << 1);
    B[ 14] = ((A[ 12] & 0xAAAAAAAA) >> 1) | (A[ 14] & 0xAAAAAAAA);
    B[ 15] = ((A[ 13] & 0xAAAAAAAA) >> 1) | (A[ 15] & 0xAAAAAAAA);
    B[ 16] = (A[ 16] & 0x55555555) | ((A[ 18] & 0x55555555) << 1);
    B[ 17] = (A[ 17] & 0x55555555) | ((A[ 19] & 0x55555555) << 1);
    B[ 18] = ((A[ 16] & 0xAAAAAAAA) >> 1) | (A[ 18] & 0xAAAAAAAA);
    B[ 19] = ((A[ 17] & 0xAAAAAAAA) >> 1) | (A[ 19] & 0xAAAAAAAA);
    B[ 20] = (A[ 20] & 0x55555555) | ((A[ 22] & 0x55555555) << 1);
    B[ 21] = (A[ 21] & 0x55555555) | ((A[ 23] & 0x55555555) << 1);
    B[ 22] = ((A[ 20] & 0xAAAAAAAA) >> 1) | (A[ 22] & 0xAAAAAAAA);
    B[ 23] = ((A[ 21] & 0xAAAAAAAA) >> 1) | (A[ 23] & 0xAAAAAAAA);
    B[ 24] = (A[ 24] & 0x55555555) | ((A[ 26] & 0x55555555) << 1);
    B[ 25] = (A[ 25] & 0x55555555) | ((A[ 27] & 0x55555555) << 1);
    B[ 26] = ((A[ 24] & 0xAAAAAAAA) >> 1) | (A[ 26] & 0xAAAAAAAA);
    B[ 27] = ((A[ 25] & 0xAAAAAAAA) >> 1) | (A[ 27] & 0xAAAAAAAA);
    B[ 28] = (A[ 28] & 0x55555555) | ((A[ 30] & 0x55555555) << 1);
    B[ 29] = (A[ 29] & 0x55555555) | ((A[ 31] & 0x55555555) << 1);
    B[ 30] = ((A[ 28] & 0xAAAAAAAA) >> 1) | (A[ 30] & 0xAAAAAAAA);
    B[ 31] = ((A[ 29] & 0xAAAAAAAA) >> 1) | (A[ 31] & 0xAAAAAAAA);
    B[ 32] = (A[ 32] & 0x55555555) | ((A[ 34] & 0x55555555) << 1);
    B[ 33] = (A[ 33] & 0x55555555) | ((A[ 35] & 0x55555555) << 1);
    B[ 34] = ((A[ 32] & 0xAAAAAAAA) >> 1) | (A[ 34] & 0xAAAAAAAA);
    B[ 35] = ((A[ 33] & 0xAAAAAAAA) >> 1) | (A[ 35] & 0xAAAAAAAA);
    B[ 36] = (A[ 36] & 0x55555555) | ((A[ 38] & 0x55555555) << 1);
    B[ 37] = (A[ 37] & 0x55555555) | ((A[ 39] & 0x55555555) << 1);
    B[ 38] = ((A[ 36] & 0xAAAAAAAA) >> 1) | (A[ 38] & 0xAAAAAAAA);
    B[ 39] = ((A[ 37] & 0xAAAAAAAA) >> 1) | (A[ 39] & 0xAAAAAAAA);
    B[ 40] = (A[ 40] & 0x55555555) | ((A[ 42] & 0x55555555) << 1);
    B[ 41] = (A[ 41] & 0x55555555) | ((A[ 43] & 0x55555555) << 1);
    B[ 42] = ((A[ 40] & 0xAAAAAAAA) >> 1) | (A[ 42] & 0xAAAAAAAA);
    B[ 43] = ((A[ 41] & 0xAAAAAAAA) >> 1) | (A[ 43] & 0xAAAAAAAA);
    B[ 44] = (A[ 44] & 0x55555555) | ((A[ 46] & 0x55555555) << 1);
    B[ 45] = (A[ 45] & 0x55555555) | ((A[ 47] & 0x55555555) << 1);
    B[ 46] = ((A[ 44] & 0xAAAAAAAA) >> 1) | (A[ 46] & 0xAAAAAAAA);
    B[ 47] = ((A[ 45] & 0xAAAAAAAA) >> 1) | (A[ 47] & 0xAAAAAAAA);
    B[ 48] = (A[ 48] & 0x55555555) | ((A[ 50] & 0x55555555) << 1);
    B[ 49] = (A[ 49] & 0x55555555) | ((A[ 51] & 0x55555555) << 1);
    B[ 50] = ((A[ 48] & 0xAAAAAAAA) >> 1) | (A[ 50] & 0xAAAAAAAA);
    B[ 51] = ((A[ 49] & 0xAAAAAAAA) >> 1) | (A[ 51] & 0xAAAAAAAA);
    B[ 52] = (A[ 52] & 0x55555555) | ((A[ 54] & 0x55555555) << 1);
    B[ 53] = (A[ 53] & 0x55555555) | ((A[ 55] & 0x55555555) << 1);
    B[ 54] = ((A[ 52] & 0xAAAAAAAA) >> 1) | (A[ 54] & 0xAAAAAAAA);
    B[ 55] = ((A[ 53] & 0xAAAAAAAA) >> 1) | (A[ 55] & 0xAAAAAAAA);
    B[ 56] = (A[ 56] & 0x55555555) | ((A[ 58] & 0x55555555) << 1);
    B[ 57] = (A[ 57] & 0x55555555) | ((A[ 59] & 0x55555555) << 1);
    B[ 58] = ((A[ 56] & 0xAAAAAAAA) >> 1) | (A[ 58] & 0xAAAAAAAA);
    B[ 59] = ((A[ 57] & 0xAAAAAAAA) >> 1) | (A[ 59] & 0xAAAAAAAA);
    B[ 60] = (A[ 60] & 0x55555555) | ((A[ 62] & 0x55555555) << 1);
    B[ 61] = (A[ 61] & 0x55555555) | ((A[ 63] & 0x55555555) << 1);
    B[ 62] = ((A[ 60] & 0xAAAAAAAA) >> 1) | (A[ 62] & 0xAAAAAAAA);
    B[ 63] = ((A[ 61] & 0xAAAAAAAA) >> 1) | (A[ 63] & 0xAAAAAAAA);
    B[ 64] = (A[ 64] & 0x55555555) | ((A[ 66] & 0x55555555) << 1);
    B[ 65] = (A[ 65] & 0x55555555) | ((A[ 67] & 0x55555555) << 1);
    B[ 66] = ((A[ 64] & 0xAAAAAAAA) >> 1) | (A[ 66] & 0xAAAAAAAA);
    B[ 67] = ((A[ 65] & 0xAAAAAAAA) >> 1) | (A[ 67] & 0xAAAAAAAA);
    B[ 68] = (A[ 68] & 0x55555555) | ((A[ 70] & 0x55555555) << 1);
    B[ 69] = (A[ 69] & 0x55555555) | ((A[ 71] & 0x55555555) << 1);
    B[ 70] = ((A[ 68] & 0xAAAAAAAA) >> 1) | (A[ 70] & 0xAAAAAAAA);
    B[ 71] = ((A[ 69] & 0xAAAAAAAA) >> 1) | (A[ 71] & 0xAAAAAAAA);
    B[ 72] = (A[ 72] & 0x55555555) | ((A[ 74] & 0x55555555) << 1);
    B[ 73] = (A[ 73] & 0x55555555) | ((A[ 75] & 0x55555555) << 1);
    B[ 74] = ((A[ 72] & 0xAAAAAAAA) >> 1) | (A[ 74] & 0xAAAAAAAA);
    B[ 75] = ((A[ 73] & 0xAAAAAAAA) >> 1) | (A[ 75] & 0xAAAAAAAA);
    B[ 76] = (A[ 76] & 0x55555555) | ((A[ 78] & 0x55555555) << 1);
    B[ 77] = (A[ 77] & 0x55555555) | ((A[ 79] & 0x55555555) << 1);
    B[ 78] = ((A[ 76] & 0xAAAAAAAA) >> 1) | (A[ 78] & 0xAAAAAAAA);
    B[ 79] = ((A[ 77] & 0xAAAAAAAA) >> 1) | (A[ 79] & 0xAAAAAAAA);
    B[ 80] = (A[ 80] & 0x55555555) | ((A[ 82] & 0x55555555) << 1);
    B[ 81] = (A[ 81] & 0x55555555) | ((A[ 83] & 0x55555555) << 1);
    B[ 82] = ((A[ 80] & 0xAAAAAAAA) >> 1) | (A[ 82] & 0xAAAAAAAA);
    B[ 83] = ((A[ 81] & 0xAAAAAAAA) >> 1) | (A[ 83] & 0xAAAAAAAA);
    B[ 84] = (A[ 84] & 0x55555555) | ((A[ 86] & 0x55555555) << 1);
    B[ 85] = (A[ 85] & 0x55555555) | ((A[ 87] & 0x55555555) << 1);
    B[ 86] = ((A[ 84] & 0xAAAAAAAA) >> 1) | (A[ 86] & 0xAAAAAAAA);
    B[ 87] = ((A[ 85] & 0xAAAAAAAA) >> 1) | (A[ 87] & 0xAAAAAAAA);
    B[ 88] = (A[ 88] & 0x55555555) | ((A[ 90] & 0x55555555) << 1);
    B[ 89] = (A[ 89] & 0x55555555) | ((A[ 91] & 0x55555555) << 1);
    B[ 90] = ((A[ 88] & 0xAAAAAAAA) >> 1) | (A[ 90] & 0xAAAAAAAA);
    B[ 91] = ((A[ 89] & 0xAAAAAAAA) >> 1) | (A[ 91] & 0xAAAAAAAA);
    B[ 92] = (A[ 92] & 0x55555555) | ((A[ 94] & 0x55555555) << 1);
    B[ 93] = (A[ 93] & 0x55555555) | ((A[ 95] & 0x55555555) << 1);
    B[ 94] = ((A[ 92] & 0xAAAAAAAA) >> 1) | (A[ 94] & 0xAAAAAAAA);
    B[ 95] = ((A[ 93] & 0xAAAAAAAA) >> 1) | (A[ 95] & 0xAAAAAAAA);
    B[ 96] = (A[ 96] & 0x55555555) | ((A[ 98] & 0x55555555) << 1);
    B[ 97] = (A[ 97] & 0x55555555) | ((A[ 99] & 0x55555555) << 1);
    B[ 98] = ((A[ 96] & 0xAAAAAAAA) >> 1) | (A[ 98] & 0xAAAAAAAA);
    B[ 99] = ((A[ 97] & 0xAAAAAAAA) >> 1) | (A[ 99] & 0xAAAAAAAA);
    B[100] = (A[100] & 0x55555555) | ((A[102] & 0x55555555) << 1);
    B[101] = (A[101] & 0x55555555) | ((A[103] & 0x55555555) << 1);
    B[102] = ((A[100] & 0xAAAAAAAA) >> 1) | (A[102] & 0xAAAAAAAA);
    B[103] = ((A[101] & 0xAAAAAAAA) >> 1) | (A[103] & 0xAAAAAAAA);
    B[104] = (A[104] & 0x55555555) | ((A[106] & 0x55555555) << 1);
    B[105] = (A[105] & 0x55555555) | ((A[107] & 0x55555555) << 1);
    B[106] = ((A[104] & 0xAAAAAAAA) >> 1) | (A[106] & 0xAAAAAAAA);
    B[107] = ((A[105] & 0xAAAAAAAA) >> 1) | (A[107] & 0xAAAAAAAA);
    B[108] = (A[108] & 0x55555555) | ((A[110] & 0x55555555) << 1);
    B[109] = (A[109] & 0x55555555) | ((A[111] & 0x55555555) << 1);
    B[110] = ((A[108] & 0xAAAAAAAA) >> 1) | (A[110] & 0xAAAAAAAA);
    B[111] = ((A[109] & 0xAAAAAAAA) >> 1) | (A[111] & 0xAAAAAAAA);
    B[112] = (A[112] & 0x55555555) | ((A[114] & 0x55555555) << 1);
    B[113] = (A[113] & 0x55555555) | ((A[115] & 0x55555555) << 1);
    B[114] = ((A[112] & 0xAAAAAAAA) >> 1) | (A[114] & 0xAAAAAAAA);
    B[115] = ((A[113] & 0xAAAAAAAA) >> 1) | (A[115] & 0xAAAAAAAA);
    B[116] = (A[116] & 0x55555555) | ((A[118] & 0x55555555) << 1);
    B[117] = (A[117] & 0x55555555) | ((A[119] & 0x55555555) << 1);
    B[118] = ((A[116] & 0xAAAAAAAA) >> 1) | (A[118] & 0xAAAAAAAA);
    B[119] = ((A[117] & 0xAAAAAAAA) >> 1) | (A[119] & 0xAAAAAAAA);
    B[120] = (A[120] & 0x55555555) | ((A[122] & 0x55555555) << 1);
    B[121] = (A[121] & 0x55555555) | ((A[123] & 0x55555555) << 1);
    B[122] = ((A[120] & 0xAAAAAAAA) >> 1) | (A[122] & 0xAAAAAAAA);
    B[123] = ((A[121] & 0xAAAAAAAA) >> 1) | (A[123] & 0xAAAAAAAA);
    B[124] = (A[124] & 0x55555555) | ((A[126] & 0x55555555) << 1);
    B[125] = (A[125] & 0x55555555) | ((A[127] & 0x55555555) << 1);
    B[126] = ((A[124] & 0xAAAAAAAA) >> 1) | (A[126] & 0xAAAAAAAA);
    B[127] = ((A[125] & 0xAAAAAAAA) >> 1) | (A[127] & 0xAAAAAAAA);
    C[  0] = (B[  0] & 0x33333333) | ((B[  4] & 0x33333333) << 2);
    C[  1] = (B[  1] & 0x33333333) | ((B[  5] & 0x33333333) << 2);
    C[  4] = ((B[  0] & 0xCCCCCCCC) >> 2) | (B[  4] & 0xCCCCCCCC);
    C[  5] = ((B[  1] & 0xCCCCCCCC) >> 2) | (B[  5] & 0xCCCCCCCC);
    C[  2] = (B[  2] & 0x33333333) | ((B[  6] & 0x33333333) << 2);
    C[  3] = (B[  3] & 0x33333333) | ((B[  7] & 0x33333333) << 2);
    C[  6] = ((B[  2] & 0xCCCCCCCC) >> 2) | (B[  6] & 0xCCCCCCCC);
    C[  7] = ((B[  3] & 0xCCCCCCCC) >> 2) | (B[  7] & 0xCCCCCCCC);
    C[  8] = (B[  8] & 0x33333333) | ((B[ 12] & 0x33333333) << 2);
    C[  9] = (B[  9] & 0x33333333) | ((B[ 13] & 0x33333333) << 2);
    C[ 12] = ((B[  8] & 0xCCCCCCCC) >> 2) | (B[ 12] & 0xCCCCCCCC);
    C[ 13] = ((B[  9] & 0xCCCCCCCC) >> 2) | (B[ 13] & 0xCCCCCCCC);
    C[ 10] = (B[ 10] & 0x33333333) | ((B[ 14] & 0x33333333) << 2);
    C[ 11] = (B[ 11] & 0x33333333) | ((B[ 15] & 0x33333333) << 2);
    C[ 14] = ((B[ 10] & 0xCCCCCCCC) >> 2) | (B[ 14] & 0xCCCCCCCC);
    C[ 15] = ((B[ 11] & 0xCCCCCCCC) >> 2) | (B[ 15] & 0xCCCCCCCC);
    C[ 16] = (B[ 16] & 0x33333333) | ((B[ 20] & 0x33333333) << 2);
    C[ 17] = (B[ 17] & 0x33333333) | ((B[ 21] & 0x33333333) << 2);
    C[ 20] = ((B[ 16] & 0xCCCCCCCC) >> 2) | (B[ 20] & 0xCCCCCCCC);
    C[ 21] = ((B[ 17] & 0xCCCCCCCC) >> 2) | (B[ 21] & 0xCCCCCCCC);
    C[ 18] = (B[ 18] & 0x33333333) | ((B[ 22] & 0x33333333) << 2);
    C[ 19] = (B[ 19] & 0x33333333) | ((B[ 23] & 0x33333333) << 2);
    C[ 22] = ((B[ 18] & 0xCCCCCCCC) >> 2) | (B[ 22] & 0xCCCCCCCC);
    C[ 23] = ((B[ 19] & 0xCCCCCCCC) >> 2) | (B[ 23] & 0xCCCCCCCC);
    C[ 24] = (B[ 24] & 0x33333333) | ((B[ 28] & 0x33333333) << 2);
    C[ 25] = (B[ 25] & 0x33333333) | ((B[ 29] & 0x33333333) << 2);
    C[ 28] = ((B[ 24] & 0xCCCCCCCC) >> 2) | (B[ 28] & 0xCCCCCCCC);
    C[ 29] = ((B[ 25] & 0xCCCCCCCC) >> 2) | (B[ 29] & 0xCCCCCCCC);
    C[ 26] = (B[ 26] & 0x33333333) | ((B[ 30] & 0x33333333) << 2);
    C[ 27] = (B[ 27] & 0x33333333) | ((B[ 31] & 0x33333333) << 2);
    C[ 30] = ((B[ 26] & 0xCCCCCCCC) >> 2) | (B[ 30] & 0xCCCCCCCC);
    C[ 31] = ((B[ 27] & 0xCCCCCCCC) >> 2) | (B[ 31] & 0xCCCCCCCC);
    C[ 32] = (B[ 32] & 0x33333333) | ((B[ 36] & 0x33333333) << 2);
    C[ 33] = (B[ 33] & 0x33333333) | ((B[ 37] & 0x33333333) << 2);
    C[ 36] = ((B[ 32] & 0xCCCCCCCC) >> 2) | (B[ 36] & 0xCCCCCCCC);
    C[ 37] = ((B[ 33] & 0xCCCCCCCC) >> 2) | (B[ 37] & 0xCCCCCCCC);
    C[ 34] = (B[ 34] & 0x33333333) | ((B[ 38] & 0x33333333) << 2);
    C[ 35] = (B[ 35] & 0x33333333) | ((B[ 39] & 0x33333333) << 2);
    C[ 38] = ((B[ 34] & 0xCCCCCCCC) >> 2) | (B[ 38] & 0xCCCCCCCC);
    C[ 39] = ((B[ 35] & 0xCCCCCCCC) >> 2) | (B[ 39] & 0xCCCCCCCC);
    C[ 40] = (B[ 40] & 0x33333333) | ((B[ 44] & 0x33333333) << 2);
    C[ 41] = (B[ 41] & 0x33333333) | ((B[ 45] & 0x33333333) << 2);
    C[ 44] = ((B[ 40] & 0xCCCCCCCC) >> 2) | (B[ 44] & 0xCCCCCCCC);
    C[ 45] = ((B[ 41] & 0xCCCCCCCC) >> 2) | (B[ 45] & 0xCCCCCCCC);
    C[ 42] = (B[ 42] & 0x33333333) | ((B[ 46] & 0x33333333) << 2);
    C[ 43] = (B[ 43] & 0x33333333) | ((B[ 47] & 0x33333333) << 2);
    C[ 46] = ((B[ 42] & 0xCCCCCCCC) >> 2) | (B[ 46] & 0xCCCCCCCC);
    C[ 47] = ((B[ 43] & 0xCCCCCCCC) >> 2) | (B[ 47] & 0xCCCCCCCC);
    C[ 48] = (B[ 48] & 0x33333333) | ((B[ 52] & 0x33333333) << 2);
    C[ 49] = (B[ 49] & 0x33333333) | ((B[ 53] & 0x33333333) << 2);
    C[ 52] = ((B[ 48] & 0xCCCCCCCC) >> 2) | (B[ 52] & 0xCCCCCCCC);
    C[ 53] = ((B[ 49] & 0xCCCCCCCC) >> 2) | (B[ 53] & 0xCCCCCCCC);
    C[ 50] = (B[ 50] & 0x33333333) | ((B[ 54] & 0x33333333) << 2);
    C[ 51] = (B[ 51] & 0x33333333) | ((B[ 55] & 0x33333333) << 2);
    C[ 54] = ((B[ 50] & 0xCCCCCCCC) >> 2) | (B[ 54] & 0xCCCCCCCC);
    C[ 55] = ((B[ 51] & 0xCCCCCCCC) >> 2) | (B[ 55] & 0xCCCCCCCC);
    C[ 56] = (B[ 56] & 0x33333333) | ((B[ 60] & 0x33333333) << 2);
    C[ 57] = (B[ 57] & 0x33333333) | ((B[ 61] & 0x33333333) << 2);
    C[ 60] = ((B[ 56] & 0xCCCCCCCC) >> 2) | (B[ 60] & 0xCCCCCCCC);
    C[ 61] = ((B[ 57] & 0xCCCCCCCC) >> 2) | (B[ 61] & 0xCCCCCCCC);
    C[ 58] = (B[ 58] & 0x33333333) | ((B[ 62] & 0x33333333) << 2);
    C[ 59] = (B[ 59] & 0x33333333) | ((B[ 63] & 0x33333333) << 2);
    C[ 62] = ((B[ 58] & 0xCCCCCCCC) >> 2) | (B[ 62] & 0xCCCCCCCC);
    C[ 63] = ((B[ 59] & 0xCCCCCCCC) >> 2) | (B[ 63] & 0xCCCCCCCC);
    C[ 64] = (B[ 64] & 0x33333333) | ((B[ 68] & 0x33333333) << 2);
    C[ 65] = (B[ 65] & 0x33333333) | ((B[ 69] & 0x33333333) << 2);
    C[ 68] = ((B[ 64] & 0xCCCCCCCC) >> 2) | (B[ 68] & 0xCCCCCCCC);
    C[ 69] = ((B[ 65] & 0xCCCCCCCC) >> 2) | (B[ 69] & 0xCCCCCCCC);
    C[ 66] = (B[ 66] & 0x33333333) | ((B[ 70] & 0x33333333) << 2);
    C[ 67] = (B[ 67] & 0x33333333) | ((B[ 71] & 0x33333333) << 2);
    C[ 70] = ((B[ 66] & 0xCCCCCCCC) >> 2) | (B[ 70] & 0xCCCCCCCC);
    C[ 71] = ((B[ 67] & 0xCCCCCCCC) >> 2) | (B[ 71] & 0xCCCCCCCC);
    C[ 72] = (B[ 72] & 0x33333333) | ((B[ 76] & 0x33333333) << 2);
    C[ 73] = (B[ 73] & 0x33333333) | ((B[ 77] & 0x33333333) << 2);
    C[ 76] = ((B[ 72] & 0xCCCCCCCC) >> 2) | (B[ 76] & 0xCCCCCCCC);
    C[ 77] = ((B[ 73] & 0xCCCCCCCC) >> 2) | (B[ 77] & 0xCCCCCCCC);
    C[ 74] = (B[ 74] & 0x33333333) | ((B[ 78] & 0x33333333) << 2);
    C[ 75] = (B[ 75] & 0x33333333) | ((B[ 79] & 0x33333333) << 2);
    C[ 78] = ((B[ 74] & 0xCCCCCCCC) >> 2) | (B[ 78] & 0xCCCCCCCC);
    C[ 79] = ((B[ 75] & 0xCCCCCCCC) >> 2) | (B[ 79] & 0xCCCCCCCC);
    C[ 80] = (B[ 80] & 0x33333333) | ((B[ 84] & 0x33333333) << 2);
    C[ 81] = (B[ 81] & 0x33333333) | ((B[ 85] & 0x33333333) << 2);
    C[ 84] = ((B[ 80] & 0xCCCCCCCC) >> 2) | (B[ 84] & 0xCCCCCCCC);
    C[ 85] = ((B[ 81] & 0xCCCCCCCC) >> 2) | (B[ 85] & 0xCCCCCCCC);
    C[ 82] = (B[ 82] & 0x33333333) | ((B[ 86] & 0x33333333) << 2);
    C[ 83] = (B[ 83] & 0x33333333) | ((B[ 87] & 0x33333333) << 2);
    C[ 86] = ((B[ 82] & 0xCCCCCCCC) >> 2) | (B[ 86] & 0xCCCCCCCC);
    C[ 87] = ((B[ 83] & 0xCCCCCCCC) >> 2) | (B[ 87] & 0xCCCCCCCC);
    C[ 88] = (B[ 88] & 0x33333333) | ((B[ 92] & 0x33333333) << 2);
    C[ 89] = (B[ 89] & 0x33333333) | ((B[ 93] & 0x33333333) << 2);
    C[ 92] = ((B[ 88] & 0xCCCCCCCC) >> 2) | (B[ 92] & 0xCCCCCCCC);
    C[ 93] = ((B[ 89] & 0xCCCCCCCC) >> 2) | (B[ 93] & 0xCCCCCCCC);
    C[ 90] = (B[ 90] & 0x33333333) | ((B[ 94] & 0x33333333) << 2);
    C[ 91] = (B[ 91] & 0x33333333) | ((B[ 95] & 0x33333333) << 2);
    C[ 94] = ((B[ 90] & 0xCCCCCCCC) >> 2) | (B[ 94] & 0xCCCCCCCC);
    C[ 95] = ((B[ 91] & 0xCCCCCCCC) >> 2) | (B[ 95] & 0xCCCCCCCC);
    C[ 96] = (B[ 96] & 0x33333333) | ((B[100] & 0x33333333) << 2);
    C[ 97] = (B[ 97] & 0x33333333) | ((B[101] & 0x33333333) << 2);
    C[100] = ((B[ 96] & 0xCCCCCCCC) >> 2) | (B[100] & 0xCCCCCCCC);
    C[101] = ((B[ 97] & 0xCCCCCCCC) >> 2) | (B[101] & 0xCCCCCCCC);
    C[ 98] = (B[ 98] & 0x33333333) | ((B[102] & 0x33333333) << 2);
    C[ 99] = (B[ 99] & 0x33333333) | ((B[103] & 0x33333333) << 2);
    C[102] = ((B[ 98] & 0xCCCCCCCC) >> 2) | (B[102] & 0xCCCCCCCC);
    C[103] = ((B[ 99] & 0xCCCCCCCC) >> 2) | (B[103] & 0xCCCCCCCC);
    C[104] = (B[104] & 0x33333333) | ((B[108] & 0x33333333) << 2);
    C[105] = (B[105] & 0x33333333) | ((B[109] & 0x33333333) << 2);
    C[108] = ((B[104] & 0xCCCCCCCC) >> 2) | (B[108] & 0xCCCCCCCC);
    C[109] = ((B[105] & 0xCCCCCCCC) >> 2) | (B[109] & 0xCCCCCCCC);
    C[106] = (B[106] & 0x33333333) | ((B[110] & 0x33333333) << 2);
    C[107] = (B[107] & 0x33333333) | ((B[111] & 0x33333333) << 2);
    C[110] = ((B[106] & 0xCCCCCCCC) >> 2) | (B[110] & 0xCCCCCCCC);
    C[111] = ((B[107] & 0xCCCCCCCC) >> 2) | (B[111] & 0xCCCCCCCC);
    C[112] = (B[112] & 0x33333333) | ((B[116] & 0x33333333) << 2);
    C[113] = (B[113] & 0x33333333) | ((B[117] & 0x33333333) << 2);
    C[116] = ((B[112] & 0xCCCCCCCC) >> 2) | (B[116] & 0xCCCCCCCC);
    C[117] = ((B[113] & 0xCCCCCCCC) >> 2) | (B[117] & 0xCCCCCCCC);
    C[114] = (B[114] & 0x33333333) | ((B[118] & 0x33333333) << 2);
    C[115] = (B[115] & 0x33333333) | ((B[119] & 0x33333333) << 2);
    C[118] = ((B[114] & 0xCCCCCCCC) >> 2) | (B[118] & 0xCCCCCCCC);
    C[119] = ((B[115] & 0xCCCCCCCC) >> 2) | (B[119] & 0xCCCCCCCC);
    C[120] = (B[120] & 0x33333333) | ((B[124] & 0x33333333) << 2);
    C[121] = (B[121] & 0x33333333) | ((B[125] & 0x33333333) << 2);
    C[124] = ((B[120] & 0xCCCCCCCC) >> 2) | (B[124] & 0xCCCCCCCC);
    C[125] = ((B[121] & 0xCCCCCCCC) >> 2) | (B[125] & 0xCCCCCCCC);
    C[122] = (B[122] & 0x33333333) | ((B[126] & 0x33333333) << 2);
    C[123] = (B[123] & 0x33333333) | ((B[127] & 0x33333333) << 2);
    C[126] = ((B[122] & 0xCCCCCCCC) >> 2) | (B[126] & 0xCCCCCCCC);
    C[127] = ((B[123] & 0xCCCCCCCC) >> 2) | (B[127] & 0xCCCCCCCC);
    B[  0] = (C[  0] & 0x0F0F0F0F) | ((C[  8] & 0x0F0F0F0F) << 4);
    B[  1] = (C[  1] & 0x0F0F0F0F) | ((C[  9] & 0x0F0F0F0F) << 4);
    B[  8] = ((C[  0] & 0xF0F0F0F0) >> 4) | (C[  8] & 0xF0F0F0F0);
    B[  9] = ((C[  1] & 0xF0F0F0F0) >> 4) | (C[  9] & 0xF0F0F0F0);
    B[  2] = (C[  2] & 0x0F0F0F0F) | ((C[ 10] & 0x0F0F0F0F) << 4);
    B[  3] = (C[  3] & 0x0F0F0F0F) | ((C[ 11] & 0x0F0F0F0F) << 4);
    B[ 10] = ((C[  2] & 0xF0F0F0F0) >> 4) | (C[ 10] & 0xF0F0F0F0);
    B[ 11] = ((C[  3] & 0xF0F0F0F0) >> 4) | (C[ 11] & 0xF0F0F0F0);
    B[  4] = (C[  4] & 0x0F0F0F0F) | ((C[ 12] & 0x0F0F0F0F) << 4);
    B[  5] = (C[  5] & 0x0F0F0F0F) | ((C[ 13] & 0x0F0F0F0F) << 4);
    B[ 12] = ((C[  4] & 0xF0F0F0F0) >> 4) | (C[ 12] & 0xF0F0F0F0);
    B[ 13] = ((C[  5] & 0xF0F0F0F0) >> 4) | (C[ 13] & 0xF0F0F0F0);
    B[  6] = (C[  6] & 0x0F0F0F0F) | ((C[ 14] & 0x0F0F0F0F) << 4);
    B[  7] = (C[  7] & 0x0F0F0F0F) | ((C[ 15] & 0x0F0F0F0F) << 4);
    B[ 14] = ((C[  6] & 0xF0F0F0F0) >> 4) | (C[ 14] & 0xF0F0F0F0);
    B[ 15] = ((C[  7] & 0xF0F0F0F0) >> 4) | (C[ 15] & 0xF0F0F0F0);
    B[ 16] = (C[ 16] & 0x0F0F0F0F) | ((C[ 24] & 0x0F0F0F0F) << 4);
    B[ 17] = (C[ 17] & 0x0F0F0F0F) | ((C[ 25] & 0x0F0F0F0F) << 4);
    B[ 24] = ((C[ 16] & 0xF0F0F0F0) >> 4) | (C[ 24] & 0xF0F0F0F0);
    B[ 25] = ((C[ 17] & 0xF0F0F0F0) >> 4) | (C[ 25] & 0xF0F0F0F0);
    B[ 18] = (C[ 18] & 0x0F0F0F0F) | ((C[ 26] & 0x0F0F0F0F) << 4);
    B[ 19] = (C[ 19] & 0x0F0F0F0F) | ((C[ 27] & 0x0F0F0F0F) << 4);
    B[ 26] = ((C[ 18] & 0xF0F0F0F0) >> 4) | (C[ 26] & 0xF0F0F0F0);
    B[ 27] = ((C[ 19] & 0xF0F0F0F0) >> 4) | (C[ 27] & 0xF0F0F0F0);
    B[ 20] = (C[ 20] & 0x0F0F0F0F) | ((C[ 28] & 0x0F0F0F0F) << 4);
    B[ 21] = (C[ 21] & 0x0F0F0F0F) | ((C[ 29] & 0x0F0F0F0F) << 4);
    B[ 28] = ((C[ 20] & 0xF0F0F0F0) >> 4) | (C[ 28] & 0xF0F0F0F0);
    B[ 29] = ((C[ 21] & 0xF0F0F0F0) >> 4) | (C[ 29] & 0xF0F0F0F0);
    B[ 22] = (C[ 22] & 0x0F0F0F0F) | ((C[ 30] & 0x0F0F0F0F) << 4);
    B[ 23] = (C[ 23] & 0x0F0F0F0F) | ((C[ 31] & 0x0F0F0F0F) << 4);
    B[ 30] = ((C[ 22] & 0xF0F0F0F0) >> 4) | (C[ 30] & 0xF0F0F0F0);
    B[ 31] = ((C[ 23] & 0xF0F0F0F0) >> 4) | (C[ 31] & 0xF0F0F0F0);
    B[ 32] = (C[ 32] & 0x0F0F0F0F) | ((C[ 40] & 0x0F0F0F0F) << 4);
    B[ 33] = (C[ 33] & 0x0F0F0F0F) | ((C[ 41] & 0x0F0F0F0F) << 4);
    B[ 40] = ((C[ 32] & 0xF0F0F0F0) >> 4) | (C[ 40] & 0xF0F0F0F0);
    B[ 41] = ((C[ 33] & 0xF0F0F0F0) >> 4) | (C[ 41] & 0xF0F0F0F0);
    B[ 34] = (C[ 34] & 0x0F0F0F0F) | ((C[ 42] & 0x0F0F0F0F) << 4);
    B[ 35] = (C[ 35] & 0x0F0F0F0F) | ((C[ 43] & 0x0F0F0F0F) << 4);
    B[ 42] = ((C[ 34] & 0xF0F0F0F0) >> 4) | (C[ 42] & 0xF0F0F0F0);
    B[ 43] = ((C[ 35] & 0xF0F0F0F0) >> 4) | (C[ 43] & 0xF0F0F0F0);
    B[ 36] = (C[ 36] & 0x0F0F0F0F) | ((C[ 44] & 0x0F0F0F0F) << 4);
    B[ 37] = (C[ 37] & 0x0F0F0F0F) | ((C[ 45] & 0x0F0F0F0F) << 4);
    B[ 44] = ((C[ 36] & 0xF0F0F0F0) >> 4) | (C[ 44] & 0xF0F0F0F0);
    B[ 45] = ((C[ 37] & 0xF0F0F0F0) >> 4) | (C[ 45] & 0xF0F0F0F0);
    B[ 38] = (C[ 38] & 0x0F0F0F0F) | ((C[ 46] & 0x0F0F0F0F) << 4);
    B[ 39] = (C[ 39] & 0x0F0F0F0F) | ((C[ 47] & 0x0F0F0F0F) << 4);
    B[ 46] = ((C[ 38] & 0xF0F0F0F0) >> 4) | (C[ 46] & 0xF0F0F0F0);
    B[ 47] = ((C[ 39] & 0xF0F0F0F0) >> 4) | (C[ 47] & 0xF0F0F0F0);
    B[ 48] = (C[ 48] & 0x0F0F0F0F) | ((C[ 56] & 0x0F0F0F0F) << 4);
    B[ 49] = (C[ 49] & 0x0F0F0F0F) | ((C[ 57] & 0x0F0F0F0F) << 4);
    B[ 56] = ((C[ 48] & 0xF0F0F0F0) >> 4) | (C[ 56] & 0xF0F0F0F0);
    B[ 57] = ((C[ 49] & 0xF0F0F0F0) >> 4) | (C[ 57] & 0xF0F0F0F0);
    B[ 50] = (C[ 50] & 0x0F0F0F0F) | ((C[ 58] & 0x0F0F0F0F) << 4);
    B[ 51] = (C[ 51] & 0x0F0F0F0F) | ((C[ 59] & 0x0F0F0F0F) << 4);
    B[ 58] = ((C[ 50] & 0xF0F0F0F0) >> 4) | (C[ 58] & 0xF0F0F0F0);
    B[ 59] = ((C[ 51] & 0xF0F0F0F0) >> 4) | (C[ 59] & 0xF0F0F0F0);
    B[ 52] = (C[ 52] & 0x0F0F0F0F) | ((C[ 60] & 0x0F0F0F0F) << 4);
    B[ 53] = (C[ 53] & 0x0F0F0F0F) | ((C[ 61] & 0x0F0F0F0F) << 4);
    B[ 60] = ((C[ 52] & 0xF0F0F0F0) >> 4) | (C[ 60] & 0xF0F0F0F0);
    B[ 61] = ((C[ 53] & 0xF0F0F0F0) >> 4) | (C[ 61] & 0xF0F0F0F0);
    B[ 54] = (C[ 54] & 0x0F0F0F0F) | ((C[ 62] & 0x0F0F0F0F) << 4);
    B[ 55] = (C[ 55] & 0x0F0F0F0F) | ((C[ 63] & 0x0F0F0F0F) << 4);
    B[ 62] = ((C[ 54] & 0xF0F0F0F0) >> 4) | (C[ 62] & 0xF0F0F0F0);
    B[ 63] = ((C[ 55] & 0xF0F0F0F0) >> 4) | (C[ 63] & 0xF0F0F0F0);
    B[ 64] = (C[ 64] & 0x0F0F0F0F) | ((C[ 72] & 0x0F0F0F0F) << 4);
    B[ 65] = (C[ 65] & 0x0F0F0F0F) | ((C[ 73] & 0x0F0F0F0F) << 4);
    B[ 72] = ((C[ 64] & 0xF0F0F0F0) >> 4) | (C[ 72] & 0xF0F0F0F0);
    B[ 73] = ((C[ 65] & 0xF0F0F0F0) >> 4) | (C[ 73] & 0xF0F0F0F0);
    B[ 66] = (C[ 66] & 0x0F0F0F0F) | ((C[ 74] & 0x0F0F0F0F) << 4);
    B[ 67] = (C[ 67] & 0x0F0F0F0F) | ((C[ 75] & 0x0F0F0F0F) << 4);
    B[ 74] = ((C[ 66] & 0xF0F0F0F0) >> 4) | (C[ 74] & 0xF0F0F0F0);
    B[ 75] = ((C[ 67] & 0xF0F0F0F0) >> 4) | (C[ 75] & 0xF0F0F0F0);
    B[ 68] = (C[ 68] & 0x0F0F0F0F) | ((C[ 76] & 0x0F0F0F0F) << 4);
    B[ 69] = (C[ 69] & 0x0F0F0F0F) | ((C[ 77] & 0x0F0F0F0F) << 4);
    B[ 76] = ((C[ 68] & 0xF0F0F0F0) >> 4) | (C[ 76] & 0xF0F0F0F0);
    B[ 77] = ((C[ 69] & 0xF0F0F0F0) >> 4) | (C[ 77] & 0xF0F0F0F0);
    B[ 70] = (C[ 70] & 0x0F0F0F0F) | ((C[ 78] & 0x0F0F0F0F) << 4);
    B[ 71] = (C[ 71] & 0x0F0F0F0F) | ((C[ 79] & 0x0F0F0F0F) << 4);
    B[ 78] = ((C[ 70] & 0xF0F0F0F0) >> 4) | (C[ 78] & 0xF0F0F0F0);
    B[ 79] = ((C[ 71] & 0xF0F0F0F0) >> 4) | (C[ 79] & 0xF0F0F0F0);
    B[ 80] = (C[ 80] & 0x0F0F0F0F) | ((C[ 88] & 0x0F0F0F0F) << 4);
    B[ 81] = (C[ 81] & 0x0F0F0F0F) | ((C[ 89] & 0x0F0F0F0F) << 4);
    B[ 88] = ((C[ 80] & 0xF0F0F0F0) >> 4) | (C[ 88] & 0xF0F0F0F0);
    B[ 89] = ((C[ 81] & 0xF0F0F0F0) >> 4) | (C[ 89] & 0xF0F0F0F0);
    B[ 82] = (C[ 82] & 0x0F0F0F0F) | ((C[ 90] & 0x0F0F0F0F) << 4);
    B[ 83] = (C[ 83] & 0x0F0F0F0F) | ((C[ 91] & 0x0F0F0F0F) << 4);
    B[ 90] = ((C[ 82] & 0xF0F0F0F0) >> 4) | (C[ 90] & 0xF0F0F0F0);
    B[ 91] = ((C[ 83] & 0xF0F0F0F0) >> 4) | (C[ 91] & 0xF0F0F0F0);
    B[ 84] = (C[ 84] & 0x0F0F0F0F) | ((C[ 92] & 0x0F0F0F0F) << 4);
    B[ 85] = (C[ 85] & 0x0F0F0F0F) | ((C[ 93] & 0x0F0F0F0F) << 4);
    B[ 92] = ((C[ 84] & 0xF0F0F0F0) >> 4) | (C[ 92] & 0xF0F0F0F0);
    B[ 93] = ((C[ 85] & 0xF0F0F0F0) >> 4) | (C[ 93] & 0xF0F0F0F0);
    B[ 86] = (C[ 86] & 0x0F0F0F0F) | ((C[ 94] & 0x0F0F0F0F) << 4);
    B[ 87] = (C[ 87] & 0x0F0F0F0F) | ((C[ 95] & 0x0F0F0F0F) << 4);
    B[ 94] = ((C[ 86] & 0xF0F0F0F0) >> 4) | (C[ 94] & 0xF0F0F0F0);
    B[ 95] = ((C[ 87] & 0xF0F0F0F0) >> 4) | (C[ 95] & 0xF0F0F0F0);
    B[ 96] = (C[ 96] & 0x0F0F0F0F) | ((C[104] & 0x0F0F0F0F) << 4);
    B[ 97] = (C[ 97] & 0x0F0F0F0F) | ((C[105] & 0x0F0F0F0F) << 4);
    B[104] = ((C[ 96] & 0xF0F0F0F0) >> 4) | (C[104] & 0xF0F0F0F0);
    B[105] = ((C[ 97] & 0xF0F0F0F0) >> 4) | (C[105] & 0xF0F0F0F0);
    B[ 98] = (C[ 98] & 0x0F0F0F0F) | ((C[106] & 0x0F0F0F0F) << 4);
    B[ 99] = (C[ 99] & 0x0F0F0F0F) | ((C[107] & 0x0F0F0F0F) << 4);
    B[106] = ((C[ 98] & 0xF0F0F0F0) >> 4) | (C[106] & 0xF0F0F0F0);
    B[107] = ((C[ 99] & 0xF0F0F0F0) >> 4) | (C[107] & 0xF0F0F0F0);
    B[100] = (C[100] & 0x0F0F0F0F) | ((C[108] & 0x0F0F0F0F) << 4);
    B[101] = (C[101] & 0x0F0F0F0F) | ((C[109] & 0x0F0F0F0F) << 4);
    B[108] = ((C[100] & 0xF0F0F0F0) >> 4) | (C[108] & 0xF0F0F0F0);
    B[109] = ((C[101] & 0xF0F0F0F0) >> 4) | (C[109] & 0xF0F0F0F0);
    B[102] = (C[102] & 0x0F0F0F0F) | ((C[110] & 0x0F0F0F0F) << 4);
    B[103] = (C[103] & 0x0F0F0F0F) | ((C[111] & 0x0F0F0F0F) << 4);
    B[110] = ((C[102] & 0xF0F0F0F0) >> 4) | (C[110] & 0xF0F0F0F0);
    B[111] = ((C[103] & 0xF0F0F0F0) >> 4) | (C[111] & 0xF0F0F0F0);
    B[112] = (C[112] & 0x0F0F0F0F) | ((C[120] & 0x0F0F0F0F) << 4);
    B[113] = (C[113] & 0x0F0F0F0F) | ((C[121] & 0x0F0F0F0F) << 4);
    B[120] = ((C[112] & 0xF0F0F0F0) >> 4) | (C[120] & 0xF0F0F0F0);
    B[121] = ((C[113] & 0xF0F0F0F0) >> 4) | (C[121] & 0xF0F0F0F0);
    B[114] = (C[114] & 0x0F0F0F0F) | ((C[122] & 0x0F0F0F0F) << 4);
    B[115] = (C[115] & 0x0F0F0F0F) | ((C[123] & 0x0F0F0F0F) << 4);
    B[122] = ((C[114] & 0xF0F0F0F0) >> 4) | (C[122] & 0xF0F0F0F0);
    B[123] = ((C[115] & 0xF0F0F0F0) >> 4) | (C[123] & 0xF0F0F0F0);
    B[116] = (C[116] & 0x0F0F0F0F) | ((C[124] & 0x0F0F0F0F) << 4);
    B[117] = (C[117] & 0x0F0F0F0F) | ((C[125] & 0x0F0F0F0F) << 4);
    B[124] = ((C[116] & 0xF0F0F0F0) >> 4) | (C[124] & 0xF0F0F0F0);
    B[125] = ((C[117] & 0xF0F0F0F0) >> 4) | (C[125] & 0xF0F0F0F0);
    B[118] = (C[118] & 0x0F0F0F0F) | ((C[126] & 0x0F0F0F0F) << 4);
    B[119] = (C[119] & 0x0F0F0F0F) | ((C[127] & 0x0F0F0F0F) << 4);
    B[126] = ((C[118] & 0xF0F0F0F0) >> 4) | (C[126] & 0xF0F0F0F0);
    B[127] = ((C[119] & 0xF0F0F0F0) >> 4) | (C[127] & 0xF0F0F0F0);
    C[  0] = (B[  0] & 0x00FF00FF) | ((B[ 16] & 0x00FF00FF) << 8);
    C[  1] = (B[  1] & 0x00FF00FF) | ((B[ 17] & 0x00FF00FF) << 8);
    C[ 16] = ((B[  0] & 0xFF00FF00) >> 8) | (B[ 16] & 0xFF00FF00);
    C[ 17] = ((B[  1] & 0xFF00FF00) >> 8) | (B[ 17] & 0xFF00FF00);
    C[  2] = (B[  2] & 0x00FF00FF) | ((B[ 18] & 0x00FF00FF) << 8);
    C[  3] = (B[  3] & 0x00FF00FF) | ((B[ 19] & 0x00FF00FF) << 8);
    C[ 18] = ((B[  2] & 0xFF00FF00) >> 8) | (B[ 18] & 0xFF00FF00);
    C[ 19] = ((B[  3] & 0xFF00FF00) >> 8) | (B[ 19] & 0xFF00FF00);
    C[  4] = (B[  4] & 0x00FF00FF) | ((B[ 20] & 0x00FF00FF) << 8);
    C[  5] = (B[  5] & 0x00FF00FF) | ((B[ 21] & 0x00FF00FF) << 8);
    C[ 20] = ((B[  4] & 0xFF00FF00) >> 8) | (B[ 20] & 0xFF00FF00);
    C[ 21] = ((B[  5] & 0xFF00FF00) >> 8) | (B[ 21] & 0xFF00FF00);
    C[  6] = (B[  6] & 0x00FF00FF) | ((B[ 22] & 0x00FF00FF) << 8);
    C[  7] = (B[  7] & 0x00FF00FF) | ((B[ 23] & 0x00FF00FF) << 8);
    C[ 22] = ((B[  6] & 0xFF00FF00) >> 8) | (B[ 22] & 0xFF00FF00);
    C[ 23] = ((B[  7] & 0xFF00FF00) >> 8) | (B[ 23] & 0xFF00FF00);
    C[  8] = (B[  8] & 0x00FF00FF) | ((B[ 24] & 0x00FF00FF) << 8);
    C[  9] = (B[  9] & 0x00FF00FF) | ((B[ 25] & 0x00FF00FF) << 8);
    C[ 24] = ((B[  8] & 0xFF00FF00) >> 8) | (B[ 24] & 0xFF00FF00);
    C[ 25] = ((B[  9] & 0xFF00FF00) >> 8) | (B[ 25] & 0xFF00FF00);
    C[ 10] = (B[ 10] & 0x00FF00FF) | ((B[ 26] & 0x00FF00FF) << 8);
    C[ 11] = (B[ 11] & 0x00FF00FF) | ((B[ 27] & 0x00FF00FF) << 8);
    C[ 26] = ((B[ 10] & 0xFF00FF00) >> 8) | (B[ 26] & 0xFF00FF00);
    C[ 27] = ((B[ 11] & 0xFF00FF00) >> 8) | (B[ 27] & 0xFF00FF00);
    C[ 12] = (B[ 12] & 0x00FF00FF) | ((B[ 28] & 0x00FF00FF) << 8);
    C[ 13] = (B[ 13] & 0x00FF00FF) | ((B[ 29] & 0x00FF00FF) << 8);
    C[ 28] = ((B[ 12] & 0xFF00FF00) >> 8) | (B[ 28] & 0xFF00FF00);
    C[ 29] = ((B[ 13] & 0xFF00FF00) >> 8) | (B[ 29] & 0xFF00FF00);
    C[ 14] = (B[ 14] & 0x00FF00FF) | ((B[ 30] & 0x00FF00FF) << 8);
    C[ 15] = (B[ 15] & 0x00FF00FF) | ((B[ 31] & 0x00FF00FF) << 8);
    C[ 30] = ((B[ 14] & 0xFF00FF00) >> 8) | (B[ 30] & 0xFF00FF00);
    C[ 31] = ((B[ 15] & 0xFF00FF00) >> 8) | (B[ 31] & 0xFF00FF00);
    C[ 32] = (B[ 32] & 0x00FF00FF) | ((B[ 48] & 0x00FF00FF) << 8);
    C[ 33] = (B[ 33] & 0x00FF00FF) | ((B[ 49] & 0x00FF00FF) << 8);
    C[ 48] = ((B[ 32] & 0xFF00FF00) >> 8) | (B[ 48] & 0xFF00FF00);
    C[ 49] = ((B[ 33] & 0xFF00FF00) >> 8) | (B[ 49] & 0xFF00FF00);
    C[ 34] = (B[ 34] & 0x00FF00FF) | ((B[ 50] & 0x00FF00FF) << 8);
    C[ 35] = (B[ 35] & 0x00FF00FF) | ((B[ 51] & 0x00FF00FF) << 8);
    C[ 50] = ((B[ 34] & 0xFF00FF00) >> 8) | (B[ 50] & 0xFF00FF00);
    C[ 51] = ((B[ 35] & 0xFF00FF00) >> 8) | (B[ 51] & 0xFF00FF00);
    C[ 36] = (B[ 36] & 0x00FF00FF) | ((B[ 52] & 0x00FF00FF) << 8);
    C[ 37] = (B[ 37] & 0x00FF00FF) | ((B[ 53] & 0x00FF00FF) << 8);
    C[ 52] = ((B[ 36] & 0xFF00FF00) >> 8) | (B[ 52] & 0xFF00FF00);
    C[ 53] = ((B[ 37] & 0xFF00FF00) >> 8) | (B[ 53] & 0xFF00FF00);
    C[ 38] = (B[ 38] & 0x00FF00FF) | ((B[ 54] & 0x00FF00FF) << 8);
    C[ 39] = (B[ 39] & 0x00FF00FF) | ((B[ 55] & 0x00FF00FF) << 8);
    C[ 54] = ((B[ 38] & 0xFF00FF00) >> 8) | (B[ 54] & 0xFF00FF00);
    C[ 55] = ((B[ 39] & 0xFF00FF00) >> 8) | (B[ 55] & 0xFF00FF00);
    C[ 40] = (B[ 40] & 0x00FF00FF) | ((B[ 56] & 0x00FF00FF) << 8);
    C[ 41] = (B[ 41] & 0x00FF00FF) | ((B[ 57] & 0x00FF00FF) << 8);
    C[ 56] = ((B[ 40] & 0xFF00FF00) >> 8) | (B[ 56] & 0xFF00FF00);
    C[ 57] = ((B[ 41] & 0xFF00FF00) >> 8) | (B[ 57] & 0xFF00FF00);
    C[ 42] = (B[ 42] & 0x00FF00FF) | ((B[ 58] & 0x00FF00FF) << 8);
    C[ 43] = (B[ 43] & 0x00FF00FF) | ((B[ 59] & 0x00FF00FF) << 8);
    C[ 58] = ((B[ 42] & 0xFF00FF00) >> 8) | (B[ 58] & 0xFF00FF00);
    C[ 59] = ((B[ 43] & 0xFF00FF00) >> 8) | (B[ 59] & 0xFF00FF00);
    C[ 44] = (B[ 44] & 0x00FF00FF) | ((B[ 60] & 0x00FF00FF) << 8);
    C[ 45] = (B[ 45] & 0x00FF00FF) | ((B[ 61] & 0x00FF00FF) << 8);
    C[ 60] = ((B[ 44] & 0xFF00FF00) >> 8) | (B[ 60] & 0xFF00FF00);
    C[ 61] = ((B[ 45] & 0xFF00FF00) >> 8) | (B[ 61] & 0xFF00FF00);
    C[ 46] = (B[ 46] & 0x00FF00FF) | ((B[ 62] & 0x00FF00FF) << 8);
    C[ 47] = (B[ 47] & 0x00FF00FF) | ((B[ 63] & 0x00FF00FF) << 8);
    C[ 62] = ((B[ 46] & 0xFF00FF00) >> 8) | (B[ 62] & 0xFF00FF00);
    C[ 63] = ((B[ 47] & 0xFF00FF00) >> 8) | (B[ 63] & 0xFF00FF00);
    C[ 64] = (B[ 64] & 0x00FF00FF) | ((B[ 80] & 0x00FF00FF) << 8);
    C[ 65] = (B[ 65] & 0x00FF00FF) | ((B[ 81] & 0x00FF00FF) << 8);
    C[ 80] = ((B[ 64] & 0xFF00FF00) >> 8) | (B[ 80] & 0xFF00FF00);
    C[ 81] = ((B[ 65] & 0xFF00FF00) >> 8) | (B[ 81] & 0xFF00FF00);
    C[ 66] = (B[ 66] & 0x00FF00FF) | ((B[ 82] & 0x00FF00FF) << 8);
    C[ 67] = (B[ 67] & 0x00FF00FF) | ((B[ 83] & 0x00FF00FF) << 8);
    C[ 82] = ((B[ 66] & 0xFF00FF00) >> 8) | (B[ 82] & 0xFF00FF00);
    C[ 83] = ((B[ 67] & 0xFF00FF00) >> 8) | (B[ 83] & 0xFF00FF00);
    C[ 68] = (B[ 68] & 0x00FF00FF) | ((B[ 84] & 0x00FF00FF) << 8);
    C[ 69] = (B[ 69] & 0x00FF00FF) | ((B[ 85] & 0x00FF00FF) << 8);
    C[ 84] = ((B[ 68] & 0xFF00FF00) >> 8) | (B[ 84] & 0xFF00FF00);
    C[ 85] = ((B[ 69] & 0xFF00FF00) >> 8) | (B[ 85] & 0xFF00FF00);
    C[ 70] = (B[ 70] & 0x00FF00FF) | ((B[ 86] & 0x00FF00FF) << 8);
    C[ 71] = (B[ 71] & 0x00FF00FF) | ((B[ 87] & 0x00FF00FF) << 8);
    C[ 86] = ((B[ 70] & 0xFF00FF00) >> 8) | (B[ 86] & 0xFF00FF00);
    C[ 87] = ((B[ 71] & 0xFF00FF00) >> 8) | (B[ 87] & 0xFF00FF00);
    C[ 72] = (B[ 72] & 0x00FF00FF) | ((B[ 88] & 0x00FF00FF) << 8);
    C[ 73] = (B[ 73] & 0x00FF00FF) | ((B[ 89] & 0x00FF00FF) << 8);
    C[ 88] = ((B[ 72] & 0xFF00FF00) >> 8) | (B[ 88] & 0xFF00FF00);
    C[ 89] = ((B[ 73] & 0xFF00FF00) >> 8) | (B[ 89] & 0xFF00FF00);
    C[ 74] = (B[ 74] & 0x00FF00FF) | ((B[ 90] & 0x00FF00FF) << 8);
    C[ 75] = (B[ 75] & 0x00FF00FF) | ((B[ 91] & 0x00FF00FF) << 8);
    C[ 90] = ((B[ 74] & 0xFF00FF00) >> 8) | (B[ 90] & 0xFF00FF00);
    C[ 91] = ((B[ 75] & 0xFF00FF00) >> 8) | (B[ 91] & 0xFF00FF00);
    C[ 76] = (B[ 76] & 0x00FF00FF) | ((B[ 92] & 0x00FF00FF) << 8);
    C[ 77] = (B[ 77] & 0x00FF00FF) | ((B[ 93] & 0x00FF00FF) << 8);
    C[ 92] = ((B[ 76] & 0xFF00FF00) >> 8) | (B[ 92] & 0xFF00FF00);
    C[ 93] = ((B[ 77] & 0xFF00FF00) >> 8) | (B[ 93] & 0xFF00FF00);
    C[ 78] = (B[ 78] & 0x00FF00FF) | ((B[ 94] & 0x00FF00FF) << 8);
    C[ 79] = (B[ 79] & 0x00FF00FF) | ((B[ 95] & 0x00FF00FF) << 8);
    C[ 94] = ((B[ 78] & 0xFF00FF00) >> 8) | (B[ 94] & 0xFF00FF00);
    C[ 95] = ((B[ 79] & 0xFF00FF00) >> 8) | (B[ 95] & 0xFF00FF00);
    C[ 96] = (B[ 96] & 0x00FF00FF) | ((B[112] & 0x00FF00FF) << 8);
    C[ 97] = (B[ 97] & 0x00FF00FF) | ((B[113] & 0x00FF00FF) << 8);
    C[112] = ((B[ 96] & 0xFF00FF00) >> 8) | (B[112] & 0xFF00FF00);
    C[113] = ((B[ 97] & 0xFF00FF00) >> 8) | (B[113] & 0xFF00FF00);
    C[ 98] = (B[ 98] & 0x00FF00FF) | ((B[114] & 0x00FF00FF) << 8);
    C[ 99] = (B[ 99] & 0x00FF00FF) | ((B[115] & 0x00FF00FF) << 8);
    C[114] = ((B[ 98] & 0xFF00FF00) >> 8) | (B[114] & 0xFF00FF00);
    C[115] = ((B[ 99] & 0xFF00FF00) >> 8) | (B[115] & 0xFF00FF00);
    C[100] = (B[100] & 0x00FF00FF) | ((B[116] & 0x00FF00FF) << 8);
    C[101] = (B[101] & 0x00FF00FF) | ((B[117] & 0x00FF00FF) << 8);
    C[116] = ((B[100] & 0xFF00FF00) >> 8) | (B[116] & 0xFF00FF00);
    C[117] = ((B[101] & 0xFF00FF00) >> 8) | (B[117] & 0xFF00FF00);
    C[102] = (B[102] & 0x00FF00FF) | ((B[118] & 0x00FF00FF) << 8);
    C[103] = (B[103] & 0x00FF00FF) | ((B[119] & 0x00FF00FF) << 8);
    C[118] = ((B[102] & 0xFF00FF00) >> 8) | (B[118] & 0xFF00FF00);
    C[119] = ((B[103] & 0xFF00FF00) >> 8) | (B[119] & 0xFF00FF00);
    C[104] = (B[104] & 0x00FF00FF) | ((B[120] & 0x00FF00FF) << 8);
    C[105] = (B[105] & 0x00FF00FF) | ((B[121] & 0x00FF00FF) << 8);
    C[120] = ((B[104] & 0xFF00FF00) >> 8) | (B[120] & 0xFF00FF00);
    C[121] = ((B[105] & 0xFF00FF00) >> 8) | (B[121] & 0xFF00FF00);
    C[106] = (B[106] & 0x00FF00FF) | ((B[122] & 0x00FF00FF) << 8);
    C[107] = (B[107] & 0x00FF00FF) | ((B[123] & 0x00FF00FF) << 8);
    C[122] = ((B[106] & 0xFF00FF00) >> 8) | (B[122] & 0xFF00FF00);
    C[123] = ((B[107] & 0xFF00FF00) >> 8) | (B[123] & 0xFF00FF00);
    C[108] = (B[108] & 0x00FF00FF) | ((B[124] & 0x00FF00FF) << 8);
    C[109] = (B[109] & 0x00FF00FF) | ((B[125] & 0x00FF00FF) << 8);
    C[124] = ((B[108] & 0xFF00FF00) >> 8) | (B[124] & 0xFF00FF00);
    C[125] = ((B[109] & 0xFF00FF00) >> 8) | (B[125] & 0xFF00FF00);
    C[110] = (B[110] & 0x00FF00FF) | ((B[126] & 0x00FF00FF) << 8);
    C[111] = (B[111] & 0x00FF00FF) | ((B[127] & 0x00FF00FF) << 8);
    C[126] = ((B[110] & 0xFF00FF00) >> 8) | (B[126] & 0xFF00FF00);
    C[127] = ((B[111] & 0xFF00FF00) >> 8) | (B[127] & 0xFF00FF00);
    B[  0] = (C[  0] & 0x0000FFFF) | ((C[ 32] & 0x0000FFFF) << 16);
    B[  1] = (C[  1] & 0x0000FFFF) | ((C[ 33] & 0x0000FFFF) << 16);
    B[ 32] = ((C[  0] & 0xFFFF0000) >> 16) | (C[ 32] & 0xFFFF0000);
    B[ 33] = ((C[  1] & 0xFFFF0000) >> 16) | (C[ 33] & 0xFFFF0000);
    B[  2] = (C[  2] & 0x0000FFFF) | ((C[ 34] & 0x0000FFFF) << 16);
    B[  3] = (C[  3] & 0x0000FFFF) | ((C[ 35] & 0x0000FFFF) << 16);
    B[ 34] = ((C[  2] & 0xFFFF0000) >> 16) | (C[ 34] & 0xFFFF0000);
    B[ 35] = ((C[  3] & 0xFFFF0000) >> 16) | (C[ 35] & 0xFFFF0000);
    B[  4] = (C[  4] & 0x0000FFFF) | ((C[ 36] & 0x0000FFFF) << 16);
    B[  5] = (C[  5] & 0x0000FFFF) | ((C[ 37] & 0x0000FFFF) << 16);
    B[ 36] = ((C[  4] & 0xFFFF0000) >> 16) | (C[ 36] & 0xFFFF0000);
    B[ 37] = ((C[  5] & 0xFFFF0000) >> 16) | (C[ 37] & 0xFFFF0000);
    B[  6] = (C[  6] & 0x0000FFFF) | ((C[ 38] & 0x0000FFFF) << 16);
    B[  7] = (C[  7] & 0x0000FFFF) | ((C[ 39] & 0x0000FFFF) << 16);
    B[ 38] = ((C[  6] & 0xFFFF0000) >> 16) | (C[ 38] & 0xFFFF0000);
    B[ 39] = ((C[  7] & 0xFFFF0000) >> 16) | (C[ 39] & 0xFFFF0000);
    B[  8] = (C[  8] & 0x0000FFFF) | ((C[ 40] & 0x0000FFFF) << 16);
    B[  9] = (C[  9] & 0x0000FFFF) | ((C[ 41] & 0x0000FFFF) << 16);
    B[ 40] = ((C[  8] & 0xFFFF0000) >> 16) | (C[ 40] & 0xFFFF0000);
    B[ 41] = ((C[  9] & 0xFFFF0000) >> 16) | (C[ 41] & 0xFFFF0000);
    B[ 10] = (C[ 10] & 0x0000FFFF) | ((C[ 42] & 0x0000FFFF) << 16);
    B[ 11] = (C[ 11] & 0x0000FFFF) | ((C[ 43] & 0x0000FFFF) << 16);
    B[ 42] = ((C[ 10] & 0xFFFF0000) >> 16) | (C[ 42] & 0xFFFF0000);
    B[ 43] = ((C[ 11] & 0xFFFF0000) >> 16) | (C[ 43] & 0xFFFF0000);
    B[ 12] = (C[ 12] & 0x0000FFFF) | ((C[ 44] & 0x0000FFFF) << 16);
    B[ 13] = (C[ 13] & 0x0000FFFF) | ((C[ 45] & 0x0000FFFF) << 16);
    B[ 44] = ((C[ 12] & 0xFFFF0000) >> 16) | (C[ 44] & 0xFFFF0000);
    B[ 45] = ((C[ 13] & 0xFFFF0000) >> 16) | (C[ 45] & 0xFFFF0000);
    B[ 14] = (C[ 14] & 0x0000FFFF) | ((C[ 46] & 0x0000FFFF) << 16);
    B[ 15] = (C[ 15] & 0x0000FFFF) | ((C[ 47] & 0x0000FFFF) << 16);
    B[ 46] = ((C[ 14] & 0xFFFF0000) >> 16) | (C[ 46] & 0xFFFF0000);
    B[ 47] = ((C[ 15] & 0xFFFF0000) >> 16) | (C[ 47] & 0xFFFF0000);
    B[ 16] = (C[ 16] & 0x0000FFFF) | ((C[ 48] & 0x0000FFFF) << 16);
    B[ 17] = (C[ 17] & 0x0000FFFF) | ((C[ 49] & 0x0000FFFF) << 16);
    B[ 48] = ((C[ 16] & 0xFFFF0000) >> 16) | (C[ 48] & 0xFFFF0000);
    B[ 49] = ((C[ 17] & 0xFFFF0000) >> 16) | (C[ 49] & 0xFFFF0000);
    B[ 18] = (C[ 18] & 0x0000FFFF) | ((C[ 50] & 0x0000FFFF) << 16);
    B[ 19] = (C[ 19] & 0x0000FFFF) | ((C[ 51] & 0x0000FFFF) << 16);
    B[ 50] = ((C[ 18] & 0xFFFF0000) >> 16) | (C[ 50] & 0xFFFF0000);
    B[ 51] = ((C[ 19] & 0xFFFF0000) >> 16) | (C[ 51] & 0xFFFF0000);
    B[ 20] = (C[ 20] & 0x0000FFFF) | ((C[ 52] & 0x0000FFFF) << 16);
    B[ 21] = (C[ 21] & 0x0000FFFF) | ((C[ 53] & 0x0000FFFF) << 16);
    B[ 52] = ((C[ 20] & 0xFFFF0000) >> 16) | (C[ 52] & 0xFFFF0000);
    B[ 53] = ((C[ 21] & 0xFFFF0000) >> 16) | (C[ 53] & 0xFFFF0000);
    B[ 22] = (C[ 22] & 0x0000FFFF) | ((C[ 54] & 0x0000FFFF) << 16);
    B[ 23] = (C[ 23] & 0x0000FFFF) | ((C[ 55] & 0x0000FFFF) << 16);
    B[ 54] = ((C[ 22] & 0xFFFF0000) >> 16) | (C[ 54] & 0xFFFF0000);
    B[ 55] = ((C[ 23] & 0xFFFF0000) >> 16) | (C[ 55] & 0xFFFF0000);
    B[ 24] = (C[ 24] & 0x0000FFFF) | ((C[ 56] & 0x0000FFFF) << 16);
    B[ 25] = (C[ 25] & 0x0000FFFF) | ((C[ 57] & 0x0000FFFF) << 16);
    B[ 56] = ((C[ 24] & 0xFFFF0000) >> 16) | (C[ 56] & 0xFFFF0000);
    B[ 57] = ((C[ 25] & 0xFFFF0000) >> 16) | (C[ 57] & 0xFFFF0000);
    B[ 26] = (C[ 26] & 0x0000FFFF) | ((C[ 58] & 0x0000FFFF) << 16);
    B[ 27] = (C[ 27] & 0x0000FFFF) | ((C[ 59] & 0x0000FFFF) << 16);
    B[ 58] = ((C[ 26] & 0xFFFF0000) >> 16) | (C[ 58] & 0xFFFF0000);
    B[ 59] = ((C[ 27] & 0xFFFF0000) >> 16) | (C[ 59] & 0xFFFF0000);
    B[ 28] = (C[ 28] & 0x0000FFFF) | ((C[ 60] & 0x0000FFFF) << 16);
    B[ 29] = (C[ 29] & 0x0000FFFF) | ((C[ 61] & 0x0000FFFF) << 16);
    B[ 60] = ((C[ 28] & 0xFFFF0000) >> 16) | (C[ 60] & 0xFFFF0000);
    B[ 61] = ((C[ 29] & 0xFFFF0000) >> 16) | (C[ 61] & 0xFFFF0000);
    B[ 30] = (C[ 30] & 0x0000FFFF) | ((C[ 62] & 0x0000FFFF) << 16);
    B[ 31] = (C[ 31] & 0x0000FFFF) | ((C[ 63] & 0x0000FFFF) << 16);
    B[ 62] = ((C[ 30] & 0xFFFF0000) >> 16) | (C[ 62] & 0xFFFF0000);
    B[ 63] = ((C[ 31] & 0xFFFF0000) >> 16) | (C[ 63] & 0xFFFF0000);
    B[ 64] = (C[ 64] & 0x0000FFFF) | ((C[ 96] & 0x0000FFFF) << 16);
    B[ 65] = (C[ 65] & 0x0000FFFF) | ((C[ 97] & 0x0000FFFF) << 16);
    B[ 96] = ((C[ 64] & 0xFFFF0000) >> 16) | (C[ 96] & 0xFFFF0000);
    B[ 97] = ((C[ 65] & 0xFFFF0000) >> 16) | (C[ 97] & 0xFFFF0000);
    B[ 66] = (C[ 66] & 0x0000FFFF) | ((C[ 98] & 0x0000FFFF) << 16);
    B[ 67] = (C[ 67] & 0x0000FFFF) | ((C[ 99] & 0x0000FFFF) << 16);
    B[ 98] = ((C[ 66] & 0xFFFF0000) >> 16) | (C[ 98] & 0xFFFF0000);
    B[ 99] = ((C[ 67] & 0xFFFF0000) >> 16) | (C[ 99] & 0xFFFF0000);
    B[ 68] = (C[ 68] & 0x0000FFFF) | ((C[100] & 0x0000FFFF) << 16);
    B[ 69] = (C[ 69] & 0x0000FFFF) | ((C[101] & 0x0000FFFF) << 16);
    B[100] = ((C[ 68] & 0xFFFF0000) >> 16) | (C[100] & 0xFFFF0000);
    B[101] = ((C[ 69] & 0xFFFF0000) >> 16) | (C[101] & 0xFFFF0000);
    B[ 70] = (C[ 70] & 0x0000FFFF) | ((C[102] & 0x0000FFFF) << 16);
    B[ 71] = (C[ 71] & 0x0000FFFF) | ((C[103] & 0x0000FFFF) << 16);
    B[102] = ((C[ 70] & 0xFFFF0000) >> 16) | (C[102] & 0xFFFF0000);
    B[103] = ((C[ 71] & 0xFFFF0000) >> 16) | (C[103] & 0xFFFF0000);
    B[ 72] = (C[ 72] & 0x0000FFFF) | ((C[104] & 0x0000FFFF) << 16);
    B[ 73] = (C[ 73] & 0x0000FFFF) | ((C[105] & 0x0000FFFF) << 16);
    B[104] = ((C[ 72] & 0xFFFF0000) >> 16) | (C[104] & 0xFFFF0000);
    B[105] = ((C[ 73] & 0xFFFF0000) >> 16) | (C[105] & 0xFFFF0000);
    B[ 74] = (C[ 74] & 0x0000FFFF) | ((C[106] & 0x0000FFFF) << 16);
    B[ 75] = (C[ 75] & 0x0000FFFF) | ((C[107] & 0x0000FFFF) << 16);
    B[106] = ((C[ 74] & 0xFFFF0000) >> 16) | (C[106] & 0xFFFF0000);
    B[107] = ((C[ 75] & 0xFFFF0000) >> 16) | (C[107] & 0xFFFF0000);
    B[ 76] = (C[ 76] & 0x0000FFFF) | ((C[108] & 0x0000FFFF) << 16);
    B[ 77] = (C[ 77] & 0x0000FFFF) | ((C[109] & 0x0000FFFF) << 16);
    B[108] = ((C[ 76] & 0xFFFF0000) >> 16) | (C[108] & 0xFFFF0000);
    B[109] = ((C[ 77] & 0xFFFF0000) >> 16) | (C[109] & 0xFFFF0000);
    B[ 78] = (C[ 78] & 0x0000FFFF) | ((C[110] & 0x0000FFFF) << 16);
    B[ 79] = (C[ 79] & 0x0000FFFF) | ((C[111] & 0x0000FFFF) << 16);
    B[110] = ((C[ 78] & 0xFFFF0000) >> 16) | (C[110] & 0xFFFF0000);
    B[111] = ((C[ 79] & 0xFFFF0000) >> 16) | (C[111] & 0xFFFF0000);
    B[ 80] = (C[ 80] & 0x0000FFFF) | ((C[112] & 0x0000FFFF) << 16);
    B[ 81] = (C[ 81] & 0x0000FFFF) | ((C[113] & 0x0000FFFF) << 16);
    B[112] = ((C[ 80] & 0xFFFF0000) >> 16) | (C[112] & 0xFFFF0000);
    B[113] = ((C[ 81] & 0xFFFF0000) >> 16) | (C[113] & 0xFFFF0000);
    B[ 82] = (C[ 82] & 0x0000FFFF) | ((C[114] & 0x0000FFFF) << 16);
    B[ 83] = (C[ 83] & 0x0000FFFF) | ((C[115] & 0x0000FFFF) << 16);
    B[114] = ((C[ 82] & 0xFFFF0000) >> 16) | (C[114] & 0xFFFF0000);
    B[115] = ((C[ 83] & 0xFFFF0000) >> 16) | (C[115] & 0xFFFF0000);
    B[ 84] = (C[ 84] & 0x0000FFFF) | ((C[116] & 0x0000FFFF) << 16);
    B[ 85] = (C[ 85] & 0x0000FFFF) | ((C[117] & 0x0000FFFF) << 16);
    B[116] = ((C[ 84] & 0xFFFF0000) >> 16) | (C[116] & 0xFFFF0000);
    B[117] = ((C[ 85] & 0xFFFF0000) >> 16) | (C[117] & 0xFFFF0000);
    B[ 86] = (C[ 86] & 0x0000FFFF) | ((C[118] & 0x0000FFFF) << 16);
    B[ 87] = (C[ 87] & 0x0000FFFF) | ((C[119] & 0x0000FFFF) << 16);
    B[118] = ((C[ 86] & 0xFFFF0000) >> 16) | (C[118] & 0xFFFF0000);
    B[119] = ((C[ 87] & 0xFFFF0000) >> 16) | (C[119] & 0xFFFF0000);
    B[ 88] = (C[ 88] & 0x0000FFFF) | ((C[120] & 0x0000FFFF) << 16);
    B[ 89] = (C[ 89] & 0x0000FFFF) | ((C[121] & 0x0000FFFF) << 16);
    B[120] = ((C[ 88] & 0xFFFF0000) >> 16) | (C[120] & 0xFFFF0000);
    B[121] = ((C[ 89] & 0xFFFF0000) >> 16) | (C[121] & 0xFFFF0000);
    B[ 90] = (C[ 90] & 0x0000FFFF) | ((C[122] & 0x0000FFFF) << 16);
    B[ 91] = (C[ 91] & 0x0000FFFF) | ((C[123] & 0x0000FFFF) << 16);
    B[122] = ((C[ 90] & 0xFFFF0000) >> 16) | (C[122] & 0xFFFF0000);
    B[123] = ((C[ 91] & 0xFFFF0000) >> 16) | (C[123] & 0xFFFF0000);
    B[ 92] = (C[ 92] & 0x0000FFFF) | ((C[124] & 0x0000FFFF) << 16);
    B[ 93] = (C[ 93] & 0x0000FFFF) | ((C[125] & 0x0000FFFF) << 16);
    B[124] = ((C[ 92] & 0xFFFF0000) >> 16) | (C[124] & 0xFFFF0000);
    B[125] = ((C[ 93] & 0xFFFF0000) >> 16) | (C[125] & 0xFFFF0000);
    B[ 94] = (C[ 94] & 0x0000FFFF) | ((C[126] & 0x0000FFFF) << 16);
    B[ 95] = (C[ 95] & 0x0000FFFF) | ((C[127] & 0x0000FFFF) << 16);
    B[126] = ((C[ 94] & 0xFFFF0000) >> 16) | (C[126] & 0xFFFF0000);
    B[127] = ((C[ 95] & 0xFFFF0000) >> 16) | (C[127] & 0xFFFF0000);
    A[  0] = B[  0];
    A[  1] = B[ 64];
    A[ 64] = B[  1];
    A[ 65] = B[ 65];
    A[  2] = B[  2];
    A[  3] = B[ 66];
    A[ 66] = B[  3];
    A[ 67] = B[ 67];
    A[  4] = B[  4];
    A[  5] = B[ 68];
    A[ 68] = B[  5];
    A[ 69] = B[ 69];
    A[  6] = B[  6];
    A[  7] = B[ 70];
    A[ 70] = B[  7];
    A[ 71] = B[ 71];
    A[  8] = B[  8];
    A[  9] = B[ 72];
    A[ 72] = B[  9];
    A[ 73] = B[ 73];
    A[ 10] = B[ 10];
    A[ 11] = B[ 74];
    A[ 74] = B[ 11];
    A[ 75] = B[ 75];
    A[ 12] = B[ 12];
    A[ 13] = B[ 76];
    A[ 76] = B[ 13];
    A[ 77] = B[ 77];
    A[ 14] = B[ 14];
    A[ 15] = B[ 78];
    A[ 78] = B[ 15];
    A[ 79] = B[ 79];
    A[ 16] = B[ 16];
    A[ 17] = B[ 80];
    A[ 80] = B[ 17];
    A[ 81] = B[ 81];
    A[ 18] = B[ 18];
    A[ 19] = B[ 82];
    A[ 82] = B[ 19];
    A[ 83] = B[ 83];
    A[ 20] = B[ 20];
    A[ 21] = B[ 84];
    A[ 84] = B[ 21];
    A[ 85] = B[ 85];
    A[ 22] = B[ 22];
    A[ 23] = B[ 86];
    A[ 86] = B[ 23];
    A[ 87] = B[ 87];
    A[ 24] = B[ 24];
    A[ 25] = B[ 88];
    A[ 88] = B[ 25];
    A[ 89] = B[ 89];
    A[ 26] = B[ 26];
    A[ 27] = B[ 90];
    A[ 90] = B[ 27];
    A[ 91] = B[ 91];
    A[ 28] = B[ 28];
    A[ 29] = B[ 92];
    A[ 92] = B[ 29];
    A[ 93] = B[ 93];
    A[ 30] = B[ 30];
    A[ 31] = B[ 94];
    A[ 94] = B[ 31];
    A[ 95] = B[ 95];
    A[ 32] = B[ 32];
    A[ 33] = B[ 96];
    A[ 96] = B[ 33];
    A[ 97] = B[ 97];
    A[ 34] = B[ 34];
    A[ 35] = B[ 98];
    A[ 98] = B[ 35];
    A[ 99] = B[ 99];
    A[ 36] = B[ 36];
    A[ 37] = B[100];
    A[100] = B[ 37];
    A[101] = B[101];
    A[ 38] = B[ 38];
    A[ 39] = B[102];
    A[102] = B[ 39];
    A[103] = B[103];
    A[ 40] = B[ 40];
    A[ 41] = B[104];
    A[104] = B[ 41];
    A[105] = B[105];
    A[ 42] = B[ 42];
    A[ 43] = B[106];
    A[106] = B[ 43];
    A[107] = B[107];
    A[ 44] = B[ 44];
    A[ 45] = B[108];
    A[108] = B[ 45];
    A[109] = B[109];
    A[ 46] = B[ 46];
    A[ 47] = B[110];
    A[110] = B[ 47];
    A[111] = B[111];
    A[ 48] = B[ 48];
    A[ 49] = B[112];
    A[112] = B[ 49];
    A[113] = B[113];
    A[ 50] = B[ 50];
    A[ 51] = B[114];
    A[114] = B[ 51];
    A[115] = B[115];
    A[ 52] = B[ 52];
    A[ 53] = B[116];
    A[116] = B[ 53];
    A[117] = B[117];
    A[ 54] = B[ 54];
    A[ 55] = B[118];
    A[118] = B[ 55];
    A[119] = B[119];
    A[ 56] = B[ 56];
    A[ 57] = B[120];
    A[120] = B[ 57];
    A[121] = B[121];
    A[ 58] = B[ 58];
    A[ 59] = B[122];
    A[122] = B[ 59];
    A[123] = B[123];
    A[ 60] = B[ 60];
    A[ 61] = B[124];
    A[124] = B[ 61];
    A[125] = B[125];
    A[ 62] = B[ 62];
    A[ 63] = B[126];
    A[126] = B[ 63];
    A[127] = B[127];
}



// perform a single 64*64 transpose in-place (assuming rearranged data)
static void transposeInplaceArranged64(uint4* arr) {
    unsigned int* A = reinterpret_cast<unsigned int*>(arr);
    unsigned int B[128];
    unsigned int C[128];
    B[  0] = (A[  0] & 0x55555555) | ((A[  4] & 0x55555555) << 1);
    B[  1] = (A[  1] & 0x55555555) | ((A[  5] & 0x55555555) << 1);
    B[  4] = ((A[  0] & 0xAAAAAAAA) >> 1) | (A[  4] & 0xAAAAAAAA);
    B[  5] = ((A[  1] & 0xAAAAAAAA) >> 1) | (A[  5] & 0xAAAAAAAA);
    B[  8] = (A[  8] & 0x55555555) | ((A[ 12] & 0x55555555) << 1);
    B[  9] = (A[  9] & 0x55555555) | ((A[ 13] & 0x55555555) << 1);
    B[ 12] = ((A[  8] & 0xAAAAAAAA) >> 1) | (A[ 12] & 0xAAAAAAAA);
    B[ 13] = ((A[  9] & 0xAAAAAAAA) >> 1) | (A[ 13] & 0xAAAAAAAA);
    B[ 16] = (A[ 16] & 0x55555555) | ((A[ 20] & 0x55555555) << 1);
    B[ 17] = (A[ 17] & 0x55555555) | ((A[ 21] & 0x55555555) << 1);
    B[ 20] = ((A[ 16] & 0xAAAAAAAA) >> 1) | (A[ 20] & 0xAAAAAAAA);
    B[ 21] = ((A[ 17] & 0xAAAAAAAA) >> 1) | (A[ 21] & 0xAAAAAAAA);
    B[ 24] = (A[ 24] & 0x55555555) | ((A[ 28] & 0x55555555) << 1);
    B[ 25] = (A[ 25] & 0x55555555) | ((A[ 29] & 0x55555555) << 1);
    B[ 28] = ((A[ 24] & 0xAAAAAAAA) >> 1) | (A[ 28] & 0xAAAAAAAA);
    B[ 29] = ((A[ 25] & 0xAAAAAAAA) >> 1) | (A[ 29] & 0xAAAAAAAA);
    B[ 32] = (A[ 32] & 0x55555555) | ((A[ 36] & 0x55555555) << 1);
    B[ 33] = (A[ 33] & 0x55555555) | ((A[ 37] & 0x55555555) << 1);
    B[ 36] = ((A[ 32] & 0xAAAAAAAA) >> 1) | (A[ 36] & 0xAAAAAAAA);
    B[ 37] = ((A[ 33] & 0xAAAAAAAA) >> 1) | (A[ 37] & 0xAAAAAAAA);
    B[ 40] = (A[ 40] & 0x55555555) | ((A[ 44] & 0x55555555) << 1);
    B[ 41] = (A[ 41] & 0x55555555) | ((A[ 45] & 0x55555555) << 1);
    B[ 44] = ((A[ 40] & 0xAAAAAAAA) >> 1) | (A[ 44] & 0xAAAAAAAA);
    B[ 45] = ((A[ 41] & 0xAAAAAAAA) >> 1) | (A[ 45] & 0xAAAAAAAA);
    B[ 48] = (A[ 48] & 0x55555555) | ((A[ 52] & 0x55555555) << 1);
    B[ 49] = (A[ 49] & 0x55555555) | ((A[ 53] & 0x55555555) << 1);
    B[ 52] = ((A[ 48] & 0xAAAAAAAA) >> 1) | (A[ 52] & 0xAAAAAAAA);
    B[ 53] = ((A[ 49] & 0xAAAAAAAA) >> 1) | (A[ 53] & 0xAAAAAAAA);
    B[ 56] = (A[ 56] & 0x55555555) | ((A[ 60] & 0x55555555) << 1);
    B[ 57] = (A[ 57] & 0x55555555) | ((A[ 61] & 0x55555555) << 1);
    B[ 60] = ((A[ 56] & 0xAAAAAAAA) >> 1) | (A[ 60] & 0xAAAAAAAA);
    B[ 61] = ((A[ 57] & 0xAAAAAAAA) >> 1) | (A[ 61] & 0xAAAAAAAA);
    B[ 64] = (A[ 64] & 0x55555555) | ((A[ 68] & 0x55555555) << 1);
    B[ 65] = (A[ 65] & 0x55555555) | ((A[ 69] & 0x55555555) << 1);
    B[ 68] = ((A[ 64] & 0xAAAAAAAA) >> 1) | (A[ 68] & 0xAAAAAAAA);
    B[ 69] = ((A[ 65] & 0xAAAAAAAA) >> 1) | (A[ 69] & 0xAAAAAAAA);
    B[ 72] = (A[ 72] & 0x55555555) | ((A[ 76] & 0x55555555) << 1);
    B[ 73] = (A[ 73] & 0x55555555) | ((A[ 77] & 0x55555555) << 1);
    B[ 76] = ((A[ 72] & 0xAAAAAAAA) >> 1) | (A[ 76] & 0xAAAAAAAA);
    B[ 77] = ((A[ 73] & 0xAAAAAAAA) >> 1) | (A[ 77] & 0xAAAAAAAA);
    B[ 80] = (A[ 80] & 0x55555555) | ((A[ 84] & 0x55555555) << 1);
    B[ 81] = (A[ 81] & 0x55555555) | ((A[ 85] & 0x55555555) << 1);
    B[ 84] = ((A[ 80] & 0xAAAAAAAA) >> 1) | (A[ 84] & 0xAAAAAAAA);
    B[ 85] = ((A[ 81] & 0xAAAAAAAA) >> 1) | (A[ 85] & 0xAAAAAAAA);
    B[ 88] = (A[ 88] & 0x55555555) | ((A[ 92] & 0x55555555) << 1);
    B[ 89] = (A[ 89] & 0x55555555) | ((A[ 93] & 0x55555555) << 1);
    B[ 92] = ((A[ 88] & 0xAAAAAAAA) >> 1) | (A[ 92] & 0xAAAAAAAA);
    B[ 93] = ((A[ 89] & 0xAAAAAAAA) >> 1) | (A[ 93] & 0xAAAAAAAA);
    B[ 96] = (A[ 96] & 0x55555555) | ((A[100] & 0x55555555) << 1);
    B[ 97] = (A[ 97] & 0x55555555) | ((A[101] & 0x55555555) << 1);
    B[100] = ((A[ 96] & 0xAAAAAAAA) >> 1) | (A[100] & 0xAAAAAAAA);
    B[101] = ((A[ 97] & 0xAAAAAAAA) >> 1) | (A[101] & 0xAAAAAAAA);
    B[104] = (A[104] & 0x55555555) | ((A[108] & 0x55555555) << 1);
    B[105] = (A[105] & 0x55555555) | ((A[109] & 0x55555555) << 1);
    B[108] = ((A[104] & 0xAAAAAAAA) >> 1) | (A[108] & 0xAAAAAAAA);
    B[109] = ((A[105] & 0xAAAAAAAA) >> 1) | (A[109] & 0xAAAAAAAA);
    B[112] = (A[112] & 0x55555555) | ((A[116] & 0x55555555) << 1);
    B[113] = (A[113] & 0x55555555) | ((A[117] & 0x55555555) << 1);
    B[116] = ((A[112] & 0xAAAAAAAA) >> 1) | (A[116] & 0xAAAAAAAA);
    B[117] = ((A[113] & 0xAAAAAAAA) >> 1) | (A[117] & 0xAAAAAAAA);
    B[120] = (A[120] & 0x55555555) | ((A[124] & 0x55555555) << 1);
    B[121] = (A[121] & 0x55555555) | ((A[125] & 0x55555555) << 1);
    B[124] = ((A[120] & 0xAAAAAAAA) >> 1) | (A[124] & 0xAAAAAAAA);
    B[125] = ((A[121] & 0xAAAAAAAA) >> 1) | (A[125] & 0xAAAAAAAA);
    B[  2] = (A[  2] & 0x55555555) | ((A[  6] & 0x55555555) << 1);
    B[  3] = (A[  3] & 0x55555555) | ((A[  7] & 0x55555555) << 1);
    B[  6] = ((A[  2] & 0xAAAAAAAA) >> 1) | (A[  6] & 0xAAAAAAAA);
    B[  7] = ((A[  3] & 0xAAAAAAAA) >> 1) | (A[  7] & 0xAAAAAAAA);
    B[ 10] = (A[ 10] & 0x55555555) | ((A[ 14] & 0x55555555) << 1);
    B[ 11] = (A[ 11] & 0x55555555) | ((A[ 15] & 0x55555555) << 1);
    B[ 14] = ((A[ 10] & 0xAAAAAAAA) >> 1) | (A[ 14] & 0xAAAAAAAA);
    B[ 15] = ((A[ 11] & 0xAAAAAAAA) >> 1) | (A[ 15] & 0xAAAAAAAA);
    B[ 18] = (A[ 18] & 0x55555555) | ((A[ 22] & 0x55555555) << 1);
    B[ 19] = (A[ 19] & 0x55555555) | ((A[ 23] & 0x55555555) << 1);
    B[ 22] = ((A[ 18] & 0xAAAAAAAA) >> 1) | (A[ 22] & 0xAAAAAAAA);
    B[ 23] = ((A[ 19] & 0xAAAAAAAA) >> 1) | (A[ 23] & 0xAAAAAAAA);
    B[ 26] = (A[ 26] & 0x55555555) | ((A[ 30] & 0x55555555) << 1);
    B[ 27] = (A[ 27] & 0x55555555) | ((A[ 31] & 0x55555555) << 1);
    B[ 30] = ((A[ 26] & 0xAAAAAAAA) >> 1) | (A[ 30] & 0xAAAAAAAA);
    B[ 31] = ((A[ 27] & 0xAAAAAAAA) >> 1) | (A[ 31] & 0xAAAAAAAA);
    B[ 34] = (A[ 34] & 0x55555555) | ((A[ 38] & 0x55555555) << 1);
    B[ 35] = (A[ 35] & 0x55555555) | ((A[ 39] & 0x55555555) << 1);
    B[ 38] = ((A[ 34] & 0xAAAAAAAA) >> 1) | (A[ 38] & 0xAAAAAAAA);
    B[ 39] = ((A[ 35] & 0xAAAAAAAA) >> 1) | (A[ 39] & 0xAAAAAAAA);
    B[ 42] = (A[ 42] & 0x55555555) | ((A[ 46] & 0x55555555) << 1);
    B[ 43] = (A[ 43] & 0x55555555) | ((A[ 47] & 0x55555555) << 1);
    B[ 46] = ((A[ 42] & 0xAAAAAAAA) >> 1) | (A[ 46] & 0xAAAAAAAA);
    B[ 47] = ((A[ 43] & 0xAAAAAAAA) >> 1) | (A[ 47] & 0xAAAAAAAA);
    B[ 50] = (A[ 50] & 0x55555555) | ((A[ 54] & 0x55555555) << 1);
    B[ 51] = (A[ 51] & 0x55555555) | ((A[ 55] & 0x55555555) << 1);
    B[ 54] = ((A[ 50] & 0xAAAAAAAA) >> 1) | (A[ 54] & 0xAAAAAAAA);
    B[ 55] = ((A[ 51] & 0xAAAAAAAA) >> 1) | (A[ 55] & 0xAAAAAAAA);
    B[ 58] = (A[ 58] & 0x55555555) | ((A[ 62] & 0x55555555) << 1);
    B[ 59] = (A[ 59] & 0x55555555) | ((A[ 63] & 0x55555555) << 1);
    B[ 62] = ((A[ 58] & 0xAAAAAAAA) >> 1) | (A[ 62] & 0xAAAAAAAA);
    B[ 63] = ((A[ 59] & 0xAAAAAAAA) >> 1) | (A[ 63] & 0xAAAAAAAA);
    B[ 66] = (A[ 66] & 0x55555555) | ((A[ 70] & 0x55555555) << 1);
    B[ 67] = (A[ 67] & 0x55555555) | ((A[ 71] & 0x55555555) << 1);
    B[ 70] = ((A[ 66] & 0xAAAAAAAA) >> 1) | (A[ 70] & 0xAAAAAAAA);
    B[ 71] = ((A[ 67] & 0xAAAAAAAA) >> 1) | (A[ 71] & 0xAAAAAAAA);
    B[ 74] = (A[ 74] & 0x55555555) | ((A[ 78] & 0x55555555) << 1);
    B[ 75] = (A[ 75] & 0x55555555) | ((A[ 79] & 0x55555555) << 1);
    B[ 78] = ((A[ 74] & 0xAAAAAAAA) >> 1) | (A[ 78] & 0xAAAAAAAA);
    B[ 79] = ((A[ 75] & 0xAAAAAAAA) >> 1) | (A[ 79] & 0xAAAAAAAA);
    B[ 82] = (A[ 82] & 0x55555555) | ((A[ 86] & 0x55555555) << 1);
    B[ 83] = (A[ 83] & 0x55555555) | ((A[ 87] & 0x55555555) << 1);
    B[ 86] = ((A[ 82] & 0xAAAAAAAA) >> 1) | (A[ 86] & 0xAAAAAAAA);
    B[ 87] = ((A[ 83] & 0xAAAAAAAA) >> 1) | (A[ 87] & 0xAAAAAAAA);
    B[ 90] = (A[ 90] & 0x55555555) | ((A[ 94] & 0x55555555) << 1);
    B[ 91] = (A[ 91] & 0x55555555) | ((A[ 95] & 0x55555555) << 1);
    B[ 94] = ((A[ 90] & 0xAAAAAAAA) >> 1) | (A[ 94] & 0xAAAAAAAA);
    B[ 95] = ((A[ 91] & 0xAAAAAAAA) >> 1) | (A[ 95] & 0xAAAAAAAA);
    B[ 98] = (A[ 98] & 0x55555555) | ((A[102] & 0x55555555) << 1);
    B[ 99] = (A[ 99] & 0x55555555) | ((A[103] & 0x55555555) << 1);
    B[102] = ((A[ 98] & 0xAAAAAAAA) >> 1) | (A[102] & 0xAAAAAAAA);
    B[103] = ((A[ 99] & 0xAAAAAAAA) >> 1) | (A[103] & 0xAAAAAAAA);
    B[106] = (A[106] & 0x55555555) | ((A[110] & 0x55555555) << 1);
    B[107] = (A[107] & 0x55555555) | ((A[111] & 0x55555555) << 1);
    B[110] = ((A[106] & 0xAAAAAAAA) >> 1) | (A[110] & 0xAAAAAAAA);
    B[111] = ((A[107] & 0xAAAAAAAA) >> 1) | (A[111] & 0xAAAAAAAA);
    B[114] = (A[114] & 0x55555555) | ((A[118] & 0x55555555) << 1);
    B[115] = (A[115] & 0x55555555) | ((A[119] & 0x55555555) << 1);
    B[118] = ((A[114] & 0xAAAAAAAA) >> 1) | (A[118] & 0xAAAAAAAA);
    B[119] = ((A[115] & 0xAAAAAAAA) >> 1) | (A[119] & 0xAAAAAAAA);
    B[122] = (A[122] & 0x55555555) | ((A[126] & 0x55555555) << 1);
    B[123] = (A[123] & 0x55555555) | ((A[127] & 0x55555555) << 1);
    B[126] = ((A[122] & 0xAAAAAAAA) >> 1) | (A[126] & 0xAAAAAAAA);
    B[127] = ((A[123] & 0xAAAAAAAA) >> 1) | (A[127] & 0xAAAAAAAA);
    C[  0] = (B[  0] & 0x33333333) | ((B[  8] & 0x33333333) << 2);
    C[  1] = (B[  1] & 0x33333333) | ((B[  9] & 0x33333333) << 2);
    C[  8] = ((B[  0] & 0xCCCCCCCC) >> 2) | (B[  8] & 0xCCCCCCCC);
    C[  9] = ((B[  1] & 0xCCCCCCCC) >> 2) | (B[  9] & 0xCCCCCCCC);
    C[  4] = (B[  4] & 0x33333333) | ((B[ 12] & 0x33333333) << 2);
    C[  5] = (B[  5] & 0x33333333) | ((B[ 13] & 0x33333333) << 2);
    C[ 12] = ((B[  4] & 0xCCCCCCCC) >> 2) | (B[ 12] & 0xCCCCCCCC);
    C[ 13] = ((B[  5] & 0xCCCCCCCC) >> 2) | (B[ 13] & 0xCCCCCCCC);
    C[ 16] = (B[ 16] & 0x33333333) | ((B[ 24] & 0x33333333) << 2);
    C[ 17] = (B[ 17] & 0x33333333) | ((B[ 25] & 0x33333333) << 2);
    C[ 24] = ((B[ 16] & 0xCCCCCCCC) >> 2) | (B[ 24] & 0xCCCCCCCC);
    C[ 25] = ((B[ 17] & 0xCCCCCCCC) >> 2) | (B[ 25] & 0xCCCCCCCC);
    C[ 20] = (B[ 20] & 0x33333333) | ((B[ 28] & 0x33333333) << 2);
    C[ 21] = (B[ 21] & 0x33333333) | ((B[ 29] & 0x33333333) << 2);
    C[ 28] = ((B[ 20] & 0xCCCCCCCC) >> 2) | (B[ 28] & 0xCCCCCCCC);
    C[ 29] = ((B[ 21] & 0xCCCCCCCC) >> 2) | (B[ 29] & 0xCCCCCCCC);
    C[ 32] = (B[ 32] & 0x33333333) | ((B[ 40] & 0x33333333) << 2);
    C[ 33] = (B[ 33] & 0x33333333) | ((B[ 41] & 0x33333333) << 2);
    C[ 40] = ((B[ 32] & 0xCCCCCCCC) >> 2) | (B[ 40] & 0xCCCCCCCC);
    C[ 41] = ((B[ 33] & 0xCCCCCCCC) >> 2) | (B[ 41] & 0xCCCCCCCC);
    C[ 36] = (B[ 36] & 0x33333333) | ((B[ 44] & 0x33333333) << 2);
    C[ 37] = (B[ 37] & 0x33333333) | ((B[ 45] & 0x33333333) << 2);
    C[ 44] = ((B[ 36] & 0xCCCCCCCC) >> 2) | (B[ 44] & 0xCCCCCCCC);
    C[ 45] = ((B[ 37] & 0xCCCCCCCC) >> 2) | (B[ 45] & 0xCCCCCCCC);
    C[ 48] = (B[ 48] & 0x33333333) | ((B[ 56] & 0x33333333) << 2);
    C[ 49] = (B[ 49] & 0x33333333) | ((B[ 57] & 0x33333333) << 2);
    C[ 56] = ((B[ 48] & 0xCCCCCCCC) >> 2) | (B[ 56] & 0xCCCCCCCC);
    C[ 57] = ((B[ 49] & 0xCCCCCCCC) >> 2) | (B[ 57] & 0xCCCCCCCC);
    C[ 52] = (B[ 52] & 0x33333333) | ((B[ 60] & 0x33333333) << 2);
    C[ 53] = (B[ 53] & 0x33333333) | ((B[ 61] & 0x33333333) << 2);
    C[ 60] = ((B[ 52] & 0xCCCCCCCC) >> 2) | (B[ 60] & 0xCCCCCCCC);
    C[ 61] = ((B[ 53] & 0xCCCCCCCC) >> 2) | (B[ 61] & 0xCCCCCCCC);
    C[ 64] = (B[ 64] & 0x33333333) | ((B[ 72] & 0x33333333) << 2);
    C[ 65] = (B[ 65] & 0x33333333) | ((B[ 73] & 0x33333333) << 2);
    C[ 72] = ((B[ 64] & 0xCCCCCCCC) >> 2) | (B[ 72] & 0xCCCCCCCC);
    C[ 73] = ((B[ 65] & 0xCCCCCCCC) >> 2) | (B[ 73] & 0xCCCCCCCC);
    C[ 68] = (B[ 68] & 0x33333333) | ((B[ 76] & 0x33333333) << 2);
    C[ 69] = (B[ 69] & 0x33333333) | ((B[ 77] & 0x33333333) << 2);
    C[ 76] = ((B[ 68] & 0xCCCCCCCC) >> 2) | (B[ 76] & 0xCCCCCCCC);
    C[ 77] = ((B[ 69] & 0xCCCCCCCC) >> 2) | (B[ 77] & 0xCCCCCCCC);
    C[ 80] = (B[ 80] & 0x33333333) | ((B[ 88] & 0x33333333) << 2);
    C[ 81] = (B[ 81] & 0x33333333) | ((B[ 89] & 0x33333333) << 2);
    C[ 88] = ((B[ 80] & 0xCCCCCCCC) >> 2) | (B[ 88] & 0xCCCCCCCC);
    C[ 89] = ((B[ 81] & 0xCCCCCCCC) >> 2) | (B[ 89] & 0xCCCCCCCC);
    C[ 84] = (B[ 84] & 0x33333333) | ((B[ 92] & 0x33333333) << 2);
    C[ 85] = (B[ 85] & 0x33333333) | ((B[ 93] & 0x33333333) << 2);
    C[ 92] = ((B[ 84] & 0xCCCCCCCC) >> 2) | (B[ 92] & 0xCCCCCCCC);
    C[ 93] = ((B[ 85] & 0xCCCCCCCC) >> 2) | (B[ 93] & 0xCCCCCCCC);
    C[ 96] = (B[ 96] & 0x33333333) | ((B[104] & 0x33333333) << 2);
    C[ 97] = (B[ 97] & 0x33333333) | ((B[105] & 0x33333333) << 2);
    C[104] = ((B[ 96] & 0xCCCCCCCC) >> 2) | (B[104] & 0xCCCCCCCC);
    C[105] = ((B[ 97] & 0xCCCCCCCC) >> 2) | (B[105] & 0xCCCCCCCC);
    C[100] = (B[100] & 0x33333333) | ((B[108] & 0x33333333) << 2);
    C[101] = (B[101] & 0x33333333) | ((B[109] & 0x33333333) << 2);
    C[108] = ((B[100] & 0xCCCCCCCC) >> 2) | (B[108] & 0xCCCCCCCC);
    C[109] = ((B[101] & 0xCCCCCCCC) >> 2) | (B[109] & 0xCCCCCCCC);
    C[112] = (B[112] & 0x33333333) | ((B[120] & 0x33333333) << 2);
    C[113] = (B[113] & 0x33333333) | ((B[121] & 0x33333333) << 2);
    C[120] = ((B[112] & 0xCCCCCCCC) >> 2) | (B[120] & 0xCCCCCCCC);
    C[121] = ((B[113] & 0xCCCCCCCC) >> 2) | (B[121] & 0xCCCCCCCC);
    C[116] = (B[116] & 0x33333333) | ((B[124] & 0x33333333) << 2);
    C[117] = (B[117] & 0x33333333) | ((B[125] & 0x33333333) << 2);
    C[124] = ((B[116] & 0xCCCCCCCC) >> 2) | (B[124] & 0xCCCCCCCC);
    C[125] = ((B[117] & 0xCCCCCCCC) >> 2) | (B[125] & 0xCCCCCCCC);
    C[  2] = (B[  2] & 0x33333333) | ((B[ 10] & 0x33333333) << 2);
    C[  3] = (B[  3] & 0x33333333) | ((B[ 11] & 0x33333333) << 2);
    C[ 10] = ((B[  2] & 0xCCCCCCCC) >> 2) | (B[ 10] & 0xCCCCCCCC);
    C[ 11] = ((B[  3] & 0xCCCCCCCC) >> 2) | (B[ 11] & 0xCCCCCCCC);
    C[  6] = (B[  6] & 0x33333333) | ((B[ 14] & 0x33333333) << 2);
    C[  7] = (B[  7] & 0x33333333) | ((B[ 15] & 0x33333333) << 2);
    C[ 14] = ((B[  6] & 0xCCCCCCCC) >> 2) | (B[ 14] & 0xCCCCCCCC);
    C[ 15] = ((B[  7] & 0xCCCCCCCC) >> 2) | (B[ 15] & 0xCCCCCCCC);
    C[ 18] = (B[ 18] & 0x33333333) | ((B[ 26] & 0x33333333) << 2);
    C[ 19] = (B[ 19] & 0x33333333) | ((B[ 27] & 0x33333333) << 2);
    C[ 26] = ((B[ 18] & 0xCCCCCCCC) >> 2) | (B[ 26] & 0xCCCCCCCC);
    C[ 27] = ((B[ 19] & 0xCCCCCCCC) >> 2) | (B[ 27] & 0xCCCCCCCC);
    C[ 22] = (B[ 22] & 0x33333333) | ((B[ 30] & 0x33333333) << 2);
    C[ 23] = (B[ 23] & 0x33333333) | ((B[ 31] & 0x33333333) << 2);
    C[ 30] = ((B[ 22] & 0xCCCCCCCC) >> 2) | (B[ 30] & 0xCCCCCCCC);
    C[ 31] = ((B[ 23] & 0xCCCCCCCC) >> 2) | (B[ 31] & 0xCCCCCCCC);
    C[ 34] = (B[ 34] & 0x33333333) | ((B[ 42] & 0x33333333) << 2);
    C[ 35] = (B[ 35] & 0x33333333) | ((B[ 43] & 0x33333333) << 2);
    C[ 42] = ((B[ 34] & 0xCCCCCCCC) >> 2) | (B[ 42] & 0xCCCCCCCC);
    C[ 43] = ((B[ 35] & 0xCCCCCCCC) >> 2) | (B[ 43] & 0xCCCCCCCC);
    C[ 38] = (B[ 38] & 0x33333333) | ((B[ 46] & 0x33333333) << 2);
    C[ 39] = (B[ 39] & 0x33333333) | ((B[ 47] & 0x33333333) << 2);
    C[ 46] = ((B[ 38] & 0xCCCCCCCC) >> 2) | (B[ 46] & 0xCCCCCCCC);
    C[ 47] = ((B[ 39] & 0xCCCCCCCC) >> 2) | (B[ 47] & 0xCCCCCCCC);
    C[ 50] = (B[ 50] & 0x33333333) | ((B[ 58] & 0x33333333) << 2);
    C[ 51] = (B[ 51] & 0x33333333) | ((B[ 59] & 0x33333333) << 2);
    C[ 58] = ((B[ 50] & 0xCCCCCCCC) >> 2) | (B[ 58] & 0xCCCCCCCC);
    C[ 59] = ((B[ 51] & 0xCCCCCCCC) >> 2) | (B[ 59] & 0xCCCCCCCC);
    C[ 54] = (B[ 54] & 0x33333333) | ((B[ 62] & 0x33333333) << 2);
    C[ 55] = (B[ 55] & 0x33333333) | ((B[ 63] & 0x33333333) << 2);
    C[ 62] = ((B[ 54] & 0xCCCCCCCC) >> 2) | (B[ 62] & 0xCCCCCCCC);
    C[ 63] = ((B[ 55] & 0xCCCCCCCC) >> 2) | (B[ 63] & 0xCCCCCCCC);
    C[ 66] = (B[ 66] & 0x33333333) | ((B[ 74] & 0x33333333) << 2);
    C[ 67] = (B[ 67] & 0x33333333) | ((B[ 75] & 0x33333333) << 2);
    C[ 74] = ((B[ 66] & 0xCCCCCCCC) >> 2) | (B[ 74] & 0xCCCCCCCC);
    C[ 75] = ((B[ 67] & 0xCCCCCCCC) >> 2) | (B[ 75] & 0xCCCCCCCC);
    C[ 70] = (B[ 70] & 0x33333333) | ((B[ 78] & 0x33333333) << 2);
    C[ 71] = (B[ 71] & 0x33333333) | ((B[ 79] & 0x33333333) << 2);
    C[ 78] = ((B[ 70] & 0xCCCCCCCC) >> 2) | (B[ 78] & 0xCCCCCCCC);
    C[ 79] = ((B[ 71] & 0xCCCCCCCC) >> 2) | (B[ 79] & 0xCCCCCCCC);
    C[ 82] = (B[ 82] & 0x33333333) | ((B[ 90] & 0x33333333) << 2);
    C[ 83] = (B[ 83] & 0x33333333) | ((B[ 91] & 0x33333333) << 2);
    C[ 90] = ((B[ 82] & 0xCCCCCCCC) >> 2) | (B[ 90] & 0xCCCCCCCC);
    C[ 91] = ((B[ 83] & 0xCCCCCCCC) >> 2) | (B[ 91] & 0xCCCCCCCC);
    C[ 86] = (B[ 86] & 0x33333333) | ((B[ 94] & 0x33333333) << 2);
    C[ 87] = (B[ 87] & 0x33333333) | ((B[ 95] & 0x33333333) << 2);
    C[ 94] = ((B[ 86] & 0xCCCCCCCC) >> 2) | (B[ 94] & 0xCCCCCCCC);
    C[ 95] = ((B[ 87] & 0xCCCCCCCC) >> 2) | (B[ 95] & 0xCCCCCCCC);
    C[ 98] = (B[ 98] & 0x33333333) | ((B[106] & 0x33333333) << 2);
    C[ 99] = (B[ 99] & 0x33333333) | ((B[107] & 0x33333333) << 2);
    C[106] = ((B[ 98] & 0xCCCCCCCC) >> 2) | (B[106] & 0xCCCCCCCC);
    C[107] = ((B[ 99] & 0xCCCCCCCC) >> 2) | (B[107] & 0xCCCCCCCC);
    C[102] = (B[102] & 0x33333333) | ((B[110] & 0x33333333) << 2);
    C[103] = (B[103] & 0x33333333) | ((B[111] & 0x33333333) << 2);
    C[110] = ((B[102] & 0xCCCCCCCC) >> 2) | (B[110] & 0xCCCCCCCC);
    C[111] = ((B[103] & 0xCCCCCCCC) >> 2) | (B[111] & 0xCCCCCCCC);
    C[114] = (B[114] & 0x33333333) | ((B[122] & 0x33333333) << 2);
    C[115] = (B[115] & 0x33333333) | ((B[123] & 0x33333333) << 2);
    C[122] = ((B[114] & 0xCCCCCCCC) >> 2) | (B[122] & 0xCCCCCCCC);
    C[123] = ((B[115] & 0xCCCCCCCC) >> 2) | (B[123] & 0xCCCCCCCC);
    C[118] = (B[118] & 0x33333333) | ((B[126] & 0x33333333) << 2);
    C[119] = (B[119] & 0x33333333) | ((B[127] & 0x33333333) << 2);
    C[126] = ((B[118] & 0xCCCCCCCC) >> 2) | (B[126] & 0xCCCCCCCC);
    C[127] = ((B[119] & 0xCCCCCCCC) >> 2) | (B[127] & 0xCCCCCCCC);
    B[  0] = (C[  0] & 0x0F0F0F0F) | ((C[ 16] & 0x0F0F0F0F) << 4);
    B[  1] = (C[  1] & 0x0F0F0F0F) | ((C[ 17] & 0x0F0F0F0F) << 4);
    B[ 16] = ((C[  0] & 0xF0F0F0F0) >> 4) | (C[ 16] & 0xF0F0F0F0);
    B[ 17] = ((C[  1] & 0xF0F0F0F0) >> 4) | (C[ 17] & 0xF0F0F0F0);
    B[  4] = (C[  4] & 0x0F0F0F0F) | ((C[ 20] & 0x0F0F0F0F) << 4);
    B[  5] = (C[  5] & 0x0F0F0F0F) | ((C[ 21] & 0x0F0F0F0F) << 4);
    B[ 20] = ((C[  4] & 0xF0F0F0F0) >> 4) | (C[ 20] & 0xF0F0F0F0);
    B[ 21] = ((C[  5] & 0xF0F0F0F0) >> 4) | (C[ 21] & 0xF0F0F0F0);
    B[  8] = (C[  8] & 0x0F0F0F0F) | ((C[ 24] & 0x0F0F0F0F) << 4);
    B[  9] = (C[  9] & 0x0F0F0F0F) | ((C[ 25] & 0x0F0F0F0F) << 4);
    B[ 24] = ((C[  8] & 0xF0F0F0F0) >> 4) | (C[ 24] & 0xF0F0F0F0);
    B[ 25] = ((C[  9] & 0xF0F0F0F0) >> 4) | (C[ 25] & 0xF0F0F0F0);
    B[ 12] = (C[ 12] & 0x0F0F0F0F) | ((C[ 28] & 0x0F0F0F0F) << 4);
    B[ 13] = (C[ 13] & 0x0F0F0F0F) | ((C[ 29] & 0x0F0F0F0F) << 4);
    B[ 28] = ((C[ 12] & 0xF0F0F0F0) >> 4) | (C[ 28] & 0xF0F0F0F0);
    B[ 29] = ((C[ 13] & 0xF0F0F0F0) >> 4) | (C[ 29] & 0xF0F0F0F0);
    B[ 32] = (C[ 32] & 0x0F0F0F0F) | ((C[ 48] & 0x0F0F0F0F) << 4);
    B[ 33] = (C[ 33] & 0x0F0F0F0F) | ((C[ 49] & 0x0F0F0F0F) << 4);
    B[ 48] = ((C[ 32] & 0xF0F0F0F0) >> 4) | (C[ 48] & 0xF0F0F0F0);
    B[ 49] = ((C[ 33] & 0xF0F0F0F0) >> 4) | (C[ 49] & 0xF0F0F0F0);
    B[ 36] = (C[ 36] & 0x0F0F0F0F) | ((C[ 52] & 0x0F0F0F0F) << 4);
    B[ 37] = (C[ 37] & 0x0F0F0F0F) | ((C[ 53] & 0x0F0F0F0F) << 4);
    B[ 52] = ((C[ 36] & 0xF0F0F0F0) >> 4) | (C[ 52] & 0xF0F0F0F0);
    B[ 53] = ((C[ 37] & 0xF0F0F0F0) >> 4) | (C[ 53] & 0xF0F0F0F0);
    B[ 40] = (C[ 40] & 0x0F0F0F0F) | ((C[ 56] & 0x0F0F0F0F) << 4);
    B[ 41] = (C[ 41] & 0x0F0F0F0F) | ((C[ 57] & 0x0F0F0F0F) << 4);
    B[ 56] = ((C[ 40] & 0xF0F0F0F0) >> 4) | (C[ 56] & 0xF0F0F0F0);
    B[ 57] = ((C[ 41] & 0xF0F0F0F0) >> 4) | (C[ 57] & 0xF0F0F0F0);
    B[ 44] = (C[ 44] & 0x0F0F0F0F) | ((C[ 60] & 0x0F0F0F0F) << 4);
    B[ 45] = (C[ 45] & 0x0F0F0F0F) | ((C[ 61] & 0x0F0F0F0F) << 4);
    B[ 60] = ((C[ 44] & 0xF0F0F0F0) >> 4) | (C[ 60] & 0xF0F0F0F0);
    B[ 61] = ((C[ 45] & 0xF0F0F0F0) >> 4) | (C[ 61] & 0xF0F0F0F0);
    B[ 64] = (C[ 64] & 0x0F0F0F0F) | ((C[ 80] & 0x0F0F0F0F) << 4);
    B[ 65] = (C[ 65] & 0x0F0F0F0F) | ((C[ 81] & 0x0F0F0F0F) << 4);
    B[ 80] = ((C[ 64] & 0xF0F0F0F0) >> 4) | (C[ 80] & 0xF0F0F0F0);
    B[ 81] = ((C[ 65] & 0xF0F0F0F0) >> 4) | (C[ 81] & 0xF0F0F0F0);
    B[ 68] = (C[ 68] & 0x0F0F0F0F) | ((C[ 84] & 0x0F0F0F0F) << 4);
    B[ 69] = (C[ 69] & 0x0F0F0F0F) | ((C[ 85] & 0x0F0F0F0F) << 4);
    B[ 84] = ((C[ 68] & 0xF0F0F0F0) >> 4) | (C[ 84] & 0xF0F0F0F0);
    B[ 85] = ((C[ 69] & 0xF0F0F0F0) >> 4) | (C[ 85] & 0xF0F0F0F0);
    B[ 72] = (C[ 72] & 0x0F0F0F0F) | ((C[ 88] & 0x0F0F0F0F) << 4);
    B[ 73] = (C[ 73] & 0x0F0F0F0F) | ((C[ 89] & 0x0F0F0F0F) << 4);
    B[ 88] = ((C[ 72] & 0xF0F0F0F0) >> 4) | (C[ 88] & 0xF0F0F0F0);
    B[ 89] = ((C[ 73] & 0xF0F0F0F0) >> 4) | (C[ 89] & 0xF0F0F0F0);
    B[ 76] = (C[ 76] & 0x0F0F0F0F) | ((C[ 92] & 0x0F0F0F0F) << 4);
    B[ 77] = (C[ 77] & 0x0F0F0F0F) | ((C[ 93] & 0x0F0F0F0F) << 4);
    B[ 92] = ((C[ 76] & 0xF0F0F0F0) >> 4) | (C[ 92] & 0xF0F0F0F0);
    B[ 93] = ((C[ 77] & 0xF0F0F0F0) >> 4) | (C[ 93] & 0xF0F0F0F0);
    B[ 96] = (C[ 96] & 0x0F0F0F0F) | ((C[112] & 0x0F0F0F0F) << 4);
    B[ 97] = (C[ 97] & 0x0F0F0F0F) | ((C[113] & 0x0F0F0F0F) << 4);
    B[112] = ((C[ 96] & 0xF0F0F0F0) >> 4) | (C[112] & 0xF0F0F0F0);
    B[113] = ((C[ 97] & 0xF0F0F0F0) >> 4) | (C[113] & 0xF0F0F0F0);
    B[100] = (C[100] & 0x0F0F0F0F) | ((C[116] & 0x0F0F0F0F) << 4);
    B[101] = (C[101] & 0x0F0F0F0F) | ((C[117] & 0x0F0F0F0F) << 4);
    B[116] = ((C[100] & 0xF0F0F0F0) >> 4) | (C[116] & 0xF0F0F0F0);
    B[117] = ((C[101] & 0xF0F0F0F0) >> 4) | (C[117] & 0xF0F0F0F0);
    B[104] = (C[104] & 0x0F0F0F0F) | ((C[120] & 0x0F0F0F0F) << 4);
    B[105] = (C[105] & 0x0F0F0F0F) | ((C[121] & 0x0F0F0F0F) << 4);
    B[120] = ((C[104] & 0xF0F0F0F0) >> 4) | (C[120] & 0xF0F0F0F0);
    B[121] = ((C[105] & 0xF0F0F0F0) >> 4) | (C[121] & 0xF0F0F0F0);
    B[108] = (C[108] & 0x0F0F0F0F) | ((C[124] & 0x0F0F0F0F) << 4);
    B[109] = (C[109] & 0x0F0F0F0F) | ((C[125] & 0x0F0F0F0F) << 4);
    B[124] = ((C[108] & 0xF0F0F0F0) >> 4) | (C[124] & 0xF0F0F0F0);
    B[125] = ((C[109] & 0xF0F0F0F0) >> 4) | (C[125] & 0xF0F0F0F0);
    B[  2] = (C[  2] & 0x0F0F0F0F) | ((C[ 18] & 0x0F0F0F0F) << 4);
    B[  3] = (C[  3] & 0x0F0F0F0F) | ((C[ 19] & 0x0F0F0F0F) << 4);
    B[ 18] = ((C[  2] & 0xF0F0F0F0) >> 4) | (C[ 18] & 0xF0F0F0F0);
    B[ 19] = ((C[  3] & 0xF0F0F0F0) >> 4) | (C[ 19] & 0xF0F0F0F0);
    B[  6] = (C[  6] & 0x0F0F0F0F) | ((C[ 22] & 0x0F0F0F0F) << 4);
    B[  7] = (C[  7] & 0x0F0F0F0F) | ((C[ 23] & 0x0F0F0F0F) << 4);
    B[ 22] = ((C[  6] & 0xF0F0F0F0) >> 4) | (C[ 22] & 0xF0F0F0F0);
    B[ 23] = ((C[  7] & 0xF0F0F0F0) >> 4) | (C[ 23] & 0xF0F0F0F0);
    B[ 10] = (C[ 10] & 0x0F0F0F0F) | ((C[ 26] & 0x0F0F0F0F) << 4);
    B[ 11] = (C[ 11] & 0x0F0F0F0F) | ((C[ 27] & 0x0F0F0F0F) << 4);
    B[ 26] = ((C[ 10] & 0xF0F0F0F0) >> 4) | (C[ 26] & 0xF0F0F0F0);
    B[ 27] = ((C[ 11] & 0xF0F0F0F0) >> 4) | (C[ 27] & 0xF0F0F0F0);
    B[ 14] = (C[ 14] & 0x0F0F0F0F) | ((C[ 30] & 0x0F0F0F0F) << 4);
    B[ 15] = (C[ 15] & 0x0F0F0F0F) | ((C[ 31] & 0x0F0F0F0F) << 4);
    B[ 30] = ((C[ 14] & 0xF0F0F0F0) >> 4) | (C[ 30] & 0xF0F0F0F0);
    B[ 31] = ((C[ 15] & 0xF0F0F0F0) >> 4) | (C[ 31] & 0xF0F0F0F0);
    B[ 34] = (C[ 34] & 0x0F0F0F0F) | ((C[ 50] & 0x0F0F0F0F) << 4);
    B[ 35] = (C[ 35] & 0x0F0F0F0F) | ((C[ 51] & 0x0F0F0F0F) << 4);
    B[ 50] = ((C[ 34] & 0xF0F0F0F0) >> 4) | (C[ 50] & 0xF0F0F0F0);
    B[ 51] = ((C[ 35] & 0xF0F0F0F0) >> 4) | (C[ 51] & 0xF0F0F0F0);
    B[ 38] = (C[ 38] & 0x0F0F0F0F) | ((C[ 54] & 0x0F0F0F0F) << 4);
    B[ 39] = (C[ 39] & 0x0F0F0F0F) | ((C[ 55] & 0x0F0F0F0F) << 4);
    B[ 54] = ((C[ 38] & 0xF0F0F0F0) >> 4) | (C[ 54] & 0xF0F0F0F0);
    B[ 55] = ((C[ 39] & 0xF0F0F0F0) >> 4) | (C[ 55] & 0xF0F0F0F0);
    B[ 42] = (C[ 42] & 0x0F0F0F0F) | ((C[ 58] & 0x0F0F0F0F) << 4);
    B[ 43] = (C[ 43] & 0x0F0F0F0F) | ((C[ 59] & 0x0F0F0F0F) << 4);
    B[ 58] = ((C[ 42] & 0xF0F0F0F0) >> 4) | (C[ 58] & 0xF0F0F0F0);
    B[ 59] = ((C[ 43] & 0xF0F0F0F0) >> 4) | (C[ 59] & 0xF0F0F0F0);
    B[ 46] = (C[ 46] & 0x0F0F0F0F) | ((C[ 62] & 0x0F0F0F0F) << 4);
    B[ 47] = (C[ 47] & 0x0F0F0F0F) | ((C[ 63] & 0x0F0F0F0F) << 4);
    B[ 62] = ((C[ 46] & 0xF0F0F0F0) >> 4) | (C[ 62] & 0xF0F0F0F0);
    B[ 63] = ((C[ 47] & 0xF0F0F0F0) >> 4) | (C[ 63] & 0xF0F0F0F0);
    B[ 66] = (C[ 66] & 0x0F0F0F0F) | ((C[ 82] & 0x0F0F0F0F) << 4);
    B[ 67] = (C[ 67] & 0x0F0F0F0F) | ((C[ 83] & 0x0F0F0F0F) << 4);
    B[ 82] = ((C[ 66] & 0xF0F0F0F0) >> 4) | (C[ 82] & 0xF0F0F0F0);
    B[ 83] = ((C[ 67] & 0xF0F0F0F0) >> 4) | (C[ 83] & 0xF0F0F0F0);
    B[ 70] = (C[ 70] & 0x0F0F0F0F) | ((C[ 86] & 0x0F0F0F0F) << 4);
    B[ 71] = (C[ 71] & 0x0F0F0F0F) | ((C[ 87] & 0x0F0F0F0F) << 4);
    B[ 86] = ((C[ 70] & 0xF0F0F0F0) >> 4) | (C[ 86] & 0xF0F0F0F0);
    B[ 87] = ((C[ 71] & 0xF0F0F0F0) >> 4) | (C[ 87] & 0xF0F0F0F0);
    B[ 74] = (C[ 74] & 0x0F0F0F0F) | ((C[ 90] & 0x0F0F0F0F) << 4);
    B[ 75] = (C[ 75] & 0x0F0F0F0F) | ((C[ 91] & 0x0F0F0F0F) << 4);
    B[ 90] = ((C[ 74] & 0xF0F0F0F0) >> 4) | (C[ 90] & 0xF0F0F0F0);
    B[ 91] = ((C[ 75] & 0xF0F0F0F0) >> 4) | (C[ 91] & 0xF0F0F0F0);
    B[ 78] = (C[ 78] & 0x0F0F0F0F) | ((C[ 94] & 0x0F0F0F0F) << 4);
    B[ 79] = (C[ 79] & 0x0F0F0F0F) | ((C[ 95] & 0x0F0F0F0F) << 4);
    B[ 94] = ((C[ 78] & 0xF0F0F0F0) >> 4) | (C[ 94] & 0xF0F0F0F0);
    B[ 95] = ((C[ 79] & 0xF0F0F0F0) >> 4) | (C[ 95] & 0xF0F0F0F0);
    B[ 98] = (C[ 98] & 0x0F0F0F0F) | ((C[114] & 0x0F0F0F0F) << 4);
    B[ 99] = (C[ 99] & 0x0F0F0F0F) | ((C[115] & 0x0F0F0F0F) << 4);
    B[114] = ((C[ 98] & 0xF0F0F0F0) >> 4) | (C[114] & 0xF0F0F0F0);
    B[115] = ((C[ 99] & 0xF0F0F0F0) >> 4) | (C[115] & 0xF0F0F0F0);
    B[102] = (C[102] & 0x0F0F0F0F) | ((C[118] & 0x0F0F0F0F) << 4);
    B[103] = (C[103] & 0x0F0F0F0F) | ((C[119] & 0x0F0F0F0F) << 4);
    B[118] = ((C[102] & 0xF0F0F0F0) >> 4) | (C[118] & 0xF0F0F0F0);
    B[119] = ((C[103] & 0xF0F0F0F0) >> 4) | (C[119] & 0xF0F0F0F0);
    B[106] = (C[106] & 0x0F0F0F0F) | ((C[122] & 0x0F0F0F0F) << 4);
    B[107] = (C[107] & 0x0F0F0F0F) | ((C[123] & 0x0F0F0F0F) << 4);
    B[122] = ((C[106] & 0xF0F0F0F0) >> 4) | (C[122] & 0xF0F0F0F0);
    B[123] = ((C[107] & 0xF0F0F0F0) >> 4) | (C[123] & 0xF0F0F0F0);
    B[110] = (C[110] & 0x0F0F0F0F) | ((C[126] & 0x0F0F0F0F) << 4);
    B[111] = (C[111] & 0x0F0F0F0F) | ((C[127] & 0x0F0F0F0F) << 4);
    B[126] = ((C[110] & 0xF0F0F0F0) >> 4) | (C[126] & 0xF0F0F0F0);
    B[127] = ((C[111] & 0xF0F0F0F0) >> 4) | (C[127] & 0xF0F0F0F0);
    C[  0] = (B[  0] & 0x00FF00FF) | ((B[ 32] & 0x00FF00FF) << 8);
    C[  1] = (B[  1] & 0x00FF00FF) | ((B[ 33] & 0x00FF00FF) << 8);
    C[ 32] = ((B[  0] & 0xFF00FF00) >> 8) | (B[ 32] & 0xFF00FF00);
    C[ 33] = ((B[  1] & 0xFF00FF00) >> 8) | (B[ 33] & 0xFF00FF00);
    C[  4] = (B[  4] & 0x00FF00FF) | ((B[ 36] & 0x00FF00FF) << 8);
    C[  5] = (B[  5] & 0x00FF00FF) | ((B[ 37] & 0x00FF00FF) << 8);
    C[ 36] = ((B[  4] & 0xFF00FF00) >> 8) | (B[ 36] & 0xFF00FF00);
    C[ 37] = ((B[  5] & 0xFF00FF00) >> 8) | (B[ 37] & 0xFF00FF00);
    C[  8] = (B[  8] & 0x00FF00FF) | ((B[ 40] & 0x00FF00FF) << 8);
    C[  9] = (B[  9] & 0x00FF00FF) | ((B[ 41] & 0x00FF00FF) << 8);
    C[ 40] = ((B[  8] & 0xFF00FF00) >> 8) | (B[ 40] & 0xFF00FF00);
    C[ 41] = ((B[  9] & 0xFF00FF00) >> 8) | (B[ 41] & 0xFF00FF00);
    C[ 12] = (B[ 12] & 0x00FF00FF) | ((B[ 44] & 0x00FF00FF) << 8);
    C[ 13] = (B[ 13] & 0x00FF00FF) | ((B[ 45] & 0x00FF00FF) << 8);
    C[ 44] = ((B[ 12] & 0xFF00FF00) >> 8) | (B[ 44] & 0xFF00FF00);
    C[ 45] = ((B[ 13] & 0xFF00FF00) >> 8) | (B[ 45] & 0xFF00FF00);
    C[ 16] = (B[ 16] & 0x00FF00FF) | ((B[ 48] & 0x00FF00FF) << 8);
    C[ 17] = (B[ 17] & 0x00FF00FF) | ((B[ 49] & 0x00FF00FF) << 8);
    C[ 48] = ((B[ 16] & 0xFF00FF00) >> 8) | (B[ 48] & 0xFF00FF00);
    C[ 49] = ((B[ 17] & 0xFF00FF00) >> 8) | (B[ 49] & 0xFF00FF00);
    C[ 20] = (B[ 20] & 0x00FF00FF) | ((B[ 52] & 0x00FF00FF) << 8);
    C[ 21] = (B[ 21] & 0x00FF00FF) | ((B[ 53] & 0x00FF00FF) << 8);
    C[ 52] = ((B[ 20] & 0xFF00FF00) >> 8) | (B[ 52] & 0xFF00FF00);
    C[ 53] = ((B[ 21] & 0xFF00FF00) >> 8) | (B[ 53] & 0xFF00FF00);
    C[ 24] = (B[ 24] & 0x00FF00FF) | ((B[ 56] & 0x00FF00FF) << 8);
    C[ 25] = (B[ 25] & 0x00FF00FF) | ((B[ 57] & 0x00FF00FF) << 8);
    C[ 56] = ((B[ 24] & 0xFF00FF00) >> 8) | (B[ 56] & 0xFF00FF00);
    C[ 57] = ((B[ 25] & 0xFF00FF00) >> 8) | (B[ 57] & 0xFF00FF00);
    C[ 28] = (B[ 28] & 0x00FF00FF) | ((B[ 60] & 0x00FF00FF) << 8);
    C[ 29] = (B[ 29] & 0x00FF00FF) | ((B[ 61] & 0x00FF00FF) << 8);
    C[ 60] = ((B[ 28] & 0xFF00FF00) >> 8) | (B[ 60] & 0xFF00FF00);
    C[ 61] = ((B[ 29] & 0xFF00FF00) >> 8) | (B[ 61] & 0xFF00FF00);
    C[ 64] = (B[ 64] & 0x00FF00FF) | ((B[ 96] & 0x00FF00FF) << 8);
    C[ 65] = (B[ 65] & 0x00FF00FF) | ((B[ 97] & 0x00FF00FF) << 8);
    C[ 96] = ((B[ 64] & 0xFF00FF00) >> 8) | (B[ 96] & 0xFF00FF00);
    C[ 97] = ((B[ 65] & 0xFF00FF00) >> 8) | (B[ 97] & 0xFF00FF00);
    C[ 68] = (B[ 68] & 0x00FF00FF) | ((B[100] & 0x00FF00FF) << 8);
    C[ 69] = (B[ 69] & 0x00FF00FF) | ((B[101] & 0x00FF00FF) << 8);
    C[100] = ((B[ 68] & 0xFF00FF00) >> 8) | (B[100] & 0xFF00FF00);
    C[101] = ((B[ 69] & 0xFF00FF00) >> 8) | (B[101] & 0xFF00FF00);
    C[ 72] = (B[ 72] & 0x00FF00FF) | ((B[104] & 0x00FF00FF) << 8);
    C[ 73] = (B[ 73] & 0x00FF00FF) | ((B[105] & 0x00FF00FF) << 8);
    C[104] = ((B[ 72] & 0xFF00FF00) >> 8) | (B[104] & 0xFF00FF00);
    C[105] = ((B[ 73] & 0xFF00FF00) >> 8) | (B[105] & 0xFF00FF00);
    C[ 76] = (B[ 76] & 0x00FF00FF) | ((B[108] & 0x00FF00FF) << 8);
    C[ 77] = (B[ 77] & 0x00FF00FF) | ((B[109] & 0x00FF00FF) << 8);
    C[108] = ((B[ 76] & 0xFF00FF00) >> 8) | (B[108] & 0xFF00FF00);
    C[109] = ((B[ 77] & 0xFF00FF00) >> 8) | (B[109] & 0xFF00FF00);
    C[ 80] = (B[ 80] & 0x00FF00FF) | ((B[112] & 0x00FF00FF) << 8);
    C[ 81] = (B[ 81] & 0x00FF00FF) | ((B[113] & 0x00FF00FF) << 8);
    C[112] = ((B[ 80] & 0xFF00FF00) >> 8) | (B[112] & 0xFF00FF00);
    C[113] = ((B[ 81] & 0xFF00FF00) >> 8) | (B[113] & 0xFF00FF00);
    C[ 84] = (B[ 84] & 0x00FF00FF) | ((B[116] & 0x00FF00FF) << 8);
    C[ 85] = (B[ 85] & 0x00FF00FF) | ((B[117] & 0x00FF00FF) << 8);
    C[116] = ((B[ 84] & 0xFF00FF00) >> 8) | (B[116] & 0xFF00FF00);
    C[117] = ((B[ 85] & 0xFF00FF00) >> 8) | (B[117] & 0xFF00FF00);
    C[ 88] = (B[ 88] & 0x00FF00FF) | ((B[120] & 0x00FF00FF) << 8);
    C[ 89] = (B[ 89] & 0x00FF00FF) | ((B[121] & 0x00FF00FF) << 8);
    C[120] = ((B[ 88] & 0xFF00FF00) >> 8) | (B[120] & 0xFF00FF00);
    C[121] = ((B[ 89] & 0xFF00FF00) >> 8) | (B[121] & 0xFF00FF00);
    C[ 92] = (B[ 92] & 0x00FF00FF) | ((B[124] & 0x00FF00FF) << 8);
    C[ 93] = (B[ 93] & 0x00FF00FF) | ((B[125] & 0x00FF00FF) << 8);
    C[124] = ((B[ 92] & 0xFF00FF00) >> 8) | (B[124] & 0xFF00FF00);
    C[125] = ((B[ 93] & 0xFF00FF00) >> 8) | (B[125] & 0xFF00FF00);
    C[  2] = (B[  2] & 0x00FF00FF) | ((B[ 34] & 0x00FF00FF) << 8);
    C[  3] = (B[  3] & 0x00FF00FF) | ((B[ 35] & 0x00FF00FF) << 8);
    C[ 34] = ((B[  2] & 0xFF00FF00) >> 8) | (B[ 34] & 0xFF00FF00);
    C[ 35] = ((B[  3] & 0xFF00FF00) >> 8) | (B[ 35] & 0xFF00FF00);
    C[  6] = (B[  6] & 0x00FF00FF) | ((B[ 38] & 0x00FF00FF) << 8);
    C[  7] = (B[  7] & 0x00FF00FF) | ((B[ 39] & 0x00FF00FF) << 8);
    C[ 38] = ((B[  6] & 0xFF00FF00) >> 8) | (B[ 38] & 0xFF00FF00);
    C[ 39] = ((B[  7] & 0xFF00FF00) >> 8) | (B[ 39] & 0xFF00FF00);
    C[ 10] = (B[ 10] & 0x00FF00FF) | ((B[ 42] & 0x00FF00FF) << 8);
    C[ 11] = (B[ 11] & 0x00FF00FF) | ((B[ 43] & 0x00FF00FF) << 8);
    C[ 42] = ((B[ 10] & 0xFF00FF00) >> 8) | (B[ 42] & 0xFF00FF00);
    C[ 43] = ((B[ 11] & 0xFF00FF00) >> 8) | (B[ 43] & 0xFF00FF00);
    C[ 14] = (B[ 14] & 0x00FF00FF) | ((B[ 46] & 0x00FF00FF) << 8);
    C[ 15] = (B[ 15] & 0x00FF00FF) | ((B[ 47] & 0x00FF00FF) << 8);
    C[ 46] = ((B[ 14] & 0xFF00FF00) >> 8) | (B[ 46] & 0xFF00FF00);
    C[ 47] = ((B[ 15] & 0xFF00FF00) >> 8) | (B[ 47] & 0xFF00FF00);
    C[ 18] = (B[ 18] & 0x00FF00FF) | ((B[ 50] & 0x00FF00FF) << 8);
    C[ 19] = (B[ 19] & 0x00FF00FF) | ((B[ 51] & 0x00FF00FF) << 8);
    C[ 50] = ((B[ 18] & 0xFF00FF00) >> 8) | (B[ 50] & 0xFF00FF00);
    C[ 51] = ((B[ 19] & 0xFF00FF00) >> 8) | (B[ 51] & 0xFF00FF00);
    C[ 22] = (B[ 22] & 0x00FF00FF) | ((B[ 54] & 0x00FF00FF) << 8);
    C[ 23] = (B[ 23] & 0x00FF00FF) | ((B[ 55] & 0x00FF00FF) << 8);
    C[ 54] = ((B[ 22] & 0xFF00FF00) >> 8) | (B[ 54] & 0xFF00FF00);
    C[ 55] = ((B[ 23] & 0xFF00FF00) >> 8) | (B[ 55] & 0xFF00FF00);
    C[ 26] = (B[ 26] & 0x00FF00FF) | ((B[ 58] & 0x00FF00FF) << 8);
    C[ 27] = (B[ 27] & 0x00FF00FF) | ((B[ 59] & 0x00FF00FF) << 8);
    C[ 58] = ((B[ 26] & 0xFF00FF00) >> 8) | (B[ 58] & 0xFF00FF00);
    C[ 59] = ((B[ 27] & 0xFF00FF00) >> 8) | (B[ 59] & 0xFF00FF00);
    C[ 30] = (B[ 30] & 0x00FF00FF) | ((B[ 62] & 0x00FF00FF) << 8);
    C[ 31] = (B[ 31] & 0x00FF00FF) | ((B[ 63] & 0x00FF00FF) << 8);
    C[ 62] = ((B[ 30] & 0xFF00FF00) >> 8) | (B[ 62] & 0xFF00FF00);
    C[ 63] = ((B[ 31] & 0xFF00FF00) >> 8) | (B[ 63] & 0xFF00FF00);
    C[ 66] = (B[ 66] & 0x00FF00FF) | ((B[ 98] & 0x00FF00FF) << 8);
    C[ 67] = (B[ 67] & 0x00FF00FF) | ((B[ 99] & 0x00FF00FF) << 8);
    C[ 98] = ((B[ 66] & 0xFF00FF00) >> 8) | (B[ 98] & 0xFF00FF00);
    C[ 99] = ((B[ 67] & 0xFF00FF00) >> 8) | (B[ 99] & 0xFF00FF00);
    C[ 70] = (B[ 70] & 0x00FF00FF) | ((B[102] & 0x00FF00FF) << 8);
    C[ 71] = (B[ 71] & 0x00FF00FF) | ((B[103] & 0x00FF00FF) << 8);
    C[102] = ((B[ 70] & 0xFF00FF00) >> 8) | (B[102] & 0xFF00FF00);
    C[103] = ((B[ 71] & 0xFF00FF00) >> 8) | (B[103] & 0xFF00FF00);
    C[ 74] = (B[ 74] & 0x00FF00FF) | ((B[106] & 0x00FF00FF) << 8);
    C[ 75] = (B[ 75] & 0x00FF00FF) | ((B[107] & 0x00FF00FF) << 8);
    C[106] = ((B[ 74] & 0xFF00FF00) >> 8) | (B[106] & 0xFF00FF00);
    C[107] = ((B[ 75] & 0xFF00FF00) >> 8) | (B[107] & 0xFF00FF00);
    C[ 78] = (B[ 78] & 0x00FF00FF) | ((B[110] & 0x00FF00FF) << 8);
    C[ 79] = (B[ 79] & 0x00FF00FF) | ((B[111] & 0x00FF00FF) << 8);
    C[110] = ((B[ 78] & 0xFF00FF00) >> 8) | (B[110] & 0xFF00FF00);
    C[111] = ((B[ 79] & 0xFF00FF00) >> 8) | (B[111] & 0xFF00FF00);
    C[ 82] = (B[ 82] & 0x00FF00FF) | ((B[114] & 0x00FF00FF) << 8);
    C[ 83] = (B[ 83] & 0x00FF00FF) | ((B[115] & 0x00FF00FF) << 8);
    C[114] = ((B[ 82] & 0xFF00FF00) >> 8) | (B[114] & 0xFF00FF00);
    C[115] = ((B[ 83] & 0xFF00FF00) >> 8) | (B[115] & 0xFF00FF00);
    C[ 86] = (B[ 86] & 0x00FF00FF) | ((B[118] & 0x00FF00FF) << 8);
    C[ 87] = (B[ 87] & 0x00FF00FF) | ((B[119] & 0x00FF00FF) << 8);
    C[118] = ((B[ 86] & 0xFF00FF00) >> 8) | (B[118] & 0xFF00FF00);
    C[119] = ((B[ 87] & 0xFF00FF00) >> 8) | (B[119] & 0xFF00FF00);
    C[ 90] = (B[ 90] & 0x00FF00FF) | ((B[122] & 0x00FF00FF) << 8);
    C[ 91] = (B[ 91] & 0x00FF00FF) | ((B[123] & 0x00FF00FF) << 8);
    C[122] = ((B[ 90] & 0xFF00FF00) >> 8) | (B[122] & 0xFF00FF00);
    C[123] = ((B[ 91] & 0xFF00FF00) >> 8) | (B[123] & 0xFF00FF00);
    C[ 94] = (B[ 94] & 0x00FF00FF) | ((B[126] & 0x00FF00FF) << 8);
    C[ 95] = (B[ 95] & 0x00FF00FF) | ((B[127] & 0x00FF00FF) << 8);
    C[126] = ((B[ 94] & 0xFF00FF00) >> 8) | (B[126] & 0xFF00FF00);
    C[127] = ((B[ 95] & 0xFF00FF00) >> 8) | (B[127] & 0xFF00FF00);
    B[  0] = (C[  0] & 0x0000FFFF) | ((C[ 64] & 0x0000FFFF) << 16);
    B[  1] = (C[  1] & 0x0000FFFF) | ((C[ 65] & 0x0000FFFF) << 16);
    B[ 64] = ((C[  0] & 0xFFFF0000) >> 16) | (C[ 64] & 0xFFFF0000);
    B[ 65] = ((C[  1] & 0xFFFF0000) >> 16) | (C[ 65] & 0xFFFF0000);
    B[  4] = (C[  4] & 0x0000FFFF) | ((C[ 68] & 0x0000FFFF) << 16);
    B[  5] = (C[  5] & 0x0000FFFF) | ((C[ 69] & 0x0000FFFF) << 16);
    B[ 68] = ((C[  4] & 0xFFFF0000) >> 16) | (C[ 68] & 0xFFFF0000);
    B[ 69] = ((C[  5] & 0xFFFF0000) >> 16) | (C[ 69] & 0xFFFF0000);
    B[  8] = (C[  8] & 0x0000FFFF) | ((C[ 72] & 0x0000FFFF) << 16);
    B[  9] = (C[  9] & 0x0000FFFF) | ((C[ 73] & 0x0000FFFF) << 16);
    B[ 72] = ((C[  8] & 0xFFFF0000) >> 16) | (C[ 72] & 0xFFFF0000);
    B[ 73] = ((C[  9] & 0xFFFF0000) >> 16) | (C[ 73] & 0xFFFF0000);
    B[ 12] = (C[ 12] & 0x0000FFFF) | ((C[ 76] & 0x0000FFFF) << 16);
    B[ 13] = (C[ 13] & 0x0000FFFF) | ((C[ 77] & 0x0000FFFF) << 16);
    B[ 76] = ((C[ 12] & 0xFFFF0000) >> 16) | (C[ 76] & 0xFFFF0000);
    B[ 77] = ((C[ 13] & 0xFFFF0000) >> 16) | (C[ 77] & 0xFFFF0000);
    B[ 16] = (C[ 16] & 0x0000FFFF) | ((C[ 80] & 0x0000FFFF) << 16);
    B[ 17] = (C[ 17] & 0x0000FFFF) | ((C[ 81] & 0x0000FFFF) << 16);
    B[ 80] = ((C[ 16] & 0xFFFF0000) >> 16) | (C[ 80] & 0xFFFF0000);
    B[ 81] = ((C[ 17] & 0xFFFF0000) >> 16) | (C[ 81] & 0xFFFF0000);
    B[ 20] = (C[ 20] & 0x0000FFFF) | ((C[ 84] & 0x0000FFFF) << 16);
    B[ 21] = (C[ 21] & 0x0000FFFF) | ((C[ 85] & 0x0000FFFF) << 16);
    B[ 84] = ((C[ 20] & 0xFFFF0000) >> 16) | (C[ 84] & 0xFFFF0000);
    B[ 85] = ((C[ 21] & 0xFFFF0000) >> 16) | (C[ 85] & 0xFFFF0000);
    B[ 24] = (C[ 24] & 0x0000FFFF) | ((C[ 88] & 0x0000FFFF) << 16);
    B[ 25] = (C[ 25] & 0x0000FFFF) | ((C[ 89] & 0x0000FFFF) << 16);
    B[ 88] = ((C[ 24] & 0xFFFF0000) >> 16) | (C[ 88] & 0xFFFF0000);
    B[ 89] = ((C[ 25] & 0xFFFF0000) >> 16) | (C[ 89] & 0xFFFF0000);
    B[ 28] = (C[ 28] & 0x0000FFFF) | ((C[ 92] & 0x0000FFFF) << 16);
    B[ 29] = (C[ 29] & 0x0000FFFF) | ((C[ 93] & 0x0000FFFF) << 16);
    B[ 92] = ((C[ 28] & 0xFFFF0000) >> 16) | (C[ 92] & 0xFFFF0000);
    B[ 93] = ((C[ 29] & 0xFFFF0000) >> 16) | (C[ 93] & 0xFFFF0000);
    B[ 32] = (C[ 32] & 0x0000FFFF) | ((C[ 96] & 0x0000FFFF) << 16);
    B[ 33] = (C[ 33] & 0x0000FFFF) | ((C[ 97] & 0x0000FFFF) << 16);
    B[ 96] = ((C[ 32] & 0xFFFF0000) >> 16) | (C[ 96] & 0xFFFF0000);
    B[ 97] = ((C[ 33] & 0xFFFF0000) >> 16) | (C[ 97] & 0xFFFF0000);
    B[ 36] = (C[ 36] & 0x0000FFFF) | ((C[100] & 0x0000FFFF) << 16);
    B[ 37] = (C[ 37] & 0x0000FFFF) | ((C[101] & 0x0000FFFF) << 16);
    B[100] = ((C[ 36] & 0xFFFF0000) >> 16) | (C[100] & 0xFFFF0000);
    B[101] = ((C[ 37] & 0xFFFF0000) >> 16) | (C[101] & 0xFFFF0000);
    B[ 40] = (C[ 40] & 0x0000FFFF) | ((C[104] & 0x0000FFFF) << 16);
    B[ 41] = (C[ 41] & 0x0000FFFF) | ((C[105] & 0x0000FFFF) << 16);
    B[104] = ((C[ 40] & 0xFFFF0000) >> 16) | (C[104] & 0xFFFF0000);
    B[105] = ((C[ 41] & 0xFFFF0000) >> 16) | (C[105] & 0xFFFF0000);
    B[ 44] = (C[ 44] & 0x0000FFFF) | ((C[108] & 0x0000FFFF) << 16);
    B[ 45] = (C[ 45] & 0x0000FFFF) | ((C[109] & 0x0000FFFF) << 16);
    B[108] = ((C[ 44] & 0xFFFF0000) >> 16) | (C[108] & 0xFFFF0000);
    B[109] = ((C[ 45] & 0xFFFF0000) >> 16) | (C[109] & 0xFFFF0000);
    B[ 48] = (C[ 48] & 0x0000FFFF) | ((C[112] & 0x0000FFFF) << 16);
    B[ 49] = (C[ 49] & 0x0000FFFF) | ((C[113] & 0x0000FFFF) << 16);
    B[112] = ((C[ 48] & 0xFFFF0000) >> 16) | (C[112] & 0xFFFF0000);
    B[113] = ((C[ 49] & 0xFFFF0000) >> 16) | (C[113] & 0xFFFF0000);
    B[ 52] = (C[ 52] & 0x0000FFFF) | ((C[116] & 0x0000FFFF) << 16);
    B[ 53] = (C[ 53] & 0x0000FFFF) | ((C[117] & 0x0000FFFF) << 16);
    B[116] = ((C[ 52] & 0xFFFF0000) >> 16) | (C[116] & 0xFFFF0000);
    B[117] = ((C[ 53] & 0xFFFF0000) >> 16) | (C[117] & 0xFFFF0000);
    B[ 56] = (C[ 56] & 0x0000FFFF) | ((C[120] & 0x0000FFFF) << 16);
    B[ 57] = (C[ 57] & 0x0000FFFF) | ((C[121] & 0x0000FFFF) << 16);
    B[120] = ((C[ 56] & 0xFFFF0000) >> 16) | (C[120] & 0xFFFF0000);
    B[121] = ((C[ 57] & 0xFFFF0000) >> 16) | (C[121] & 0xFFFF0000);
    B[ 60] = (C[ 60] & 0x0000FFFF) | ((C[124] & 0x0000FFFF) << 16);
    B[ 61] = (C[ 61] & 0x0000FFFF) | ((C[125] & 0x0000FFFF) << 16);
    B[124] = ((C[ 60] & 0xFFFF0000) >> 16) | (C[124] & 0xFFFF0000);
    B[125] = ((C[ 61] & 0xFFFF0000) >> 16) | (C[125] & 0xFFFF0000);
    B[  2] = (C[  2] & 0x0000FFFF) | ((C[ 66] & 0x0000FFFF) << 16);
    B[  3] = (C[  3] & 0x0000FFFF) | ((C[ 67] & 0x0000FFFF) << 16);
    B[ 66] = ((C[  2] & 0xFFFF0000) >> 16) | (C[ 66] & 0xFFFF0000);
    B[ 67] = ((C[  3] & 0xFFFF0000) >> 16) | (C[ 67] & 0xFFFF0000);
    B[  6] = (C[  6] & 0x0000FFFF) | ((C[ 70] & 0x0000FFFF) << 16);
    B[  7] = (C[  7] & 0x0000FFFF) | ((C[ 71] & 0x0000FFFF) << 16);
    B[ 70] = ((C[  6] & 0xFFFF0000) >> 16) | (C[ 70] & 0xFFFF0000);
    B[ 71] = ((C[  7] & 0xFFFF0000) >> 16) | (C[ 71] & 0xFFFF0000);
    B[ 10] = (C[ 10] & 0x0000FFFF) | ((C[ 74] & 0x0000FFFF) << 16);
    B[ 11] = (C[ 11] & 0x0000FFFF) | ((C[ 75] & 0x0000FFFF) << 16);
    B[ 74] = ((C[ 10] & 0xFFFF0000) >> 16) | (C[ 74] & 0xFFFF0000);
    B[ 75] = ((C[ 11] & 0xFFFF0000) >> 16) | (C[ 75] & 0xFFFF0000);
    B[ 14] = (C[ 14] & 0x0000FFFF) | ((C[ 78] & 0x0000FFFF) << 16);
    B[ 15] = (C[ 15] & 0x0000FFFF) | ((C[ 79] & 0x0000FFFF) << 16);
    B[ 78] = ((C[ 14] & 0xFFFF0000) >> 16) | (C[ 78] & 0xFFFF0000);
    B[ 79] = ((C[ 15] & 0xFFFF0000) >> 16) | (C[ 79] & 0xFFFF0000);
    B[ 18] = (C[ 18] & 0x0000FFFF) | ((C[ 82] & 0x0000FFFF) << 16);
    B[ 19] = (C[ 19] & 0x0000FFFF) | ((C[ 83] & 0x0000FFFF) << 16);
    B[ 82] = ((C[ 18] & 0xFFFF0000) >> 16) | (C[ 82] & 0xFFFF0000);
    B[ 83] = ((C[ 19] & 0xFFFF0000) >> 16) | (C[ 83] & 0xFFFF0000);
    B[ 22] = (C[ 22] & 0x0000FFFF) | ((C[ 86] & 0x0000FFFF) << 16);
    B[ 23] = (C[ 23] & 0x0000FFFF) | ((C[ 87] & 0x0000FFFF) << 16);
    B[ 86] = ((C[ 22] & 0xFFFF0000) >> 16) | (C[ 86] & 0xFFFF0000);
    B[ 87] = ((C[ 23] & 0xFFFF0000) >> 16) | (C[ 87] & 0xFFFF0000);
    B[ 26] = (C[ 26] & 0x0000FFFF) | ((C[ 90] & 0x0000FFFF) << 16);
    B[ 27] = (C[ 27] & 0x0000FFFF) | ((C[ 91] & 0x0000FFFF) << 16);
    B[ 90] = ((C[ 26] & 0xFFFF0000) >> 16) | (C[ 90] & 0xFFFF0000);
    B[ 91] = ((C[ 27] & 0xFFFF0000) >> 16) | (C[ 91] & 0xFFFF0000);
    B[ 30] = (C[ 30] & 0x0000FFFF) | ((C[ 94] & 0x0000FFFF) << 16);
    B[ 31] = (C[ 31] & 0x0000FFFF) | ((C[ 95] & 0x0000FFFF) << 16);
    B[ 94] = ((C[ 30] & 0xFFFF0000) >> 16) | (C[ 94] & 0xFFFF0000);
    B[ 95] = ((C[ 31] & 0xFFFF0000) >> 16) | (C[ 95] & 0xFFFF0000);
    B[ 34] = (C[ 34] & 0x0000FFFF) | ((C[ 98] & 0x0000FFFF) << 16);
    B[ 35] = (C[ 35] & 0x0000FFFF) | ((C[ 99] & 0x0000FFFF) << 16);
    B[ 98] = ((C[ 34] & 0xFFFF0000) >> 16) | (C[ 98] & 0xFFFF0000);
    B[ 99] = ((C[ 35] & 0xFFFF0000) >> 16) | (C[ 99] & 0xFFFF0000);
    B[ 38] = (C[ 38] & 0x0000FFFF) | ((C[102] & 0x0000FFFF) << 16);
    B[ 39] = (C[ 39] & 0x0000FFFF) | ((C[103] & 0x0000FFFF) << 16);
    B[102] = ((C[ 38] & 0xFFFF0000) >> 16) | (C[102] & 0xFFFF0000);
    B[103] = ((C[ 39] & 0xFFFF0000) >> 16) | (C[103] & 0xFFFF0000);
    B[ 42] = (C[ 42] & 0x0000FFFF) | ((C[106] & 0x0000FFFF) << 16);
    B[ 43] = (C[ 43] & 0x0000FFFF) | ((C[107] & 0x0000FFFF) << 16);
    B[106] = ((C[ 42] & 0xFFFF0000) >> 16) | (C[106] & 0xFFFF0000);
    B[107] = ((C[ 43] & 0xFFFF0000) >> 16) | (C[107] & 0xFFFF0000);
    B[ 46] = (C[ 46] & 0x0000FFFF) | ((C[110] & 0x0000FFFF) << 16);
    B[ 47] = (C[ 47] & 0x0000FFFF) | ((C[111] & 0x0000FFFF) << 16);
    B[110] = ((C[ 46] & 0xFFFF0000) >> 16) | (C[110] & 0xFFFF0000);
    B[111] = ((C[ 47] & 0xFFFF0000) >> 16) | (C[111] & 0xFFFF0000);
    B[ 50] = (C[ 50] & 0x0000FFFF) | ((C[114] & 0x0000FFFF) << 16);
    B[ 51] = (C[ 51] & 0x0000FFFF) | ((C[115] & 0x0000FFFF) << 16);
    B[114] = ((C[ 50] & 0xFFFF0000) >> 16) | (C[114] & 0xFFFF0000);
    B[115] = ((C[ 51] & 0xFFFF0000) >> 16) | (C[115] & 0xFFFF0000);
    B[ 54] = (C[ 54] & 0x0000FFFF) | ((C[118] & 0x0000FFFF) << 16);
    B[ 55] = (C[ 55] & 0x0000FFFF) | ((C[119] & 0x0000FFFF) << 16);
    B[118] = ((C[ 54] & 0xFFFF0000) >> 16) | (C[118] & 0xFFFF0000);
    B[119] = ((C[ 55] & 0xFFFF0000) >> 16) | (C[119] & 0xFFFF0000);
    B[ 58] = (C[ 58] & 0x0000FFFF) | ((C[122] & 0x0000FFFF) << 16);
    B[ 59] = (C[ 59] & 0x0000FFFF) | ((C[123] & 0x0000FFFF) << 16);
    B[122] = ((C[ 58] & 0xFFFF0000) >> 16) | (C[122] & 0xFFFF0000);
    B[123] = ((C[ 59] & 0xFFFF0000) >> 16) | (C[123] & 0xFFFF0000);
    B[ 62] = (C[ 62] & 0x0000FFFF) | ((C[126] & 0x0000FFFF) << 16);
    B[ 63] = (C[ 63] & 0x0000FFFF) | ((C[127] & 0x0000FFFF) << 16);
    B[126] = ((C[ 62] & 0xFFFF0000) >> 16) | (C[126] & 0xFFFF0000);
    B[127] = ((C[ 63] & 0xFFFF0000) >> 16) | (C[127] & 0xFFFF0000);
    A[  0] = B[  0];
    A[  1] = B[  2];
    A[  2] = B[  1];
    A[  3] = B[  3];
    A[  4] = B[  4];
    A[  5] = B[  6];
    A[  6] = B[  5];
    A[  7] = B[  7];
    A[  8] = B[  8];
    A[  9] = B[ 10];
    A[ 10] = B[  9];
    A[ 11] = B[ 11];
    A[ 12] = B[ 12];
    A[ 13] = B[ 14];
    A[ 14] = B[ 13];
    A[ 15] = B[ 15];
    A[ 16] = B[ 16];
    A[ 17] = B[ 18];
    A[ 18] = B[ 17];
    A[ 19] = B[ 19];
    A[ 20] = B[ 20];
    A[ 21] = B[ 22];
    A[ 22] = B[ 21];
    A[ 23] = B[ 23];
    A[ 24] = B[ 24];
    A[ 25] = B[ 26];
    A[ 26] = B[ 25];
    A[ 27] = B[ 27];
    A[ 28] = B[ 28];
    A[ 29] = B[ 30];
    A[ 30] = B[ 29];
    A[ 31] = B[ 31];
    A[ 32] = B[ 32];
    A[ 33] = B[ 34];
    A[ 34] = B[ 33];
    A[ 35] = B[ 35];
    A[ 36] = B[ 36];
    A[ 37] = B[ 38];
    A[ 38] = B[ 37];
    A[ 39] = B[ 39];
    A[ 40] = B[ 40];
    A[ 41] = B[ 42];
    A[ 42] = B[ 41];
    A[ 43] = B[ 43];
    A[ 44] = B[ 44];
    A[ 45] = B[ 46];
    A[ 46] = B[ 45];
    A[ 47] = B[ 47];
    A[ 48] = B[ 48];
    A[ 49] = B[ 50];
    A[ 50] = B[ 49];
    A[ 51] = B[ 51];
    A[ 52] = B[ 52];
    A[ 53] = B[ 54];
    A[ 54] = B[ 53];
    A[ 55] = B[ 55];
    A[ 56] = B[ 56];
    A[ 57] = B[ 58];
    A[ 58] = B[ 57];
    A[ 59] = B[ 59];
    A[ 60] = B[ 60];
    A[ 61] = B[ 62];
    A[ 62] = B[ 61];
    A[ 63] = B[ 63];
    A[ 64] = B[ 64];
    A[ 65] = B[ 66];
    A[ 66] = B[ 65];
    A[ 67] = B[ 67];
    A[ 68] = B[ 68];
    A[ 69] = B[ 70];
    A[ 70] = B[ 69];
    A[ 71] = B[ 71];
    A[ 72] = B[ 72];
    A[ 73] = B[ 74];
    A[ 74] = B[ 73];
    A[ 75] = B[ 75];
    A[ 76] = B[ 76];
    A[ 77] = B[ 78];
    A[ 78] = B[ 77];
    A[ 79] = B[ 79];
    A[ 80] = B[ 80];
    A[ 81] = B[ 82];
    A[ 82] = B[ 81];
    A[ 83] = B[ 83];
    A[ 84] = B[ 84];
    A[ 85] = B[ 86];
    A[ 86] = B[ 85];
    A[ 87] = B[ 87];
    A[ 88] = B[ 88];
    A[ 89] = B[ 90];
    A[ 90] = B[ 89];
    A[ 91] = B[ 91];
    A[ 92] = B[ 92];
    A[ 93] = B[ 94];
    A[ 94] = B[ 93];
    A[ 95] = B[ 95];
    A[ 96] = B[ 96];
    A[ 97] = B[ 98];
    A[ 98] = B[ 97];
    A[ 99] = B[ 99];
    A[100] = B[100];
    A[101] = B[102];
    A[102] = B[101];
    A[103] = B[103];
    A[104] = B[104];
    A[105] = B[106];
    A[106] = B[105];
    A[107] = B[107];
    A[108] = B[108];
    A[109] = B[110];
    A[110] = B[109];
    A[111] = B[111];
    A[112] = B[112];
    A[113] = B[114];
    A[114] = B[113];
    A[115] = B[115];
    A[116] = B[116];
    A[117] = B[118];
    A[118] = B[117];
    A[119] = B[119];
    A[120] = B[120];
    A[121] = B[122];
    A[122] = B[121];
    A[123] = B[123];
    A[124] = B[124];
    A[125] = B[126];
    A[126] = B[125];
    A[127] = B[127];
}



static void transposeInplaceArranged(uint4* A, int64_t n) {
  assert(n % 64 == 0);
  int64_t d = intlog2(n) - 6;
#pragma omp parallel for
  for (int64_t i = 0; i < (1L << 2*d); ++i)
    transposeInplaceArranged64(A + 32*i);
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
// x[0][0] <-- x[0][0]
// x[0][1] <-- x[0][1]
// x[1][0] <-- x[1][0]
// x[1][1] <-- x[0][1] + x[1][0] + x[1][1]
static void changeOfBasisSelfInverseInPlace(uint4* x, int64_t n) {
  assert(n >= 64);
  assert(isPowerOfTwo(n));
  int64_t d = intlog2(n) - 6;
  assert(n == (1L<<d)*64);

  for (int64_t l = d-1; l >= 0; --l) {
    int64_t logM = (2*d-2*l+3);
    int64_t M = 1L << logM;
    for (int64_t t = 0; t < (1L << 2*d+3); ++t) {
      int64_t u = t >> logM;
      int64_t v = t & ~(0xffffffffffffffff << logM);
      uint4 x01 = x[u*4*M + M   + v];
      uint4 x10 = x[u*4*M + 2*M + v];
      uint4 x11 = x[u*4*M + 3*M + v];
      x[u*4*M + 3*M + v] = x11 ^ x01 ^ x10;
    }
  }
}



// compute the following change of basis:
// xx[0][0] <--  x[0][0]
// xx[0][1] <--  x[0][1]
// xx[1][1] <--  x[0][1] + x[1][1]
// xx[1][0] <-- xx[1][1] + x[1][0]
static void changeOfBasisChainingLeftInPlace(uint4* x, int64_t n) {
  assert(n >= 64);
  assert(isPowerOfTwo(n));
  int64_t d = intlog2(n) - 6;
  assert(n == (1L<<d)*64);

  for (int64_t l = d-1; l >= 0; --l) {
    int64_t logM = (2*d-2*l+3);
    int64_t M = 1L << logM;
    for (int64_t t = 0; t < (1L << 2*d+3); ++t) {
      int64_t u = t >> logM;
      int64_t v = t & ~(0xffffffffffffffff << logM);
      uint4 x01  = x[u*4*M + M   + v];
      uint4 x10  = x[u*4*M + 2*M + v];
      uint4 x11  = x[u*4*M + 3*M + v];
      uint4 xx11 =  x01 ^ x11;
      uint4 xx10 = xx11 ^ x10;
      x[u*4*M + 2*M + v] = xx10;
      x[u*4*M + 3*M + v] = xx11;
    }
  }
}



// compute the following change of basis:
// xx[0][0] <--  x[0][0]
// xx[0][1] <--  x[0][1] + x[1][0] + x[1][1]
// xx[1][0] <--  x[1][0]
// xx[1][1] <--  x[1][0] + x[1][1]
static void changeOfBasisChainingRightInPlace(uint4* x, int64_t n) {
  assert(n >= 64);
  assert(isPowerOfTwo(n));
  int64_t d = intlog2(n) - 6;
  assert(n == (1L<<d)*64);

  for (int64_t l = d-1; l >= 0; --l) {
    int64_t logM = (2*d-2*l+3);
    int64_t M = 1L << logM;
    for (int64_t t = 0; t < (1L << 2*d+3); ++t) {
      int64_t u = t >> logM;
      int64_t v = t & ~(0xffffffffffffffff << logM);
      uint4 x01  = x[u*4*M + M   + v];
      uint4 x10  = x[u*4*M + 2*M + v];
      uint4 x11  = x[u*4*M + 3*M + v];
      uint4 xx01 = x01 ^ x10 ^ x11;
      uint4 xx11 = x10 ^ x11;
      x[u*4*M + M   + v] = xx01;
      x[u*4*M + 3*M + v] = xx11;
    }
  }
}



static void changeOfBasisSelfInverseInPlaceAvxParallel(uint4* x, int64_t n) {
  assert(n >= 64);
  assert(isPowerOfTwo(n));
  int64_t d = intlog2(n) - 6;
  assert(n == (1L<<d)*64);

  for (int64_t l = d-1; l >= 0; --l) {
    int64_t logM = (2*d-2*l+3);
    int64_t M = 1L << logM;
#pragma omp parallel for
    for (int64_t t = 0; t < (1L << 2*d+3); t += 4) {
      int64_t u = t >> logM;
      int64_t v = t & ~(0xffffffffffffffff << logM);
      __v4du x010 = *reinterpret_cast<const __v4du*>(x+u*4*M+M+v);
      __v4du x011 = *reinterpret_cast<const __v4du*>(x+u*4*M+M+v+2);
      __v4du x100 = *reinterpret_cast<const __v4du*>(x+u*4*M+2*M+v);
      __v4du x101 = *reinterpret_cast<const __v4du*>(x+u*4*M+2*M+v+2);
      __v4du x110 = *reinterpret_cast<const __v4du*>(x+u*4*M+3*M+v);
      __v4du x111 = *reinterpret_cast<const __v4du*>(x+u*4*M+3*M+v+2);
      *reinterpret_cast<__v4du*>(x + u*4*M + 3*M + v)   = x010 ^ x100 ^ x110;
      *reinterpret_cast<__v4du*>(x + u*4*M + 3*M + v+2) = x011 ^ x101 ^ x111;
    }
  }
}



static void changeOfBasisChainingLeftInPlaceAvxParallel(uint4* x, int64_t n) {
  assert(n >= 64);
  assert(isPowerOfTwo(n));
  int64_t d = intlog2(n) - 6;
  assert(n == (1L<<d)*64);

  for (int64_t l = d-1; l >= 0; --l) {
    int64_t logM = (2*d-2*l+3);
    int64_t M = 1L << logM;
#pragma omp parallel for
    for (int64_t t = 0; t < (1L << 2*d+3); t += 4) {
      int64_t u = t >> logM;
      int64_t v = t & ~(0xffffffffffffffff << logM);
      __v4du x010 = *reinterpret_cast<const __v4du*>(x+u*4*M+M+v);
      __v4du x011 = *reinterpret_cast<const __v4du*>(x+u*4*M+M+v+2);
      __v4du x100 = *reinterpret_cast<const __v4du*>(x+u*4*M+2*M+v);
      __v4du x101 = *reinterpret_cast<const __v4du*>(x+u*4*M+2*M+v+2);
      __v4du x110 = *reinterpret_cast<const __v4du*>(x+u*4*M+3*M+v);
      __v4du x111 = *reinterpret_cast<const __v4du*>(x+u*4*M+3*M+v+2);

      __v4du xx110 = x010 ^ x110;
      __v4du xx111 = x011 ^ x111;

      __v4du xx100 = xx110 ^ x100;
      __v4du xx101 = xx111 ^ x101;

      *reinterpret_cast<__v4du*>(x + u*4*M + 2*M + v)   = xx100;
      *reinterpret_cast<__v4du*>(x + u*4*M + 2*M + v+2) = xx101;
      *reinterpret_cast<__v4du*>(x + u*4*M + 3*M + v)   = xx110;
      *reinterpret_cast<__v4du*>(x + u*4*M + 3*M + v+2) = xx111;
    }
  }
}



static void changeOfBasisChainingRightInPlaceAvxParallel(uint4* x, int64_t n) {
  assert(n >= 64);
  assert(isPowerOfTwo(n));
  int64_t d = intlog2(n) - 6;
  assert(n == (1L<<d)*64);

  for (int64_t l = d-1; l >= 0; --l) {
    int64_t logM = (2*d-2*l+3);
    int64_t M = 1L << logM;
#pragma omp parallel for
    for (int64_t t = 0; t < (1L << 2*d+3); t += 4) {
      int64_t u = t >> logM;
      int64_t v = t & ~(0xffffffffffffffff << logM);
      // __v4du x000 = *reinterpret_cast<const __v4du*>(x+u*4*M+v);
      // __v4du x001 = *reinterpret_cast<const __v4du*>(x+u*4*M+v+2);
      __v4du x010 = *reinterpret_cast<const __v4du*>(x+u*4*M+M+v);
      __v4du x011 = *reinterpret_cast<const __v4du*>(x+u*4*M+M+v+2);
      __v4du x100 = *reinterpret_cast<const __v4du*>(x+u*4*M+2*M+v);
      __v4du x101 = *reinterpret_cast<const __v4du*>(x+u*4*M+2*M+v+2);
      __v4du x110 = *reinterpret_cast<const __v4du*>(x+u*4*M+3*M+v);
      __v4du x111 = *reinterpret_cast<const __v4du*>(x+u*4*M+3*M+v+2);

      __v4du xx110 = x100 ^ x110;
      __v4du xx111 = x101 ^ x111;

      __v4du xx010 = xx110 ^ x010;
      __v4du xx011 = xx111 ^ x011;

      *reinterpret_cast<__v4du*>(x + u*4*M + M + v)     = xx010;
      *reinterpret_cast<__v4du*>(x + u*4*M + M + v+2)   = xx011;
      *reinterpret_cast<__v4du*>(x + u*4*M + 3*M + v)   = xx110;
      *reinterpret_cast<__v4du*>(x + u*4*M + 3*M + v+2) = xx111;
    }
  }
}



// compute the following change of basis:
// z[0][0] <-- z[0][0]
// z[0][1] <-- z[0][1] + z[1][1]
// z[1][0] <-- z[1][0] + z[1][1]
// z[1][1] <-- z[1][1]
static void inverseChangeOfBasisSelfInverseInPlace(uint4* z, int64_t n) {
  assert(n >= 64);
  assert(isPowerOfTwo(n));
  int64_t d = intlog2(n) - 6;
  assert(n == (1L<<d)*64);

  for (int64_t l = d-1; l >= 0; --l) {
    int64_t logM = (2*d-2*l+3);
    int64_t M = 1L << logM;
    for (int64_t t = 0; t < (1L << 2*d+3); ++t) {
      int64_t u = t >> logM;
      int64_t v = t & ~(0xffffffffffffffff << logM);
      uint4 z01 = z[u*4*M + M   + v];
      uint4 z10 = z[u*4*M + 2*M + v];
      uint4 z11 = z[u*4*M + 3*M + v];
      z[u*4*M + M   + v] = z01 ^ z11;
      z[u*4*M + 2*M + v] = z10 ^ z11;
    }
  }
}



// compute the following change of basis:
// z[0][0] <-- z[0][0]
// z[0][1] <-- z[0][1]
// z[1][0] <-- z[1][0] + z[1][1]
// z[1][1] <-- z[0][1] + z[1][1]
static void inverseChangeOfBasisChainingInPlace(uint4* z, int64_t n) {
  assert(n >= 64);
  assert(isPowerOfTwo(n));
  int64_t d = intlog2(n) - 6;
  assert(n == (1L<<d)*64);

  for (int64_t l = d-1; l >= 0; --l) {
    int64_t logM = (2*d-2*l+3);
    int64_t M = 1L << logM;
    for (int64_t t = 0; t < (1L << 2*d+3); ++t) {
      int64_t u = t >> logM;
      int64_t v = t & ~(0xffffffffffffffff << logM);
      uint4 z01 = z[u*4*M + M   + v];
      uint4 z10 = z[u*4*M + 2*M + v];
      uint4 z11 = z[u*4*M + 3*M + v];
      z[u*4*M + 2*M + v] = z10 ^ z11;
      z[u*4*M + 3*M + v] = z01 ^ z11;
    }
  }
}



static void inverseChangeOfBasisSelfInverseInPlaceAvxParallel(uint4* x, int64_t n) {
  assert(n >= 64);
  assert(isPowerOfTwo(n));
  int64_t d = intlog2(n) - 6;
  assert(n == (1L<<d)*64);

  for (int64_t l = d-1; l >= 0; --l) {
    int64_t logM = (2*d-2*l+3);
    int64_t M = 1L << logM;
#pragma omp parallel for
    for (int64_t t = 0; t < (1L << 2*d+3); t += 4) {
      int64_t u = t >> logM;
      int64_t v = t & ~(0xffffffffffffffff << logM);
      __v4du z010 = *reinterpret_cast<const __v4du*>(x+u*4*M+M+v);
      __v4du z011 = *reinterpret_cast<const __v4du*>(x+u*4*M+M+v+2);
      __v4du z100 = *reinterpret_cast<const __v4du*>(x+u*4*M+2*M+v);
      __v4du z101 = *reinterpret_cast<const __v4du*>(x+u*4*M+2*M+v+2);
      __v4du z110 = *reinterpret_cast<const __v4du*>(x+u*4*M+3*M+v);
      __v4du z111 = *reinterpret_cast<const __v4du*>(x+u*4*M+3*M+v+2);
      *reinterpret_cast<__v4du*>(x + u*4*M + M + v) = z010 ^ z110;
      *reinterpret_cast<__v4du*>(x + u*4*M + M + v+2) = z011 ^ z111;
      *reinterpret_cast<__v4du*>(x + u*4*M + 2*M + v) = z100 ^ z110;
      *reinterpret_cast<__v4du*>(x + u*4*M + 2*M + v+2) = z101 ^ z111;
    }
  }
}



static void inverseChangeOfBasisChainingInPlaceAvxParallel(uint4* x, int64_t n) {
  assert(n >= 64);
  assert(isPowerOfTwo(n));
  int64_t d = intlog2(n) - 6;
  assert(n == (1L<<d)*64);

  for (int64_t l = d-1; l >= 0; --l) {
    int64_t logM = (2*d-2*l+3);
    int64_t M = 1L << logM;
#pragma omp parallel for
    for (int64_t t = 0; t < (1L << 2*d+3); t += 4) {
      int64_t u = t >> logM;
      int64_t v = t & ~(0xffffffffffffffff << logM);
      __v4du z010 = *reinterpret_cast<const __v4du*>(x+u*4*M+M+v);
      __v4du z011 = *reinterpret_cast<const __v4du*>(x+u*4*M+M+v+2);
      __v4du z100 = *reinterpret_cast<const __v4du*>(x+u*4*M+2*M+v);
      __v4du z101 = *reinterpret_cast<const __v4du*>(x+u*4*M+2*M+v+2);
      __v4du z110 = *reinterpret_cast<const __v4du*>(x+u*4*M+3*M+v);
      __v4du z111 = *reinterpret_cast<const __v4du*>(x+u*4*M+3*M+v+2);
      *reinterpret_cast<__v4du*>(x + u*4*M + 2*M + v) = z100 ^ z110;
      *reinterpret_cast<__v4du*>(x + u*4*M + 2*M + v+2) = z101 ^ z111;
      *reinterpret_cast<__v4du*>(x + u*4*M + 3*M + v) = z010 ^ z110;
      *reinterpret_cast<__v4du*>(x + u*4*M + 3*M + v+2) = z011 ^ z111;
    }
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
void cudaStrassenWinogradKernelA(const uint4* A, uint4* T, int d, int l) {
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



__global__
void cudaAlternativeBasisSelfInverseKernelA(const uint4* A, uint4* T, int d, int l) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int logM = 2*d-2*l+3;
  int M = 1 << logM;
  int u = t >> logM;
  int v = t & ~(0xffffffff << logM);
  
  uint4 A00 = A[u*4*M       + v];
  uint4 A01 = A[u*4*M +   M + v];
  uint4 A10 = A[u*4*M + 2*M + v];
  uint4 A11 = A[u*4*M + 3*M + v];
  
  const uint4& T0 = A00;
  const uint4& T1 = A01;
  const uint4& T2 = A10;
  const uint4& T3 = A11;
  const uint4& T4 = A00^A11;
  const uint4& T5 = A01^A11;
  const uint4& T6 = A10^A11;

  T[u*7*M + 0*M + v] = T0;
  T[u*7*M + 1*M + v] = T1;
  T[u*7*M + 2*M + v] = T2;
  T[u*7*M + 3*M + v] = T3;
  T[u*7*M + 4*M + v] = T4;
  T[u*7*M + 5*M + v] = T5;
  T[u*7*M + 6*M + v] = T6;
}



__global__
void cudaAlternativeBasisChainingKernelA(const uint4* A, uint4* T, int d, int l) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int logM = 2*d-2*l+3;
  int M = 1 << logM;
  int u = t >> logM;
  int v = t & ~(0xffffffff << logM);
  
  uint4 A00 = A[u*4*M       + v];
  uint4 A01 = A[u*4*M +   M + v];
  uint4 A10 = A[u*4*M + 2*M + v];
  uint4 A11 = A[u*4*M + 3*M + v];
  
  const uint4& T0 = A00;
  const uint4& T1 = A01;
  const uint4& T2 = A10;
  const uint4& T3 = A11;
  const uint4& T4 = A00^A10;
  const uint4& T5 = A01^A10;
  const uint4& T6 = A10^A11;

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
void cudaDoubleStrassenWinogradKernelA(const uint4* A, uint4* T, int d, int l) {
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



__global__
void cudaDoubleAlternativeBasisSelfInverseKernelA(const uint4* A, uint4* T, int d, int l) {
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
    const uint4& T0 = A00;
    const uint4& T1 = A01;
    const uint4& T2 = A10;
    const uint4& T3 = A11;
    const uint4& T4 = A00^A11;
    const uint4& T5 = A01^A11;
    const uint4& T6 = A10^A11;

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
    const uint4& T0 = A00;
    const uint4& T1 = A01;
    const uint4& T2 = A10;
    const uint4& T3 = A11;
    const uint4& T4 = A00^A11;
    const uint4& T5 = A01^A11;
    const uint4& T6 = A10^A11;
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
void cudaDoubleAlternativeBasisChainingKernelA(const uint4* A, uint4* T, int d, int l) {
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
    const uint4& T0 = A00;
    const uint4& T1 = A01;
    const uint4& T2 = A10;
    const uint4& T3 = A11;
    const uint4& T4 = A00^A10;
    const uint4& T5 = A01^A10;
    const uint4& T6 = A10^A11;

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
    const uint4& T0 = A00;
    const uint4& T1 = A01;
    const uint4& T2 = A10;
    const uint4& T3 = A11;
    const uint4& T4 = A00^A10;
    const uint4& T5 = A01^A10;
    const uint4& T6 = A10^A11;
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
void cudaStrassenWinogradKernelB(const uint4* B, uint4* S, int d, int l) {
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



__global__
void cudaAlternativeBasisSelfInverseKernelB(const uint4* B, uint4* S, int d, int l) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int logM = 2*d-2*l+3;
  int M = 1 << logM;
  int u = t >> logM;
  int v = t & ~(0xffffffff << logM);
  
  uint4 B00 = B[u*4*M       + v];
  uint4 B10 = B[u*4*M +   M + v];
  uint4 B01 = B[u*4*M + 2*M + v];
  uint4 B11 = B[u*4*M + 3*M + v];
 
  const uint4& S0 = B00;
  const uint4& S1 = B10;
  const uint4& S2 = B00^B11;
  const uint4& S3 = B11;
  const uint4& S4 = B01;
  const uint4& S5 = B01^B11;
  const uint4& S6 = B10^B11;

  S[u*7*M + 0*M + v] = S0;
  S[u*7*M + 1*M + v] = S1;
  S[u*7*M + 2*M + v] = S2;
  S[u*7*M + 3*M + v] = S3;
  S[u*7*M + 4*M + v] = S4;
  S[u*7*M + 5*M + v] = S5;
  S[u*7*M + 6*M + v] = S6;
}



__global__
void cudaAlternativeBasisChainingKernelB(const uint4* B, uint4* S, int d, int l) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int logM = 2*d-2*l+3;
  int M = 1 << logM;
  int u = t >> logM;
  int v = t & ~(0xffffffff << logM);
  
  uint4 B00 = B[u*4*M       + v];
  uint4 B10 = B[u*4*M +   M + v];
  uint4 B01 = B[u*4*M + 2*M + v];
  uint4 B11 = B[u*4*M + 3*M + v];
 
  const uint4& S0 = B00;
  const uint4& S1 = B10^B11;
  const uint4& S2 = B10;
  const uint4& S3 = B11;
  const uint4& S4 = B01;
  const uint4& S5 = B01^B10;
  const uint4& S6 = B00^B10;

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
void cudaDoubleStrassenWinogradKernelB(const uint4* B, uint4* S, int d, int l) {
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



__global__
void cudaDoubleAlternativeBasisSelfInverseKernelB(const uint4* B, uint4* S, int d, int l) {
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
    const uint4& S0 = B00;
    const uint4& S1 = B10;
    const uint4& S2 = B00^B11;
    const uint4& S3 = B11;
    const uint4& S4 = B01;
    const uint4& S5 = B01^B11;
    const uint4& S6 = B10^B11;

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
    const uint4& S0 = B00;
    const uint4& S1 = B10;
    const uint4& S2 = B00^B11;
    const uint4& S3 = B11;
    const uint4& S4 = B01;
    const uint4& S5 = B01^B11;
    const uint4& S6 = B10^B11;
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
void cudaDoubleAlternativeBasisChainingKernelB(const uint4* B, uint4* S, int d, int l) {
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
    const uint4& S0 = B00;
    const uint4& S1 = B10^B11;
    const uint4& S2 = B10;
    const uint4& S3 = B11;
    const uint4& S4 = B01;
    const uint4& S5 = B01^B10;
    const uint4& S6 = B00^B10;

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
    const uint4& S0 = B00;
    const uint4& S1 = B10^B11;
    const uint4& S2 = B10;
    const uint4& S3 = B11;
    const uint4& S4 = B01;
    const uint4& S5 = B01^B10;
    const uint4& S6 = B00^B10;
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
void cudaStrassenWinogradKernelC(const uint4* Ca, uint4* Cb, int d, int l) {
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



__global__
void cudaAlternativeBasisSelfInverseKernelC(const uint4* Ca, uint4* Cb, int d, int l) {
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
  uint4 C00 = Q0^Q1;
  uint4 C01 = Q4^Q6;
  uint4 C10 = Q2^Q5;
  uint4 C11 = Q1^Q3^Q5^Q6;
  Cb[u*4*M       + v] = C00;
  Cb[u*4*M +   M + v] = C01;
  Cb[u*4*M + 2*M + v] = C10;
  Cb[u*4*M + 3*M + v] = C11;
}



__global__
void cudaAlternativeBasisChainingKernelC(const uint4* Ca, uint4* Cb, int d, int l) {
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
  uint4 R = Q1^Q2^Q4;
  uint4 C00 = Q0^Q1;
  uint4 C01 = R^Q5;
  uint4 C10 = R^Q6;
  uint4 C11 = Q3^Q4;
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
void cudaDoubleStrassenWinogradKernelC(const uint4* Ca, uint4* Cb, int d, int l) {
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



__global__
void cudaDoubleAlternativeBasisSelfInverseKernelC(const uint4* Ca, uint4* Cb, int d, int l) {
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
    uint4 C00 = Q0^Q1;
    uint4 C01 = Q4^Q6;
    uint4 C10 = Q2^Q5;
    uint4 C11 = Q1^Q3^Q5^Q6;
    Cs[i][0] = C00;
    Cs[i][1] = C01;
    Cs[i][2] = C10;
    Cs[i][3] = C11;
  }

  for (int i = 0; i < 4; ++i) {
    const uint4& Q0 = Cs[0][i];
    const uint4& Q1 = Cs[1][i];
    const uint4& Q2 = Cs[2][i];
    const uint4& Q3 = Cs[3][i];
    const uint4& Q4 = Cs[4][i];
    const uint4& Q5 = Cs[5][i];
    const uint4& Q6 = Cs[6][i];
    uint4 C00 = Q0^Q1;
    uint4 C01 = Q4^Q6;
    uint4 C10 = Q2^Q5;
    uint4 C11 = Q1^Q3^Q5^Q6;
    Cb[u*16*M +      i*M + v] = C00;
    Cb[u*16*M +  (i+4)*M + v] = C01;
    Cb[u*16*M +  (i+8)*M + v] = C10;
    Cb[u*16*M + (i+12)*M + v] = C11;
  }
}



__global__
void cudaDoubleAlternativeBasisChainingKernelC(const uint4* Ca, uint4* Cb, int d, int l) {
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
    uint4 R = Q1^Q2^Q4;
    uint4 C00 = Q0^Q1;
    uint4 C01 = R^Q5;
    uint4 C10 = R^Q6;
    uint4 C11 = Q3^Q4;
    Cs[i][0] = C00;
    Cs[i][1] = C01;
    Cs[i][2] = C10;
    Cs[i][3] = C11;
  }

  for (int i = 0; i < 4; ++i) {
    const uint4& Q0 = Cs[0][i];
    const uint4& Q1 = Cs[1][i];
    const uint4& Q2 = Cs[2][i];
    const uint4& Q3 = Cs[3][i];
    const uint4& Q4 = Cs[4][i];
    const uint4& Q5 = Cs[5][i];
    const uint4& Q6 = Cs[6][i];
    uint4 R = Q1^Q2^Q4;
    uint4 C00 = Q0^Q1;
    uint4 C01 = R^Q5;
    uint4 C10 = R^Q6;
    uint4 C11 = Q3^Q4;
    Cb[u*16*M +      i*M + v] = C00;
    Cb[u*16*M +  (i+4)*M + v] = C01;
    Cb[u*16*M +  (i+8)*M + v] = C10;
    Cb[u*16*M + (i+12)*M + v] = C11;
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
void cudaDoubleStrassenWinogradCoreMmKernel(const uint4* A, const uint4* B,
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



__global__
void cudaDoubleAlternativeBasisSelfInverseCoreMmKernel(const uint4* A, 
                                                       const uint4* B,
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

  uint4 T0 = A00;
  uint4 T1 = A01;
  uint4 T2 = A10;
  uint4 T3 = A11;
  uint4 T4 = A00^A11;
  uint4 T5 = A01^A11;
  uint4 T6 = A10^A11;

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

    uint4 S0 = B00_s;
    uint4 S1 = B10_s;
    uint4 S2 = B00_s^B11_s;
    uint4 S3 = B11_s;
    uint4 S4 = B01_s;
    uint4 S5 = B01_s^B11_s;
    uint4 S6 = B10_s^B11_s;
    
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

  uint4 C00 = Q0^Q1;
  uint4 C01 = Q4^Q6;
  uint4 C10 = Q2^Q5;
  uint4 C11 = Q1^Q3^Q5^Q6;

  C[u*128      + v] = C00;
  C[u*128 + 32 + v] = C01;
  C[u*128 + 64 + v] = C10; 
  C[u*128 + 96 + v] = C11;
}



__global__
void cudaDoubleAlternativeBasisChainingCoreMmKernel(const uint4* A, 
                                                       const uint4* B,
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

  uint4 T0 = A00;
  uint4 T1 = A01;
  uint4 T2 = A10;
  uint4 T3 = A11;
  uint4 T4 = A00^A10;
  uint4 T5 = A01^A10;
  uint4 T6 = A10^A11;

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

    uint4 S0 = B00_s;
    uint4 S1 = B10_s^B11_s;
    uint4 S2 = B10_s;
    uint4 S3 = B11_s;
    uint4 S4 = B01_s;
    uint4 S5 = B01_s^B10_s;
    uint4 S6 = B00_s^B10_s;
    
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

  uint4 R = Q1^Q2^Q4;
  uint4 C00 = Q0^Q1;
  uint4 C01 = R^Q5;
  uint4 C10 = R^Q6;
  uint4 C11 = Q3^Q4;

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
static void cudaDoubleStrassenWinograd(const uint4* A, const uint4* B, uint4* C,
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
    cudaStrassenWinogradKernelA<<<numBlocks,blockSize>>>(A_d, C_d, d, l);
    CUDA_SYNC;
    std::swap(A_d,C_d);
    cudaStrassenWinogradKernelB<<<numBlocks,blockSize>>>(B_d, C_d, d, l);
    std::swap(B_d,C_d);
    // 7+4 memops / kernel / thread
    MEMOPS += totalThreads*22;
    ++l;
  }

  for (; l < d-1; l += 2) {
    totalThreads = intpow(7,l)*(1<<(2*d-2*l+1));
    determineBlockSizeAndNumBlocks(totalThreads, numBlocks, blockSize);
    cudaDoubleStrassenWinogradKernelA<<<numBlocks,blockSize>>>(A_d, C_d, d, l);
    CUDA_SYNC;
    std::swap(A_d,C_d);
    cudaDoubleStrassenWinogradKernelB<<<numBlocks,blockSize>>>(B_d, C_d, d, l);
    CUDA_SYNC;
    std::swap(B_d,C_d);

    // 4*4 + 7*7 memops / kernel / thread
    MEMOPS += 2*(16+49)*totalThreads;
  }
 
  assert(l == d-1);
  totalThreads = 32*intpow(7,d-1);
  determineBlockSizeAndNumBlocks(totalThreads, numBlocks, blockSize);
  cudaDoubleStrassenWinogradCoreMmKernel<<<numBlocks,blockSize>>>(A_d, B_d, C_d);
  CUDA_SYNC;
  // 12 memops / thread
  MEMOPS += totalThreads*12;

  for (l = d-3; l >= 0; l -= 2) {
    totalThreads = intpow(7,l)*(1<<(2*d-2*l+1));
    determineBlockSizeAndNumBlocks(totalThreads, numBlocks, blockSize);
    cudaDoubleStrassenWinogradKernelC<<<numBlocks,blockSize>>>(C_d, A_d, d, l);
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
    cudaStrassenWinogradKernelC<<<numBlocks,blockSize>>>(C_d, A_d, d, l);
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



static void cudaDoubleAlternativeBasisSelfInverse(const uint4* A, const uint4* B, uint4* C,
                                                  uint4* scratch, int n) {
  int d = intlog2(n) - 6;
  assert((1<<d)*64 == n);

  if (d == 0) {
    // account for the special case of a single 64x64 mm
    cudaMultiplicationKernel<<<1,32>>>(A, B, C);
    CUDA_SYNC;
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
    cudaAlternativeBasisSelfInverseKernelA<<<numBlocks,blockSize>>>(A_d, C_d, d, l);
    CUDA_SYNC;
    std::swap(A_d,C_d);
    cudaAlternativeBasisSelfInverseKernelB<<<numBlocks,blockSize>>>(B_d, C_d, d, l);
    std::swap(B_d,C_d);
    ++l;
  }

  for (; l < d-1; l += 2) {
    totalThreads = intpow(7,l)*(1<<(2*d-2*l+1));
    determineBlockSizeAndNumBlocks(totalThreads, numBlocks, blockSize);
    cudaDoubleAlternativeBasisSelfInverseKernelA<<<numBlocks,blockSize>>>(A_d, C_d, d, l);
    CUDA_SYNC;
    std::swap(A_d,C_d);
    cudaDoubleAlternativeBasisSelfInverseKernelB<<<numBlocks,blockSize>>>(B_d, C_d, d, l);
    CUDA_SYNC;
    std::swap(B_d,C_d);
  }
 
  assert(l == d-1);
  totalThreads = 32*intpow(7,d-1);
  determineBlockSizeAndNumBlocks(totalThreads, numBlocks, blockSize);
  cudaDoubleAlternativeBasisSelfInverseCoreMmKernel<<<numBlocks,blockSize>>>(A_d, B_d, C_d);
  CUDA_SYNC;

  for (l = d-3; l >= 0; l -= 2) {
    totalThreads = intpow(7,l)*(1<<(2*d-2*l+1));
    determineBlockSizeAndNumBlocks(totalThreads, numBlocks, blockSize);
    cudaDoubleAlternativeBasisSelfInverseKernelC<<<numBlocks,blockSize>>>(C_d, A_d, d, l);
    CUDA_SYNC;
    std::swap(C_d,A_d);
  }

  if (d % 2 == 0) {
    assert(l == -1);
    l = 0;
    totalThreads = 1 << (2*d+3);
    determineBlockSizeAndNumBlocks(totalThreads, numBlocks, blockSize);
    cudaAlternativeBasisSelfInverseKernelC<<<numBlocks,blockSize>>>(C_d, A_d, d, l);
    CUDA_SYNC;
    std::swap(C_d,A_d);
  }

  gpuMemcpy(32*(1<<2*d), C_d, C);
}



static void cudaDoubleAlternativeBasisChaining(const uint4* A, const uint4* B, uint4* C,
                                                  uint4* scratch, int n) {
  int d = intlog2(n) - 6;
  assert((1<<d)*64 == n);

  if (d == 0) {
    // account for the special case of a single 64x64 mm
    cudaMultiplicationKernel<<<1,32>>>(A, B, C);
    CUDA_SYNC;
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
    cudaAlternativeBasisChainingKernelA<<<numBlocks,blockSize>>>(A_d, C_d, d, l);
    CUDA_SYNC;
    std::swap(A_d,C_d);
    cudaAlternativeBasisChainingKernelB<<<numBlocks,blockSize>>>(B_d, C_d, d, l);
    std::swap(B_d,C_d);
    ++l;
  }

  for (; l < d-1; l += 2) {
    totalThreads = intpow(7,l)*(1<<(2*d-2*l+1));
    determineBlockSizeAndNumBlocks(totalThreads, numBlocks, blockSize);
    cudaDoubleAlternativeBasisChainingKernelA<<<numBlocks,blockSize>>>(A_d, C_d, d, l);
    CUDA_SYNC;
    std::swap(A_d,C_d);
    cudaDoubleAlternativeBasisChainingKernelB<<<numBlocks,blockSize>>>(B_d, C_d, d, l);
    CUDA_SYNC;
    std::swap(B_d,C_d);
  }
 
  assert(l == d-1);
  totalThreads = 32*intpow(7,d-1);
  determineBlockSizeAndNumBlocks(totalThreads, numBlocks, blockSize);
  cudaDoubleAlternativeBasisChainingCoreMmKernel<<<numBlocks,blockSize>>>(A_d, B_d, C_d);
  CUDA_SYNC;

  for (l = d-3; l >= 0; l -= 2) {
    totalThreads = intpow(7,l)*(1<<(2*d-2*l+1));
    determineBlockSizeAndNumBlocks(totalThreads, numBlocks, blockSize);
    cudaDoubleAlternativeBasisChainingKernelC<<<numBlocks,blockSize>>>(C_d, A_d, d, l);
    CUDA_SYNC;
    std::swap(C_d,A_d);
  }

  if (d % 2 == 0) {
    assert(l == -1);
    l = 0;
    totalThreads = 1 << (2*d+3);
    determineBlockSizeAndNumBlocks(totalThreads, numBlocks, blockSize);
    cudaAlternativeBasisChainingKernelC<<<numBlocks,blockSize>>>(C_d, A_d, d, l);
    CUDA_SYNC;
    std::swap(C_d,A_d);
  }

  gpuMemcpy(32*(1<<2*d), C_d, C);
}



static void cudaRecursiveDoubleAlternativeBasisSelfInverse(const uint4* A, const uint4* B,
                                                           uint4* C, uint4* scratch,
                                                           int n) {
  assert(isPowerOfTwo(n));
  int d = intlog2(n) - 6;
  if (d <= 8) {
    cudaDoubleAlternativeBasisSelfInverse(A, B, C, scratch, n);
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

  gpuMemcpy(submatrixLength, A00, T[0]);
  gpuMemcpy(submatrixLength, A01, T[1]);
  gpuMemcpy(submatrixLength, A10, T[2]);
  gpuMemcpy(submatrixLength, A11, T[3]);
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(A00,A11,T[4]);
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(A01,A11,T[5]);
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(A10,A11,T[6]);
  CUDA_SYNC;

  gpuMemcpy(submatrixLength, B00, S[0]);
  gpuMemcpy(submatrixLength, B10, S[1]);
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(B00,B11,S[2]);
  gpuMemcpy(submatrixLength, B11, S[3]);
  gpuMemcpy(submatrixLength, B01, S[4]);
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(B01,B11,S[5]);
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(B10,B11,S[6]);
  CUDA_SYNC;
    
  // operands must be reused because of memory allocation
  for (int i = 0; i < 7; ++i) 
    cudaRecursiveDoubleAlternativeBasisSelfInverse(T[i], S[i], Q[i], scratch + submatrixLength*21, n>>1);

  cudaUint4XorKernel<<<numBlocks,blockSize>>>(Q[1],Q[3],T[0]);
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(Q[5],Q[6],T[1]);
  CUDA_SYNC;

  uint4* C00 = C;
  uint4* C01 = C + submatrixLength;
  uint4* C10 = C + 2*submatrixLength;
  uint4* C11 = C + 3*submatrixLength;

  cudaUint4XorKernel<<<numBlocks,blockSize>>>(Q[0],Q[1],C00);
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(Q[4],Q[6],C01);
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(Q[2],Q[5],C10);
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(T[0],T[1],C11);
  CUDA_SYNC;
}



static void cudaRecursiveDoubleAlternativeBasisChaining(const uint4* A, const uint4* B,
                                                           uint4* C, uint4* scratch,
                                                           int n) {
  assert(isPowerOfTwo(n));
  int d = intlog2(n) - 6;
  if (d <= 8) {
    cudaDoubleAlternativeBasisChaining(A, B, C, scratch, n);
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

  gpuMemcpy(submatrixLength, A00, T[0]);
  gpuMemcpy(submatrixLength, A01, T[1]);
  gpuMemcpy(submatrixLength, A10, T[2]);
  gpuMemcpy(submatrixLength, A11, T[3]);
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(A00,A10,T[4]);
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(A01,A10,T[5]);
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(A10,A11,T[6]);
  CUDA_SYNC;

  gpuMemcpy(submatrixLength, B00, S[0]);
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(B10,B11,S[1]);
  gpuMemcpy(submatrixLength, B10, S[2]);
  gpuMemcpy(submatrixLength, B11, S[3]);
  gpuMemcpy(submatrixLength, B01, S[4]);
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(B01,B10,S[5]);
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(B00,B10,S[6]);
  CUDA_SYNC;
    
  // operands must be reused because of memory allocation
  for (int i = 0; i < 7; ++i) 
    cudaRecursiveDoubleAlternativeBasisChaining(T[i], S[i], Q[i], scratch + submatrixLength*21, n>>1);

  cudaUint4XorKernel<<<numBlocks,blockSize>>>(Q[1],Q[2],T[0]);
  CUDA_SYNC;
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(T[0],Q[4],T[1]);
  CUDA_SYNC;
  uint4* R = T[1];

  uint4* C00 = C;
  uint4* C01 = C + submatrixLength;
  uint4* C10 = C + 2*submatrixLength;
  uint4* C11 = C + 3*submatrixLength;

  cudaUint4XorKernel<<<numBlocks,blockSize>>>(Q[0],Q[1],C00);
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(R,Q[5],C01);
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(R,Q[6],C10);
  cudaUint4XorKernel<<<numBlocks,blockSize>>>(Q[3],Q[4],C11);
  CUDA_SYNC;
}



static void cudaRecursiveDoubleStrassenWinograd(const uint4* A, const uint4* B,
                                                uint4* C, uint4* scratch,
                                                int n) {
  assert(isPowerOfTwo(n));
  int d = intlog2(n) - 6;
  if (d <= 8) {
    cudaDoubleStrassenWinograd(A, B, C, scratch, n);
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
    cudaRecursiveDoubleStrassenWinograd(T[i], S[i], Q[i], scratch + submatrixLength*21, n>>1);

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



enum class Mode {
  NIL, CUBIC, BOOLEAN, STRASSEN_WINOGRAD, ALTERNATIVE_BASIS_SELF_INVERSE, 
  ALTERNATIVE_BASIS_CHAINING, TRANSPOSE, CHANGE_OF_BASIS
};

enum class Action {
  NIL, TEST, EVALUATE
};



// computes CPU scratch size (as the number of uint4s to reserve)
// subn = subproblem size
static int64_t computeCpuScratchSize(int64_t subn, int64_t lanes, int64_t auxmats) {
  int64_t d = intlog2(subn)-6;
  return 3L*(1L<<2*d)*32L*GPU_COUNT*lanes*auxmats;
}



// computes the amount of GPU memory that needs to be reserved given the mode
static int64_t computeGpuMemorySize(int64_t n, Mode mode) {
  // n = 2^d * 64
  int d = intlog2(n) - 6;

  if (mode == Mode::CUBIC || mode == Mode::BOOLEAN) {
    // required memory: 3*subproblem size
    return 3L*(1L<<2*d)*32L;
  }
  else if (mode == Mode::STRASSEN_WINOGRAD ||
           mode == Mode::ALTERNATIVE_BASIS_SELF_INVERSE ||
           mode == Mode::ALTERNATIVE_BASIS_CHAINING) {
    // memory requirement is 3*operand size + scratch
    // for d <= 8, scratch requirement is 3*32*7^d
    // for d > 8, add 7*3*32*4^d for i = 8,...,d-1
    if (d <= 8)
      return 3L*32L*((1<<2*d) + intpow(7,d));
    else {
      int64_t sz = 3L*32L*((1<<2*d) + intpow(7,8));
      for (int i = 8; i < d; ++i)
        sz += 3L*32L*7L*(1<<2*i);
      return sz;
    }
  }
  if (mode == Mode::TRANSPOSE) {
    // required memory: 0
      return 0;
  }

  else {
    assert(false && "invalid mode");
    return -1;
  }
}



static void toBaseR(const int64_t* rankp, int q, int64_t i, int64_t* ibaser) {
  for (int j = q-1; j >= 0; --j) {         
    ibaser[j] = i;
    for (int k = j+1; k < q; ++k)
      ibaser[j] -= ibaser[k]*rankp[k];
    ibaser[j] /= rankp[j];
  }
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
  const int MAX_Q = 20;
  assert(numberOfLanes <= MAXIMUM_NUMBER_OF_LANES);
  assert(numberOfAuxiliaryMatrices <= MAXIMUM_NUMBER_OF_AUXILIARY_MATRICES);
  assert(numberOfLanes % GPU_COUNT == 0);
  assert(numberOfLanes/GPU_COUNT*numberOfAuxiliaryMatrices <= 8);
  
  assert(subn >= 64);
  assert(isPowerOfTwo(n));
  assert(isPowerOfTwo(subn));
  const int64_t d = intlog2(subn)-6;
  const int64_t q = intlog2(n)-d-6;
  const int64_t submatrixSize = (1L<<2*d)*32;
  assert(q < MAX_Q);

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

  // total number of submatrices: 4^q // total number of submatrices: 256
  // iterate over rank^q *** 7^4 times and select the appropriate submatrices
  int firstC[1<<2*q];
  for (int i = 0; i < (1<<2*q); ++i)
    firstC[i] = true;

  vector<mutex> cMutexes(1<<2*q);
  vector<mutex> gpuMutexes(GPU_COUNT);
  array<mutex,MAXIMUM_NUMBER_OF_LANES> laneMutexes;
  array<condition_variable,MAXIMUM_NUMBER_OF_LANES> laneCvs;

  int64_t nanosPerThread[TOTAL_NUMBER_OF_THREADS];
  int64_t wordsPerThread[TOTAL_NUMBER_OF_THREADS];

  int64_t rankp[q+1];
  int64_t fourp[q+1];
  fourp[0] = rankp[0] = 1;
  for (int i = 1; i < q+1; ++i) {
    rankp[i] = rankp[i-1]*rank;
    fourp[i] = fourp[i-1]*4;
  }
  
#pragma omp parallel num_threads(TOTAL_NUMBER_OF_THREADS)
  {
    int t = omp_get_thread_num();
    // let g be the number of GPUs
    // assuming 1 lane
    // threads 0..g-1 create T
    // threads g..2g-1 create S
    // threads 2g..3g-1 perform upload & mm & download
    // threads 3g..4g-1 perform result integration

    nanosPerThread[t] = 0;
    wordsPerThread[t] = 0;
      
    int gpu = t % GPU_COUNT;
    int lane = t % numberOfLanes;
    cudaSetDevice(gpu);
    uint4* A_d = A_ds[gpu];
    uint4* B_d = B_ds[gpu];
    uint4* C_d = C_ds[gpu];
    uint4* scratch_d = scratch_ds[gpu];
    int64_t max_i = rankp[q];
    int64_t ibaser[q];
    int64_t jbase4[q];
    for (int64_t i = lane; i < max_i; i += numberOfLanes) {
      int auxiliaryMatrixNumber = (i/numberOfLanes)%numberOfAuxiliaryMatrices;
      uint4* T = Ts[lane][auxiliaryMatrixNumber];
      uint4* S = Ss[lane][auxiliaryMatrixNumber];
      uint4* Q = Qs[lane][auxiliaryMatrixNumber];
        
      // base r representation
      toBaseR(rankp, q, i, ibaser);

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
        for (int64_t j = 0; j < 1 << (2*q); ++j) {
          toBaseR(fourp, q, j, jbase4);
          int allones = 1;
          for (int k = 0; k < q; ++k)
            allones *= tVectors[ibaser[k]][jbase4[k]];
          if (allones) {
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
        for (int64_t j = 0; j < 1 << (2*q); ++j) {
          toBaseR(fourp, q, j, jbase4);
          int allones = 1;
          for (int k = 0; k < q; ++k)
            allones *= sVectors[ibaser[k]][jbase4[k]];
            
          if (allones) {
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
        for (int64_t j = 0; j < 1 << (2*q); ++j) {
          toBaseR(fourp, q, j, jbase4);
          int allones = 1;
          for (int k = 0; k < q; ++k)
            allones *= qVectors[ibaser[k]][jbase4[k]];
          if (allones) {
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

#ifdef DEBUG_REPORT_PER_THREAD_TIMES_AND_WORDS
  for (int t = 0; t < TOTAL_NUMBER_OF_THREADS; ++t) 
    cerr << "thread " << t << ": " << nanosPerThread[t] << " ns " 
         << wordsPerThread[t] << " words" << endl;
#endif // DEBUG_REPORT_PER_THREAD_TIMES_AND_WORDS
}



static void multiGpuDoubleStrassenWinograd(int64_t n, int64_t subn,
                                           const uint4* A,
                                           const uint4* B, uint4* C,
                                           uint4* cpuScratch,
                                           uint4** A_ds,
                                           uint4** B_ds, uint4** C_ds,
                                           uint4** scratch_ds,
                                           int numberOfLanes,
                                           int numberOfAuxiliaryMatrices) {
  multiGpuPipelinedDouble(n,A,B,C,cpuScratch,A_ds,B_ds,C_ds,scratch_ds,
                          STRASSEN_WINOGRAD_T_VECTORS,
                          STRASSEN_WINOGRAD_S_VECTORS,
                          STRASSEN_WINOGRAD_Q_VECTORS,
                          cudaRecursiveDoubleStrassenWinograd, 7, subn,
                          numberOfLanes, numberOfAuxiliaryMatrices);
}



static void multiGpuDoubleAlternativeBasisSelfInverse(int64_t n, int64_t subn,
                                                      const uint4* A,
                                                      const uint4* B, uint4* C,
                                                      uint4* cpuScratch,
                                                      uint4** A_ds,
                                                      uint4** B_ds, 
                                                      uint4** C_ds,
                                                      uint4** scratch_ds,
                                                      int numberOfLanes,
                                                      int numberOfAuxiliaryMatrices) {
  multiGpuPipelinedDouble(n, A, B, C, cpuScratch, A_ds, B_ds, C_ds, scratch_ds,
                          ALTERNATIVE_BASIS_SELF_INVERSE_T_VECTORS,
                          ALTERNATIVE_BASIS_SELF_INVERSE_S_VECTORS,
                          ALTERNATIVE_BASIS_SELF_INVERSE_Q_VECTORS,
                          cudaRecursiveDoubleAlternativeBasisSelfInverse, 7, 
                          subn, numberOfLanes, numberOfAuxiliaryMatrices);
}



static void multiGpuDoubleAlternativeBasisChaining(int64_t n, int64_t subn,
                                                      const uint4* A,
                                                      const uint4* B, uint4* C,
                                                      uint4* cpuScratch,
                                                      uint4** A_ds,
                                                      uint4** B_ds, 
                                                      uint4** C_ds,
                                                      uint4** scratch_ds,
                                                      int numberOfLanes,
                                                      int numberOfAuxiliaryMatrices) {
  multiGpuPipelinedDouble(n, A, B, C, cpuScratch, A_ds, B_ds, C_ds, scratch_ds,
                          ALTERNATIVE_BASIS_CHAINING_T_VECTORS,
                          ALTERNATIVE_BASIS_CHAINING_S_VECTORS,
                          ALTERNATIVE_BASIS_CHAINING_Q_VECTORS,
                          cudaRecursiveDoubleAlternativeBasisChaining, 7, 
                          subn, numberOfLanes, numberOfAuxiliaryMatrices);
}



static uint4* allocateAndZeroCpuScratch(int64_t subn, int64_t lanes, int64_t auxmats) {
  int64_t cpuScratchSize = computeCpuScratchSize(subn, lanes, auxmats);
  uint4* p = allocateAlignedMemory(cpuScratchSize);
  avxSet(p, cpuScratchSize, 0);
  return p;
}


static void testTranspose() {
  BinaryMatrix A = BinaryMatrix::random(64);
  BinaryMatrix A_a(64,64);
  arrangeDataCpuWords(A.getDataPointer(), A_a.getDataPointer(), 64*64);
  transposeNaiveInplace64(A_a.getDataPointer());
  BinaryMatrix At(64,64);
  dearrangeDataCpuWords(A_a.getDataPointer(), At.getDataPointer(), 64*64);
  assert(A.transpose() == At);
  cout << "64x64 naive transpose ok" << endl;

  assert(A != At);
  transposeInplaceNonarranged64(A.getDataPointer());
  assert(A == At);
  cout << "64x64 non-rearranged straight line transpose ok" << endl;

  arrangeDataCpuWords(A.getDataPointer(), A_a.getDataPointer(), 64*64);
  transposeInplaceArranged64(A_a.getDataPointer());
  dearrangeDataCpuWords(A_a.getDataPointer(), At.getDataPointer(), 64*64);
  assert(A.transpose() == At);
  cout << "64x64 rearranged straight line transpose ok" << endl;

  pushTime();
  A = BinaryMatrix::random(16384);
  pushTime();
  auto nanos = getDiffNanos();
  cout << "random data generation took took " << nanosToString(nanos) << endl;
  A_a = BinaryMatrix(16384,16384);
  At = BinaryMatrix(16384,16384);
  arrangeDataCpuWords(A.getDataPointer(), A_a.getDataPointer(), 16384L*16384);
  transposeInplaceArranged(A_a.getDataPointer(), 16384);  
  dearrangeDataCpuWords(A_a.getDataPointer(), At.getDataPointer(), 16384L*16384);
  for (int64_t idx_i = 0; idx_i < 256; ++idx_i)
    for (int64_t idx_j = 0; idx_j < 256; ++idx_j)
      for (int64_t i = 0; i < 64; ++i)
        for (int64_t j = 0; j < 64; ++j) 
          assert(A.get(idx_i*64 + i, idx_j*64+j) == At.get(idx_i*64 + j, idx_j*64+i));
        
  cout << "16384x16384 all 64x64 submatrices transpose ok" << endl;
  
  pushTime();
  A = BinaryMatrix::random(1L << 20);
  pushTime();
  nanos = getDiffNanos();
  cout << "random data generation took took " << nanosToString(nanos) << endl;

  pushTime();
#pragma omp parallel for
  for (int64_t i = 0; i < (1L << 28); ++i)
    transposeNaiveInplace64(A.getDataPointer() + 32*i);
  pushTime();
  nanos = getDiffNanos();
  cout << "4^14 naive transposes took " << nanosToString(nanos) << endl;

  pushTime();
#pragma omp parallel for
  for (int64_t i = 0; i < (1L << 28); ++i)
    transposeInplaceNonarranged64(A.getDataPointer() + 32*i);
  pushTime();
  nanos = getDiffNanos();
  cout << "4^14 non-rearranged straight-line transposes took " << nanosToString(nanos) << endl;

  pushTime();
#pragma omp parallel for
  for (int64_t i = 0; i < (1L << 28); ++i)
    transposeInplaceArranged64(A.getDataPointer() + 32*i);
  pushTime();
  nanos = getDiffNanos();
  cout << "4^14 rearranged straight-line transposes took " << nanosToString(nanos) << endl;
}



static void testChangeOfBasis() {
  BinaryMatrix A = BinaryMatrix::random(64);
  BinaryMatrix B(A);
  assert(A.getDataPointer() != B.getDataPointer());
  assert(A == B);

  changeOfBasisSelfInverseInPlace(A.getDataPointer(), 64);
  assert(A == B);

  inverseChangeOfBasisSelfInverseInPlace(A.getDataPointer(), 64);
  assert(A == B);

  changeOfBasisChainingLeftInPlace(A.getDataPointer(), 64);
  assert(A == B);

  changeOfBasisChainingRightInPlace(A.getDataPointer(), 64);
  assert(A == B);

  inverseChangeOfBasisChainingInPlace(A.getDataPointer(), 64);
  assert(A == B);

  for (int64_t n = 64; n <= (1L << 20); n <<= 1) {
    pushTime();
    A = BinaryMatrix::random(n);
    pushTime();
    auto nanos = getDiffNanos();
    cout << "random data generation " << n << "x" << n << " took " 
         << nanosToString(nanos) << endl;

    pushTime();
    B = BinaryMatrix(A);
    pushTime();
    nanos = getDiffNanos();
    cout << "matrix copying " << n << "x" << n << " took " 
         << nanosToString(nanos) << endl;

    pushTime();
    changeOfBasisSelfInverseInPlace(A.getDataPointer(), n);
    pushTime();
    nanos = getDiffNanos();
    cout << "non-parallel changeOfBasisSelfInverseInPlace " << n << "x" << n 
         << " took " << nanosToString(nanos) << endl;

    pushTime();
    changeOfBasisSelfInverseInPlaceAvxParallel(B.getDataPointer(), n);
    pushTime();
    nanos = getDiffNanos();
    cout << "avx parallel changeOfBasisSelfInverseInPlace " << n << "x" << n 
         << " took " << nanosToString(nanos) << endl;

    assert(A == B);
    cout << "changeOfBasisSelfInverseInPlace " << n << "x" << n 
         << " ok" << endl;

    pushTime();
    changeOfBasisChainingLeftInPlace(A.getDataPointer(), n);
    pushTime();
    nanos = getDiffNanos();
    cout << "non-parallel changeOfBasisChainingLeftInPlace " << n << "x" << n 
         << " took " << nanosToString(nanos) << endl;

    pushTime();
    changeOfBasisChainingLeftInPlaceAvxParallel(B.getDataPointer(), n);
    pushTime();
    nanos = getDiffNanos();
    cout << "avx parallel changeOfBasisChainingLeftInPlace " << n << "x" << n 
         << " took " << nanosToString(nanos) << endl;

    assert(A == B);
    cout << "changeOfBasisChainingLeftInPlace " << n << "x" << n 
         << " ok" << endl;

    pushTime();
    changeOfBasisChainingRightInPlace(A.getDataPointer(), n);
    pushTime();
    nanos = getDiffNanos();
    cout << "non-parallel changeOfBasisChainingRightInPlace " << n << "x" << n 
         << " took " << nanosToString(nanos) << endl;

    pushTime();
    changeOfBasisChainingRightInPlaceAvxParallel(B.getDataPointer(), n);
    pushTime();
    nanos = getDiffNanos();
    cout << "avx parallel changeOfBasisChainingRightInPlace " << n << "x" << n 
         << " took " << nanosToString(nanos) << endl;

    assert(A == B);
    cout << "changeOfBasisChainingRightInPlace " << n << "x" << n 
         << " ok" << endl;

    pushTime();
    inverseChangeOfBasisSelfInverseInPlace(A.getDataPointer(), n);
    pushTime();
    nanos = getDiffNanos();
    cout << "non-parallel inverseChangeOfBasisSelfInverseInPlace " << n << "x" << n 
         << " took " << nanosToString(nanos) << endl;

    pushTime();
    inverseChangeOfBasisSelfInverseInPlaceAvxParallel(B.getDataPointer(), n);
    pushTime();
    nanos = getDiffNanos();
    cout << "avx parallel inverseChangeOfBasisSelfInverseInPlace " << n << "x" << n 
         << " took " << nanosToString(nanos) << endl;

    assert(A == B);
    cout << "inverseChangeOfBasisSelfInverseInPlace " << n << "x" << n 
         << " ok" << endl;

    pushTime();
    inverseChangeOfBasisChainingInPlace(A.getDataPointer(), n);
    pushTime();
    nanos = getDiffNanos();
    cout << "non-parallel inverseChangeOfBasisChainingInPlace " << n << "x" << n 
         << " took " << nanosToString(nanos) << endl;

    pushTime();
    inverseChangeOfBasisChainingInPlaceAvxParallel(B.getDataPointer(), n);
    pushTime();
    nanos = getDiffNanos();
    cout << "avx parallel inverseChangeOfBasisChainingInPlace " << n << "x" << n 
         << " took " << nanosToString(nanos) << endl;

    assert(A == B);
    cout << "inverseChangeOfBasisChainingInPlace " << n << "x" << n 
         << " ok" << endl;
  }

}



static void test(Mode mode, int64_t maxInstanceSize, int64_t maxSubproblemSize,
                 int lanes, int auxmats) {
  if (mode == Mode::TRANSPOSE) {
    // do something completely different
    testTranspose();
    return;
  }

  if (mode == Mode::CHANGE_OF_BASIS) {
    testChangeOfBasis();
    return;
  }  
    
  pushTime();
  uint4* cpuScratch = allocateAndZeroCpuScratch(maxSubproblemSize, lanes, auxmats);
  pushTime();
  cout << "cpu scratch allocation and zeroing took " << nanosToString(getDiffNanos()) << endl;

  pushTime();
  int64_t gpuMemorySize = computeGpuMemorySize(maxSubproblemSize, mode);
  vector<uint4*> gpuMemory(GPU_COUNT,nullptr);
  vector<uint4*> A_ds(GPU_COUNT,nullptr);
  vector<uint4*> B_ds(GPU_COUNT,nullptr);
  vector<uint4*> C_ds(GPU_COUNT,nullptr);
  vector<uint4*> scratch_ds(GPU_COUNT,nullptr);
  for (int i = 0; i < GPU_COUNT; ++i) {
    cudaSetDevice(i);
    gpuMemory[i] = gpuAllocate(gpuMemorySize);
  }
  pushTime();
  cout << "gpu memory allocation took " << nanosToString(getDiffNanos()) << endl;
  
  for (int64_t n = 64; n <= maxInstanceSize; n <<= 1) {
    int q, d;
    if (n <= maxSubproblemSize) {
      q = 0;
      d = intlog2(n)-6;
    }
    else {
      d = intlog2(maxSubproblemSize)-6;
      q = intlog2(n)-6-d;
    }
    assert((1<<q)*(1<<d)*64 == n);

    int64_t nWordsGpu = (1L<<2*d)*32;
    for (int i = 0; i < GPU_COUNT; ++i) {
      A_ds[i] = gpuMemory[i];
      B_ds[i] = gpuMemory[i]+nWordsGpu;
      C_ds[i] = gpuMemory[i]+2*nWordsGpu;
      scratch_ds[i] = gpuMemory[i]+3*nWordsGpu;
    }

    pushTime();
    BinaryMatrix A, B, C(n,n), A_a(n,n), B_a(n,n), C_a(n,n);
    if (mode == Mode::BOOLEAN) {
      A = BinaryMatrix::booleanRandom(n);
      B = BinaryMatrix::booleanRandom(n);
    }
    else {
      A = BinaryMatrix::random(n);
      B = BinaryMatrix::random(n);
    }
    
    pushTime();
    cout << "data generation " << n << "x" << n << " took " << nanosToString(getDiffNanos()) << endl;
    
    if (mode == Mode::CUBIC || mode == Mode::BOOLEAN) {
      if (n <= maxSubproblemSize) {
        pushTime();
        arrangeDataForCubicCpu(A.getDataPointer(), A_a.getDataPointer(), n*n);
        arrangeDataForCubicCpu(B.getDataPointer(), B_a.getDataPointer(), n*n);
        pushTime();
        cout << "cubic data arrangement " << n << "x" << n << " took " << nanosToString(getDiffNanos()) << endl;
        
        cudaSetDevice(0);
        pushTime();
        gpuUpload((1<<2*d)*32, A_a.getDataPointer(), A_ds[0]);
        gpuUpload((1<<2*d)*32, B_a.getDataPointer(), B_ds[0]);
        if (mode == Mode::CUBIC)
          cudaCubicMultiplication(A_ds[0], B_ds[0], C_ds[0], nullptr, n);
        else if (mode == Mode::BOOLEAN)
          cudaBooleanCubicMultiplication(A_ds[0], B_ds[0], C_ds[0], nullptr, n);
        gpuDownload((1<<2*d)*32, C_ds[0], C_a.getDataPointer());
        pushTime();
        if (mode == Mode::CUBIC)
          cout << "cudaCubicMultiplication " << n << "x" << n << " took " << nanosToString(getDiffNanos()) << endl;         
        else if (mode == Mode::BOOLEAN)
          cout << "cudaBooleanCubicMultiplication " << n << "x" << n << " took " << nanosToString(getDiffNanos()) << endl;         

        pushTime();
        dearrangeDataForCubicCpu(C_a.getDataPointer(), C.getDataPointer(), n*n);
        pushTime();
        cout << "cubic data dearrangement " << n << "x" << n << " took " << nanosToString(getDiffNanos()) << endl;
      }
      else {
        pushTime();
        arrangeDataForMultiGpuCubicCpu(A.getDataPointer(), A_a.getDataPointer(), q, d);
        arrangeDataForMultiGpuCubicCpu(B.getDataPointer(), B_a.getDataPointer(), q, d);
        pushTime();
        cout << "multigpu cubic data arrangement " << n << "x" << n << " took " << nanosToString(getDiffNanos()) << endl;

        pushTime();
        if (mode == Mode::CUBIC)
          multiGpuCubic(A_a.getDataPointer(), B_a.getDataPointer(),
                        C_a.getDataPointer(), n, maxSubproblemSize, &A_ds[0], &B_ds[0], &C_ds[0]);
        else if (mode == Mode::BOOLEAN)
          multiGpuBooleanCubic(A_a.getDataPointer(), B_a.getDataPointer(),
                               C_a.getDataPointer(), n, maxSubproblemSize, &A_ds[0], &B_ds[0], &C_ds[0]);
          
        pushTime();
        if (mode == Mode::CUBIC)
          cout << "multiGpuCubic " << n << "x" << n << " took " << nanosToString(getDiffNanos()) << endl;
        else if (mode == Mode::BOOLEAN)
          cout << "multiGpuBoolean " << n << "x" << n << " took " << nanosToString(getDiffNanos()) << endl;
        
        pushTime();
        dearrangeDataForMultiGpuCubicCpu(C_a.getDataPointer(), C.getDataPointer(), q, d);
        pushTime();
        cout << "multigpu cubic data dearrangement " << n << "x" << n << " took " << nanosToString(getDiffNanos()) << endl;
      }
    }
    else if (mode == Mode::STRASSEN_WINOGRAD || 
             mode == Mode::ALTERNATIVE_BASIS_SELF_INVERSE ||
             mode == Mode::ALTERNATIVE_BASIS_CHAINING) {
      pushTime();
      arrangeDataCpuWords(A.getDataPointer(), A_a.getDataPointer(), n*n);
      arrangeDataCpuWords(B.getDataPointer(), B_a.getDataPointer(), n*n);
      pushTime();
      cout << "data arrangement " << n << "x" << n << " took " << nanosToString(getDiffNanos()) << endl;

      if (mode == Mode::ALTERNATIVE_BASIS_SELF_INVERSE) {
        pushTime();
        changeOfBasisSelfInverseInPlace(A_a.getDataPointer(), n);
        changeOfBasisSelfInverseInPlace(B_a.getDataPointer(), n);
        pushTime();
        cout << "change of basis " << n << "x" << n << " took " << nanosToString(getDiffNanos()) << endl;
      }
      else if (mode == Mode::ALTERNATIVE_BASIS_CHAINING) {
        pushTime();
        changeOfBasisChainingLeftInPlace(A_a.getDataPointer(), n);
        changeOfBasisChainingRightInPlace(B_a.getDataPointer(), n);
        pushTime();
        cout << "change of basis " << n << "x" << n << " took " << nanosToString(getDiffNanos()) << endl;
      }

      if (n <= maxSubproblemSize) {
        pushTime();
        cudaSetDevice(0);
        gpuUpload((1<<2*d)*32, A_a.getDataPointer(), A_ds[0]);
        gpuUpload((1<<2*d)*32, B_a.getDataPointer(), B_ds[0]);
        if (mode == Mode::STRASSEN_WINOGRAD)
          cudaRecursiveDoubleStrassenWinograd(A_ds[0], B_ds[0], C_ds[0], scratch_ds[0], n);
        else if (mode == Mode::ALTERNATIVE_BASIS_SELF_INVERSE)
          cudaRecursiveDoubleAlternativeBasisSelfInverse(A_ds[0], B_ds[0], 
                                                         C_ds[0], scratch_ds[0],
                                                         n);
        else if (mode == Mode::ALTERNATIVE_BASIS_CHAINING)
          cudaRecursiveDoubleAlternativeBasisChaining(A_ds[0], B_ds[0], C_ds[0],
                                                      scratch_ds[0], n);
        else
          assert(false && "invalid mode");
        gpuDownload((1<<2*d)*32, C_ds[0], C_a.getDataPointer());
        pushTime();

        cout << (mode == Mode::STRASSEN_WINOGRAD ? "cudaRecursiveAlternateBasisDoubleStrassen " :
                 mode == Mode::ALTERNATIVE_BASIS_SELF_INVERSE ? "cudaRecursiveDoubleAlternativeBasisSelfInverse" :
                 mode == Mode::ALTERNATIVE_BASIS_CHAINING ? "cudaRecursiveDoubleAlternativeBasisSelfInverse" : 
                 "ERROR") << " " << n << "x" << n
               << " took " << nanosToString(getDiffNanos()) << endl;
      }
      else {
        pushTime();
        if (mode == Mode::STRASSEN_WINOGRAD)
          multiGpuDoubleStrassenWinograd(n, maxSubproblemSize, A_a.getDataPointer(),
                                         B_a.getDataPointer(),
                                         C_a.getDataPointer(),
                                         cpuScratch, &A_ds[0],
                                         &B_ds[0], &C_ds[0],
                                         &scratch_ds[0], lanes*GPU_COUNT, auxmats);
        else if (mode == Mode::ALTERNATIVE_BASIS_SELF_INVERSE)
          multiGpuDoubleAlternativeBasisSelfInverse(n, maxSubproblemSize, 
                                                    A_a.getDataPointer(),
                                                    B_a.getDataPointer(),
                                                    C_a.getDataPointer(),
                                                    cpuScratch, &A_ds[0],
                                                    &B_ds[0], &C_ds[0],
                                                    &scratch_ds[0], 
                                                    lanes*GPU_COUNT, 
                                                    auxmats);
        else if (mode == Mode::ALTERNATIVE_BASIS_CHAINING)
          multiGpuDoubleAlternativeBasisChaining(n, maxSubproblemSize, 
                                                    A_a.getDataPointer(),
                                                    B_a.getDataPointer(),
                                                    C_a.getDataPointer(),
                                                    cpuScratch, &A_ds[0],
                                                    &B_ds[0], &C_ds[0],
                                                    &scratch_ds[0], 
                                                    lanes*GPU_COUNT, 
                                                    auxmats);
        else
          assert(false && "invalid mode");
        pushTime();
        cout << (mode == Mode::STRASSEN_WINOGRAD ? "multiGpuDoubleStrassenWinograd " :
                 mode == Mode::ALTERNATIVE_BASIS_SELF_INVERSE ? "multiGpuDoubleAlternativeBasisSelfInverse " :
                 mode == Mode::ALTERNATIVE_BASIS_CHAINING ? "multiGpuDoubleAlternativeBasisChaining " :
                 "ERROR") << n << "x" << n
             << " (sub: " << maxSubproblemSize << "x" << maxSubproblemSize 
             << ") with " << lanes << " lane(s) and " << auxmats
             << " auxmat(s) took " << nanosToString(getDiffNanos()) << endl;
      }

      if (mode == Mode::ALTERNATIVE_BASIS_SELF_INVERSE) {
        pushTime();
        inverseChangeOfBasisSelfInverseInPlace(C_a.getDataPointer(), n);
        pushTime();
        cout << "inverse change of basis " << n << "x" << n << " took " << nanosToString(getDiffNanos()) << endl;
      }
      else if (mode == Mode::ALTERNATIVE_BASIS_CHAINING) {
        pushTime();
        inverseChangeOfBasisChainingInPlace(C_a.getDataPointer(), n);
        pushTime();
        cout << "inverse change of basis " << n << "x" << n << " took " << nanosToString(getDiffNanos()) << endl;
      }

      pushTime();
      dearrangeDataCpuWords(C_a.getDataPointer(), C.getDataPointer(), n*n);
      pushTime();
      cout << "data dearrangement " << n << "x" << n << " took " << nanosToString(getDiffNanos()) << endl;     
    }
    else {
      assert(false && "INVALID MODE");
    }

    const static int64_t MAX_SIDE_LENGTH_FOR_FULL_TEST = 1024L;
    
    if (n <= MAX_SIDE_LENGTH_FOR_FULL_TEST) {
      for (index_t i = 0; i < n; ++i)
        for (index_t j = 0; j < n; ++j)
          if (mode == Mode::CUBIC || mode == Mode::STRASSEN_WINOGRAD ||
              mode == Mode::ALTERNATIVE_BASIS_SELF_INVERSE ||
              mode == Mode::ALTERNATIVE_BASIS_CHAINING)
            assert(A.dotOneBit(B,i,j) == C.get(i,j));
          else if (mode == Mode::BOOLEAN)
            assert(A.booleanDotOneBit(B,i,j) == C.get(i,j));
          else
            assert(false && "Invalid mode");
    }
    else {
      std::mt19937 rng(time(nullptr));
      std::uniform_int_distribution<index_t> dist(0,n-1);
      const int64_t MAX_SQ = MAX_SIDE_LENGTH_FOR_FULL_TEST * 
        MAX_SIDE_LENGTH_FOR_FULL_TEST;
      for (int64_t k = 0; k < MAX_SQ; ++k) {
        index_t i = dist(rng);
        index_t j = dist(rng);

        if (mode == Mode::CUBIC || mode == Mode::STRASSEN_WINOGRAD || 
            mode == Mode::ALTERNATIVE_BASIS_SELF_INVERSE ||
            mode == Mode::ALTERNATIVE_BASIS_CHAINING)
          assert(A.dotOneBit(B,i,j) == C.get(i,j));
        else if (mode == Mode::BOOLEAN)
          assert(A.booleanDotOneBit(B,i,j) == C.get(i,j));
        else
          assert(false && "Invalid mode");
      }
    }

    if (mode == Mode::CUBIC) {
      if (n <= maxSubproblemSize) 
        cout << "cuda cubic multiplication " << n << "x" << n << " ok" << endl;
      else
        cout << "multigpu cubic " << n << "x" << n << " ok" << endl;
    }
    else if (mode == Mode::BOOLEAN) {
      if (n <= maxSubproblemSize) 
        cout << "cuda boolean multiplication " << n << "x" << n << " ok" << endl;
      else
        cout << "multigpu boolean " << n << "x" << n << " ok" << endl;
    }
    else if (mode == Mode::STRASSEN_WINOGRAD) {
      if (n <= maxSubproblemSize) 
        cout << "cudaRecursiveDoubleStrassenWinograd " << n << "x" << n
             << " ok" << endl;
      else
        cout << "multiGpuDoubleStrassenWinograd " << n << "x" << n
             << " with " << lanes << " lanes and " << auxmats << " auxmats ok"
             << endl;
    }
    else if (mode == Mode::ALTERNATIVE_BASIS_SELF_INVERSE) {
      if (n <= maxSubproblemSize) 
        cout << "cudaRecursiveDoubleAlternativeBasisSelfInverse " << n 
             << "x" << n << " ok" << endl;
      else
        cout << "multiGpuRecursiveDoubleAlternativeBasisSelfInverse " << n 
             << "x" << n << " with " << lanes << " lanes and " << auxmats 
             << " auxmats ok" << endl;
    }
    else if (mode == Mode::ALTERNATIVE_BASIS_CHAINING) {
      if (n <= maxSubproblemSize) 
        cout << "cudaRecursiveDoubleAlternativeBasisChaining " << n 
             << "x" << n << " ok" << endl;
      else
        cout << "multiGpuRecursiveDoubleAlternativeBasisChaining " << n 
             << "x" << n << " with " << lanes << " lanes and " << auxmats 
             << " auxmats ok" << endl;
    }
    else {
      assert(false && "invalid mode");
    }
  }

  
  free(cpuScratch);
  for (auto p : gpuMemory)
    gpuFree(p);
}



static void evaluateChangeOfBasis(int64_t maxInstanceSize) {
  for (int64_t n = 64; n <= maxInstanceSize; n <<= 1) {
    pushTime();
    BinaryMatrix A = BinaryMatrix::random(n);
    pushTime();
    cout << "data generation " << n << "x" << n << " took "
         << nanosToString(getDiffNanos()) << endl;

    for (int rep = 0; rep < NREPEATS; ++rep) {
      pushTime();
      changeOfBasisSelfInverseInPlaceAvxParallel(A.getDataPointer(),n);
      pushTime();
      auto nanos = getDiffNanos();
      cout << "changeOfBasisSelfInverseInPlaceAvxParallel " << n << "x" << n << " took "
             << nanosToString(nanos) << endl;

      pushTime();
      inverseChangeOfBasisSelfInverseInPlaceAvxParallel(A.getDataPointer(),n);
      pushTime();
      nanos = getDiffNanos();
      cout << "inverseChangeOfBasisSelfInverseInPlaceAvxParallel " << n << "x" << n << " took "
             << nanosToString(nanos) << endl;

      pushTime();
      changeOfBasisChainingLeftInPlaceAvxParallel(A.getDataPointer(),n);
      pushTime();
      nanos = getDiffNanos();
      cout << "changeOfBasisChainingLeftInPlaceAvxParallel " << n << "x" << n << " took "
             << nanosToString(nanos) << endl;

      pushTime();
      changeOfBasisChainingRightInPlaceAvxParallel(A.getDataPointer(),n);
      pushTime();
      nanos = getDiffNanos();
      cout << "changeOfBasisChainingRightInPlaceAvxParallel " << n << "x" << n << " took "
             << nanosToString(nanos) << endl;

      pushTime();
      inverseChangeOfBasisChainingInPlaceAvxParallel(A.getDataPointer(),n);
      pushTime();
      nanos = getDiffNanos();
      cout << "inverseChangeOfBasisChainingInPlaceAvxParallel " << n << "x" << n << " took "
             << nanosToString(nanos) << endl;
    }
  }
}



static void evaluateTranspose(int64_t maxInstanceSize) {
  for (int64_t n = 64; n <= maxInstanceSize; n <<= 1) {
    pushTime();
    BinaryMatrix A = BinaryMatrix::random(n);
    pushTime();
    cout << "data generation " << n << "x" << n << " took "
         << nanosToString(getDiffNanos()) << endl;

    for (int rep = 0; rep < NREPEATS; ++rep) {
      pushTime();
      transposeInplaceArranged(A.getDataPointer(),n);
      pushTime();
      auto nanos = getDiffNanos();
      cout << "transposeInplaceArranged " << n << "x" << n << " took "
             << nanosToString(nanos) << endl;
    }
  }
}



static void evaluate(Mode mode, int64_t maxInstanceSize, int64_t maxSubproblemSize,
                     int lanes, int auxmats) {
  if (mode == Mode::TRANSPOSE) {
    evaluateTranspose(maxInstanceSize);
    return;
  }

  if (mode == Mode::CHANGE_OF_BASIS) {
    evaluateChangeOfBasis(maxInstanceSize);
    return;
  }
    
  
  pushTime();
  uint4* cpuScratch = allocateAndZeroCpuScratch(maxSubproblemSize, lanes, auxmats);
  pushTime();
  cout << "cpu scratch allocation and zeroing took " << nanosToString(getDiffNanos()) << endl;

  pushTime();
  int64_t gpuMemorySize = computeGpuMemorySize(maxSubproblemSize, mode);
  uint4* gpuMemory[GPU_COUNT];
  uint4* A_ds[GPU_COUNT];
  uint4* B_ds[GPU_COUNT];
  uint4* C_ds[GPU_COUNT];
  uint4* scratch_ds[GPU_COUNT];
  for (int i = 0; i < GPU_COUNT; ++i) {
    cudaSetDevice(i);
    gpuMemory[i] = gpuAllocate(gpuMemorySize);
  }
  pushTime();
  cout << "gpu memory allocation took " << nanosToString(getDiffNanos()) << endl;

  for (int64_t n = 64; n <= maxInstanceSize; n <<= 1) {
    int q, d;
    if (n <= maxSubproblemSize) {
      q = 0;
      d = intlog2(n)-6;
    }
    else {
      d = intlog2(maxSubproblemSize)-6;
      q = intlog2(n)-6-d;
    }
    assert((1<<q)*(1<<d)*64 == n);

    int64_t nWordsGpu = (1L<<2*d)*32L;
    for (int i = 0; i < GPU_COUNT; ++i) {
      A_ds[i] = gpuMemory[i];
      B_ds[i] = gpuMemory[i]+nWordsGpu;
      C_ds[i] = gpuMemory[i]+2*nWordsGpu;
      scratch_ds[i] = gpuMemory[i]+3*nWordsGpu;
    }

    pushTime();
    BinaryMatrix A = (mode == Mode::BOOLEAN) ?
      BinaryMatrix::booleanRandom(n) : BinaryMatrix::random(n);
    BinaryMatrix B = (mode == Mode::BOOLEAN) ?
      BinaryMatrix::booleanRandom(n) : BinaryMatrix::random(n);
    BinaryMatrix C(n,n);
    pushTime();
    cout << "data generation " << n << "x" << n << " took "
         << nanosToString(getDiffNanos()) << endl;

    for (int rep = 0; rep < NREPEATS; ++rep) {
      if (n <= maxSubproblemSize) {
        cudaSetDevice(0);
        pushTime();
        gpuUpload((1<<2*d)*32, A.getDataPointer(), A_ds[0]);
        gpuUpload((1<<2*d)*32, B.getDataPointer(), B_ds[0]);
        pushTime();
        cout << "data upload " << n << "x" << n << " took "
             << nanosToString(getDiffNanos()) << endl;

        pushTime();
        if (mode == Mode::CUBIC)
          cudaCubicMultiplication(A_ds[0], B_ds[0], C_ds[0], nullptr, n);
        else if (mode == Mode::BOOLEAN)
          cudaBooleanCubicMultiplication(A_ds[0], B_ds[0], C_ds[0], nullptr, n);
        else if (mode == Mode::STRASSEN_WINOGRAD)
          cudaRecursiveDoubleStrassenWinograd(A_ds[0], B_ds[0], C_ds[0],
                                              scratch_ds[0], n);
        else if (mode == Mode::ALTERNATIVE_BASIS_SELF_INVERSE)
          cudaRecursiveDoubleAlternativeBasisSelfInverse(A_ds[0], B_ds[0], 
                                                         C_ds[0], scratch_ds[0],
                                                         n);
        else if (mode == Mode::ALTERNATIVE_BASIS_CHAINING)
          cudaRecursiveDoubleAlternativeBasisChaining(A_ds[0], B_ds[0], 
                                                         C_ds[0], scratch_ds[0],
                                                         n);
        else
          assert(false && "INVALID MODE");
        pushTime();

        auto nanos = getDiffNanos();
        cout <<
          (mode == Mode::CUBIC ? "cudaCubicMultiplication" :
           mode == Mode::BOOLEAN ? "cudaBooleanMultiplication" :
           mode == Mode::STRASSEN_WINOGRAD ? "cudaRecursiveDoubleStrassenWinograd" :
           mode == Mode::ALTERNATIVE_BASIS_SELF_INVERSE ? "cudaRecursiveDoubleAlternativeBasisSelfInverse" :
           mode == Mode::ALTERNATIVE_BASIS_CHAINING ? "cudaRecursiveDoubleAlternativeBasisChaining" :
           "ERROR") << " " << n << "x" << n << " took " <<
          nanosToString(nanos) << " (" << computeEffectiveTbops(n,nanos)
             << " effective Tbops)" << endl;
      }
      else {
        pushTime();
        if (mode == Mode::CUBIC)
          multiGpuCubic(A.getDataPointer(), B.getDataPointer(), C.getDataPointer(),
                        n, maxSubproblemSize, A_ds, B_ds, C_ds);
        else if (mode == Mode::BOOLEAN)
          multiGpuBooleanCubic(A.getDataPointer(), B.getDataPointer(), C.getDataPointer(),
                               n, maxSubproblemSize, A_ds, B_ds, C_ds);
        else if (mode == Mode::STRASSEN_WINOGRAD)
          multiGpuDoubleStrassenWinograd(n, maxSubproblemSize, 
                                         A.getDataPointer(), B.getDataPointer(),
                                         C.getDataPointer(), cpuScratch, A_ds,
                                         B_ds, C_ds, scratch_ds, 
                                         lanes*GPU_COUNT, auxmats);
        else if (mode == Mode::ALTERNATIVE_BASIS_SELF_INVERSE)
          multiGpuDoubleAlternativeBasisSelfInverse(n, maxSubproblemSize, 
                                                    A.getDataPointer(), 
                                                    B.getDataPointer(),
                                                    C.getDataPointer(), 
                                                    cpuScratch, A_ds, B_ds, 
                                                    C_ds, scratch_ds, 
                                                    lanes*GPU_COUNT, auxmats);
        else if (mode == Mode::ALTERNATIVE_BASIS_CHAINING)
          multiGpuDoubleAlternativeBasisChaining(n, maxSubproblemSize, 
                                                    A.getDataPointer(), 
                                                    B.getDataPointer(),
                                                    C.getDataPointer(), 
                                                    cpuScratch, A_ds, B_ds, 
                                                    C_ds, scratch_ds, 
                                                    lanes*GPU_COUNT, auxmats);
        else
          assert(false && "Invalid mode");
        pushTime();
        
        auto nanos = getDiffNanos();
        cout <<
          (mode == Mode::CUBIC ? "multiGpuCubic" :
           mode == Mode::BOOLEAN ? "multiGpuBooleanCubic" :
           mode == Mode::STRASSEN_WINOGRAD ? "multiGpuDoubleStrassenWinograd" :
           mode == Mode::ALTERNATIVE_BASIS_SELF_INVERSE ? "multiGpuDoubleAlternativeBasisSelfInverse" : 
           mode == Mode::ALTERNATIVE_BASIS_CHAINING ? "multiGpuDoubleAlternativeBasisChaining" : 
           "ERROR") << " " << n << "x" << n << " (sub: "
             << maxSubproblemSize << "x" << maxSubproblemSize << ") took "
             << nanosToString(nanos) << " (" << computeEffectiveTbops(n,nanos)
             << " effective Tbops)" << endl;
      }
    }
  }
  

  free(cpuScratch);
  for (auto p : gpuMemory)
    gpuFree(p);
}



static string timestamp() {
  std::ostringstream oss;
  time_t t = time(nullptr);
  tm tm = *localtime(&t);
  oss << std::put_time(&tm, "%c %Z");
  return oss.str();
}



static string getHostname() {
  char hostname[256];
  struct utsname undata;
  uname(&undata);
  strcpy(hostname, undata.nodename);
  return hostname;
}




static void printHelp(char* exeName) {
  cout << "usage: " << exeName
       << " <test|evaluate> "
       << "<transpose|change-of-basis|cubic|boolean|"
       << "strassen-winograd|absinv|abschain> <n> <m> <l> <a>" << endl
       << "  where" << endl
       << endl
       << " test:     perform validity test" << endl
       << " evaluate: evaluate performance" << endl
       << endl
       << " transpose:         64*64 submatrix transpose only (no mm)" << endl
       << " change-of-basis:   change of basis only (no mm)" << endl
       << " cubic:             binary cubic algorithm" << endl
       << " boolean:           boolean cubic algorithm" << endl
       << " strassen-winograd: binary Strassen-Winograd algorithm" << endl
       << " absinv:            alternative-basis, self-inverse mm" << endl
       << " abchain:           alternative-basis, chaining mm" << endl
       << endl
       << " n: maximum instance size (integer, power of 2)" << endl
       << " m: single-gpu instance size (integer, power of 2)" << endl
       << " l: number of lanes (integer)" << endl
       << " a: number of auxiliary matrices (integer)" << endl;
}



int main(int argc, char* argv[]) {
  // Visible devices are stored in environment variable CUDA_VISIBLE_DEVICES
  // Format: (e.g.) CUDA_VISIBLE_DEVICES=0,1,2,3,4

  cerr << "start time: " << timestamp() << endl;

  cerr << "host: " << getHostname() << endl;
  
  CUDA_WRAP(cudaGetDeviceCount(&GPU_COUNT));
  cerr << GPU_COUNT << " GPU(s) detected" << endl;

  cudaDeviceProp prop;
  CUDA_WRAP(cudaGetDeviceProperties(&prop, 0));
  fprintf(stderr, "Compute capability %d.%d detected\n", prop.major,
          prop.minor);

  int uvaEnabled;
  cuDeviceGetAttribute(&uvaEnabled, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, 0);
           
  if (!uvaEnabled) {
    cerr << "ERROR: UVA is disabled" << endl;
  }


  enableTimings();

  bool argsValid = true;
  Action action = Action::NIL;
  Mode mode = Mode::NIL;
  long maxInstanceSize = -1;
  long maxSingleGpuSize = -1;
  int numberOfLanes = -1;
  int numberOfAuxMats = -1;
  
  if (argc == 7) {
    if (string(argv[1]) == "test")
      action = Action::TEST;
    else if (string(argv[1]) == "evaluate")
      action = Action::EVALUATE;
    else {
      cerr << "Invalid action `" << argv[1] << "' specified!" << endl;
      argsValid = false;
    }

    if (string(argv[2]) == "cubic")
      mode = Mode::CUBIC;
    else if (string(argv[2]) == "boolean")
      mode = Mode::BOOLEAN;
    else if (string(argv[2]) == "strassen-winograd")
      mode = Mode::STRASSEN_WINOGRAD;
    else if (string(argv[2]) == "absinv")
      mode = Mode::ALTERNATIVE_BASIS_SELF_INVERSE;
    else if (string(argv[2]) == "abchain")
      mode = Mode::ALTERNATIVE_BASIS_CHAINING;
    else if (string(argv[2]) == "transpose")
      mode = Mode::TRANSPOSE;
    else if (string(argv[2]) == "change-of-basis")
      mode = Mode::CHANGE_OF_BASIS;
    else {
      cerr << "Invalid mode `" << argv[2] << "' specified!" << endl;
      argsValid = false;
    }

    if (sscanf(argv[3], "%ld", &maxInstanceSize) != 1) {
      cerr << "Non-integer maximum instance size `" << argv[3] << "' specified!" << endl;
      argsValid = false;
    }
    else if (!isPowerOfTwo(maxInstanceSize)) {
      cerr << "Maximum instance size `" << argv[3] << "' is not a power of 2!" << endl;
      argsValid = false;
    }

    if (sscanf(argv[4], "%ld", &maxSingleGpuSize) != 1) {
      cerr << "Non-integer single GPU instance size `" << argv[4] << "' specified!" << endl;
      argsValid = false;
    }
    else if (!isPowerOfTwo(maxSingleGpuSize)) {
      cerr << "Single GPU instance size `" << argv[4] << "' is not a power of 2!" << endl;
      argsValid = false;
    }

    if (sscanf(argv[5], "%d", &numberOfLanes) != 1) {
      cerr << "Non-integer number `" << argv[5] << "' of lanes given!" << endl;
      argsValid = false;
    }
    else if (numberOfLanes < 1 || numberOfLanes > (MAXIMUM_NUMBER_OF_LANES/GPU_COUNT)) {
      cerr << "Invalid number `" << numberOfLanes << "' of lanes given!" << endl;
      argsValid = false;
    }

    if (sscanf(argv[6], "%d", &numberOfAuxMats) != 1) {
      cerr << "Non-integer number `" << argv[6] << "' of auxiliary matrices given!" << endl;
      argsValid = false;
    }
    else if (numberOfAuxMats < 1 || numberOfAuxMats > MAXIMUM_NUMBER_OF_AUXILIARY_MATRICES) {
      cerr << "Invalid number `" << numberOfAuxMats << "' of auxiliary matrices given!" << endl;
      argsValid = false;
    }

  }
  else {
    argsValid = false;
    printHelp(argv[0]);
  }

  if (argsValid) {
    if (action == Action::TEST)
      test(mode, maxInstanceSize, maxSingleGpuSize, numberOfLanes, numberOfAuxMats);
    else if (action == Action::EVALUATE)
      evaluate(mode, maxInstanceSize, maxSingleGpuSize, numberOfLanes, numberOfAuxMats);
  }
  
  disableTimings();

  cerr << "end time: " << timestamp() << endl;

  return EXIT_SUCCESS;
}
