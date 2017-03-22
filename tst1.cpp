//#define NO_MKL

#include <stdint.h>
#include <string.h>
#include <vector>
#include <random>
#include <functional>           // for std::bind
#include <algorithm>
#ifndef NO_MKL
//#include "mkl_cblas.h"
#include <openblas/cblas.h>
#endif

void ref_noncblas_sgemm(
 int M, int N, int K, 
 float alpha, 
 const float *A, int lda, 
 const float *B, int ldb,
 float beta, 
 float *C, int ldc);

void scalar_noncblas_sgemm(
 int M, int N, int K, 
 float alpha, 
 const float *A, int lda, 
 const float *B, int ldb,
 float beta, 
 float *C, int ldc);

void avx128_noncblas_sgemm(
 int M, int N, int K, 
 float alpha, 
 const float *A, int lda, 
 const float *B, int ldb,
 float beta, 
 float *C, int ldc);

void fma128_noncblas_sgemm(
 int M, int N, int K, 
 float alpha, 
 const float *A, int lda, 
 const float *B, int ldb,
 float beta, 
 float *C, int ldc);

void avx256_noncblas_sgemm(
 int M, int N, int K, 
 float alpha, 
 const float *A, int lda, 
 const float *B, int ldb,
 float beta, 
 float *C, int ldc);

void fma256_noncblas_sgemm(
 int M, int N, int K, 
 float alpha, 
 const float *A, int lda, 
 const float *B, int ldb,
 float beta, 
 float *C, int ldc);

#ifndef NO_MKL
// adapt MKL cblas_sgemm to my 'noncblas' calling order
static void MKL_noncblas_sgemm(
 int M, int N, int K, 
 float alpha, 
 const float *A, int lda, 
 const float *B, int ldb,
 float beta, 
 float *C, int ldc)
{
  cblas_sgemm(
    CblasRowMajor, CblasNoTrans, CblasNoTrans
    , M, N, K
    , alpha
    , A, lda
    , B, ldb
    , beta
    , C, ldc);
}
#endif

static void test_noncblas_sgemm(
 int M, int N, int K, 
 float alpha, 
 const float *A, int lda, 
 const float *B, int ldb,
 float beta, 
 float *C, int ldc,
 int nIter,
 const float *srcC,
void (*uut)(
 int M, int N, int K, 
 float alpha, 
 const float *A, int lda, 
 const float *B, int ldb,
 float beta, 
 float *C, int ldc)
 );


bool IsFMA3Supported()
{
#ifdef __GNUC__
  __builtin_cpu_init();
  // GCC has broken CPU detection support - no way to ask for presence of FMA3. 
  // So, in the mean time, I am asking for AVX2 that, in the mean time, 
  // happens to always co-exist with FMA3. The test is not future proof.
  // I wonder why gcc decided to make things so messy
  return __builtin_cpu_supports("avx2");
#else
  int cpuInfo[4];
  __cpuid(cpuInfo, 1); //  EAX, EBX, ECX, and EDX
  //printf("CPU features: %08x:%08x:%08x:%08x\n", cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
  return (cpuInfo[2] & (1 << 12)) != 0; // check for a presence of FMA3 extension
#endif
}

int main(int argz, char** argv)
{
  int M = 128;
  int N = 361;
  int K = 1152;
  float alpha = 1;
  float beta  = 0;
  int lda = 0;
  int ldb = 0;
  int ldc = 0;

  for (int arg_i = 1; arg_i < argz; ++arg_i) {
    char* arg = argv[arg_i];
    static const char* prefTab[] = {
      "alpha", "beta", "M", "N", "K", "lda", "ldb", "ldc"
    };
    for (int pref_i = 0; pref_i < sizeof(prefTab)/sizeof(prefTab[0]); ++pref_i) {
      const char* pref = prefTab[pref_i];
      size_t preflen = strlen(pref);
      if (strncasecmp(pref, arg, preflen)==0 && arg[preflen]=='=') {
        if (pref_i < 2) {
          // floating point arguments
          char* endp;
          double val = strtod(&arg[preflen+1], &endp);
          if (endp==&arg[preflen+1]) {
            fprintf(stderr, "Bad parameter '%s'. '%s' is not a number.\n", arg, &arg[preflen+1]);
            return 1;
          }
          switch (pref_i) {
            case 0: alpha = float(val); break;
            case 1: beta = float(val);  break;
            default:break;
          }
        } else {
          // integer arguments
          char* endp;
          long val = strtol(&arg[preflen+1], &endp, 0);
          if (endp==&arg[preflen+1] || val <= 0) {
            fprintf(stderr, "Bad parameter '%s'. '%s' is not a positive number.\n", arg, &arg[preflen+1]);
            return 1;
          }
          switch (pref_i) {
            case 2: M = val; break;
            case 3: N = val; break;
            case 4: K = val; break;
            case 5: lda = val; break;
            case 6: ldb = val; break;
            case 7: ldc = val; break;
            default:break;
          }
        }
        goto next_arg;
      }
    }
    next_arg:;
  }

  if (lda == 0) lda = K;
  if (ldb == 0) ldb = N;
  if (ldc == 0) ldc = N;

  if (lda < K) {
    fprintf(stderr, "Bad parameter lda=%d. Should be greater or equal to K=%d\n", lda, K);
    return 1;
  }
  if (ldb < N) {
    fprintf(stderr, "Bad parameter ldb=%d. Should be greater or equal to N=%d\n", ldb, N);
    return 1;
  }
  if (ldc < N) {
    fprintf(stderr, "Bad parameter ldc=%d. Should be greater or equal to N=%d\n", ldc, N);
    return 1;
  }

  printf("Running SGEMM with M=%d, N=%d, K=%d, alpha=%f, lda=%d, ldb=%d, beta=%f, ldc=%d\n",
    M, N, K, alpha, lda, ldb, beta, ldc);



  const int nIter = 11;

  std::vector<float> A(nIter*M*lda);
  std::vector<float> B(nIter*K*ldb);
  std::vector<float> C(nIter*M*ldc);
  std::vector<float> srcC(nIter*M*ldc);

  std::mt19937_64 rndGen;
  std::uniform_real_distribution<float> rndDistr(-1.0f, 1.0f);
  auto rndFunc = std::bind ( rndDistr, std::ref(rndGen) );
  for (int i = 0; i < nIter*M*lda; ++i)
    A[i] = rndFunc();
  for (int i = 0; i < nIter*K*ldb; ++i)
    B[i] = rndFunc();
  for (int i = 0; i < nIter*M*ldc; ++i)
    srcC[i] = rndFunc();

  bool hasFma3 = IsFMA3Supported();

#ifndef NO_MKL
  printf("Testing Intel MKL...\n");
  test_noncblas_sgemm(M, N, K, alpha
    , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
    , nIter, &srcC.at(0),
    MKL_noncblas_sgemm);
#endif

  printf("Testing my scalar hack...\n");
  test_noncblas_sgemm(M, N, K, alpha
    , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
    , nIter, &srcC.at(0),
    scalar_noncblas_sgemm);

  printf("Testing my 128-bit AVX hack...\n");
  test_noncblas_sgemm(M, N, K, alpha
    , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
    , nIter, &srcC.at(0),
    avx128_noncblas_sgemm);

  if (hasFma3) {
    printf("Testing my 128-bit FMA hack...\n");
    test_noncblas_sgemm(M, N, K, alpha
      , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
      , nIter, &srcC.at(0),
      fma128_noncblas_sgemm);
  }

  printf("Testing my 256-bit AVX hack...\n");
  test_noncblas_sgemm(M, N, K, alpha
    , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
    , nIter, &srcC.at(0),
    avx256_noncblas_sgemm);

  if (hasFma3) {
    printf("Testing my 256-bit FMA hack...\n");
    test_noncblas_sgemm(M, N, K, alpha
      , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
      , nIter, &srcC.at(0),
      fma256_noncblas_sgemm);
  }

  return 0;
}

static void cmp_results(
 int M, int N,
 const float *ref,
 const float *res,
 int ld)
{
  double maxErr = 0;
  double s2Err = 0;
  double s1Ref = 0;
  double s2Ref = 0;
  int maxI = 0;
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      double refV = ref[m*ld+n];
      double resV = res[m*ld+n];
      double err  = resV - refV;
      if (maxErr < fabs(err)) {
        maxErr = fabs(err);
        maxI = m*ld+n;
      }
      s2Err += err*err;
      s1Ref += refV;
      s2Ref += refV*refV;
    }
  }
  double stdErr = sqrt(s2Err / (M*N));
  double stdRef = sqrt(s2Ref*(M*N) - s1Ref*s1Ref)/((M*N));
  printf("%.3e/%.3e=%.3e. %.3e at [%3d,%3d] %18.10e vs %18.10e %s\n"
    , stdErr, stdRef, stdErr/stdRef
    , maxErr, maxI/ld, maxI%ld
    , double(ref[maxI]), double(res[maxI])
    , maxErr > stdRef*1e-5 ? "FAIL !!!" : (maxErr > stdRef*3e-5 || stdErr > stdRef*1e-6 ? "Sucks !" : "")
    );
}

static void test_noncblas_sgemm(
 int M, int N, int K, 
 float alpha, 
 const float *A, int lda, 
 const float *B, int ldb,
 float beta, 
 float *C, int ldc,
 int nIter,
 const float *srcC,
void (*uut)(
 int M, int N, int K, 
 float alpha, 
 const float *A, int lda, 
 const float *B, int ldb,
 float beta, 
 float *C, int ldc)
 )
{
  for (int i = 0; i < nIter*M*ldc; ++i)
    C[i] = srcC[i];

  std::vector<uint64_t> dt(nIter);
  for (int it = 0; it < nIter; ++it) {
    uint64_t t0 = __rdtsc();
    uut(
      M, N, K
      , alpha
      , &A[it*M*lda], lda
      , &B[it*K*ldb], ldb
      , beta
      , &C[it*M*ldc], ldc
      );
    uint64_t t1 = __rdtsc();
    dt[it] = t1-t0;
  }
  for (int it = 0; it < nIter; ++it)
    printf(" %.0f", double(dt[it]));
  std::nth_element(dt.begin(), dt.begin()+nIter/2, dt.begin()+nIter);
  printf(": med %I64u. %.3f FLOP/clk\n", dt[nIter/2], double(M)*N*K*2/double(dt[nIter/2]));
#define VERIFY
#ifdef VERIFY
  std::vector<float> refC(M*ldc);
  for (int it = 0; it < nIter; ++it) {
    for (int i = 0; i < M*ldc; ++i)
      refC[i] = srcC[it*M*ldc+i];
    ref_noncblas_sgemm(
      M, N, K
      , alpha
      , &A[it*M*lda], lda
      , &B[it*K*ldb], ldb
      , beta
      , &refC.at(0), ldc
      );
    cmp_results(
      M, N
      , &refC.at(0)
      , &C[it*M*ldc]
      , ldc
      );
  }
#endif
}
