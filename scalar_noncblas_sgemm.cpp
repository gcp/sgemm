#include <string.h>

const int bb_nCols = 52;
const int bb_nRows = 60;

struct noncblas_sgemm_prm_t {
  int   M;
  int   lda;
  int   ldc;
  float alpha;
  float bb[bb_nCols*bb_nRows];
};

static void scalar_noncblas_sgemm_core(
 const noncblas_sgemm_prm_t* pPrm,
 const float *A,
 float *C)
{
  int lda = pPrm->lda;
  int ldc = pPrm->ldc;
  int m;
  for (m = 0; m < pPrm->M-1; A += lda*2, C += ldc*2, m += 2) {
    for (int n = 0; n < bb_nCols; n += 2) {
      const float *Bcol = &pPrm->bb[n];
      float acc00 = 0;
      float acc10 = 0;
      float acc01 = 0;
      float acc11 = 0;
      for (int k = 0; k < bb_nRows; ++k) {
        acc00  += A[k] * Bcol[0]; 
        acc10  += A[k] * Bcol[1]; 
        acc01  += A[k+lda] * Bcol[0]; 
        acc11  += A[k+lda] * Bcol[1]; 
        Bcol += bb_nCols;
      }
      C[n+0] += acc00*pPrm->alpha;
      C[n+1] += acc10*pPrm->alpha;
      C[ldc+n+0] += acc01*pPrm->alpha;
      C[ldc+n+1] += acc11*pPrm->alpha;
    }
  }
  if (m < pPrm->M) {
    for (int n = 0; n < bb_nCols; n += 2) {
      const float *Bcol = &pPrm->bb[n];
      float acc00 = 0;
      float acc10 = 0;
      for (int k = 0; k < bb_nRows; ++k) {
        acc00  += A[k] * Bcol[0]; 
        acc10  += A[k] * Bcol[1]; 
        Bcol += bb_nCols;
      }
      C[n+0] += acc00*pPrm->alpha;
      C[n+1] += acc10*pPrm->alpha;
    }
  }
}

static void scalar_noncblas_sgemm_core_bottomRows(
 const noncblas_sgemm_prm_t* pPrm,
 const float *A,
 float *C,
 int nRows)
{
  int lda = pPrm->lda;
  int ldc = pPrm->ldc;
  int m;
  for (m = 0; m < pPrm->M-1; A += lda*2, C += ldc*2, m += 2) {
    for (int n = 0; n < bb_nCols; n += 2) {
      const float *Bcol = &pPrm->bb[n];
      float acc00 = 0;
      float acc10 = 0;
      float acc01 = 0;
      float acc11 = 0;
      for (int k = 0; k < nRows; ++k) {
        acc00  += A[k] * Bcol[0]; 
        acc10  += A[k] * Bcol[1]; 
        acc01  += A[k+lda] * Bcol[0]; 
        acc11  += A[k+lda] * Bcol[1]; 
        Bcol += bb_nCols;
      }
      C[n+0] += acc00*pPrm->alpha;
      C[n+1] += acc10*pPrm->alpha;
      C[ldc+n+0] += acc01*pPrm->alpha;
      C[ldc+n+1] += acc11*pPrm->alpha;
    }
  }
  if (m < pPrm->M) {
    for (int n = 0; n < bb_nCols; n += 2) {
      const float *Bcol = &pPrm->bb[n];
      float acc00 = 0;
      float acc10 = 0;
      for (int k = 0; k < nRows; ++k) {
        acc00  += A[k] * Bcol[0]; 
        acc10  += A[k] * Bcol[1]; 
        Bcol += bb_nCols;
      }
      C[n+0] += acc00*pPrm->alpha;
      C[n+1] += acc10*pPrm->alpha;
    }
  }
}

static void scalar_noncblas_sgemm_core_rightmostColumns(
 const noncblas_sgemm_prm_t* pPrm,
 const float *A,
 float *C,
 int nCols, 
 int nRows)
{
  int lda = pPrm->lda;
  int ldc = pPrm->ldc;
  int m;
  for (m = 0; m < pPrm->M-1; A += lda*2, C += ldc*2, m += 2) {
    int n;
    for (n = 0; n < nCols-1; n += 2) {
      const float *Bcol = &pPrm->bb[n];
      float acc00 = 0;
      float acc10 = 0;
      float acc01 = 0;
      float acc11 = 0;
      for (int k = 0; k < nRows; ++k) {
        acc00  += A[k] * Bcol[0]; 
        acc10  += A[k] * Bcol[1]; 
        acc01  += A[k+lda] * Bcol[0]; 
        acc11  += A[k+lda] * Bcol[1]; 
        Bcol += bb_nCols;
      }
      C[n+0] += acc00*pPrm->alpha;
      C[n+1] += acc10*pPrm->alpha;
      C[ldc+n+0] += acc01*pPrm->alpha;
      C[ldc+n+1] += acc11*pPrm->alpha;
    }
    if (n < nCols) {
      const float *Bcol = &pPrm->bb[n];
      float acc00 = 0;
      float acc01 = 0;
      for (int k = 0; k < nRows; ++k) {
        acc00  += A[k] * Bcol[0]; 
        acc01  += A[k+lda] * Bcol[0]; 
        Bcol += bb_nCols;
      }
      C[n+0] += acc00*pPrm->alpha;
      C[ldc+n+0] += acc01*pPrm->alpha;
    }
  }
  if (m < pPrm->M) {
    int n;
    for (n = 0; n < nCols-1; n += 2) {
      const float *Bcol = &pPrm->bb[n];
      float acc00 = 0;
      float acc10 = 0;
      for (int k = 0; k < nRows; ++k) {
        acc00  += A[k] * Bcol[0]; 
        acc10  += A[k] * Bcol[1]; 
        Bcol += bb_nCols;
      }
      C[n+0] += acc00*pPrm->alpha;
      C[n+1] += acc10*pPrm->alpha;
    }
    if (n < nCols) {
      const float *Bcol = &pPrm->bb[n];
      float acc00 = 0;
      for (int k = 0; k < nRows; ++k) {
        acc00  += A[k] * Bcol[0]; 
        Bcol += bb_nCols;
      }
      C[n+0] += acc00*pPrm->alpha;
    }
  }
}



static void scalar_noncblas_sgemm_multC(
 int M, int N,
 float beta, 
 float *C, int ldc)
{
  if (beta != 0) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n)
        C[n] *= beta;
      C += ldc;
    }
  } else {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n)
        C[n] = 0;
      C += ldc;
    }
  }
}

void scalar_noncblas_sgemm(
 int M, int N, int K, 
 float alpha, 
 const float *A, int lda, 
 const float *B, int ldb,
 float beta, 
 float *C, int ldc)
{
  scalar_noncblas_sgemm_multC(M, N, beta, C, ldc);

  noncblas_sgemm_prm_t prm;
  prm.M      = M;
  prm.lda   = lda;
  prm.ldc   = ldc;
  prm.alpha = alpha;

  int n_Rsteps = K / bb_nRows;
  int n_Csteps = N / bb_nCols;
  int row = 0;
  for (int ri = 0; ri < n_Rsteps; ++ri) {
    int col = 0;
    for (int ci = 0; ci < n_Csteps; ++ci) {
      // process full rectangles
      const float* bSrc = &B[row*ldb + col];
      for (int i = 0; i < bb_nRows; ++i) {
        memcpy(&prm.bb[bb_nCols*i], bSrc, bb_nCols*sizeof(*B));
        bSrc += ldb;
      }
      scalar_noncblas_sgemm_core(&prm, &A[row], &C[col]);
      col += bb_nCols;
    }
    if (col < N) {
      // process rightmost rectangle of the full-height band
      const float* bSrc = &B[row*ldb + col];
      for (int i = 0; i < bb_nRows; ++i) {
        memcpy(&prm.bb[bb_nCols*i], bSrc, (N-col)*sizeof(*B));
        bSrc += ldb;
      }
      scalar_noncblas_sgemm_core_rightmostColumns(&prm, &A[row], &C[col], N-col, bb_nRows);
    }
    row += bb_nRows;
  }
  if (row < K) {
    // bottom band
    int col = 0;
    for (int ci = 0; ci < n_Csteps; ++ci) {
      // process full-width rectangles
      const float* bSrc = &B[row*ldb + col];
      for (int i = 0; i < K-row; ++i) {
        memcpy(&prm.bb[bb_nCols*i], bSrc, bb_nCols*sizeof(*B));
        bSrc += ldb;
      }
      scalar_noncblas_sgemm_core_bottomRows(&prm, &A[row], &C[col], K-row);
      col += bb_nCols;
    }
    if (col < N) {
      // process bottom-right corner rectangle
      const float* bSrc = &B[row*ldb + col];
      for (int i = 0; i < K-row; ++i) {
        memcpy(&prm.bb[bb_nCols*i], bSrc, (N-col)*sizeof(*B));
        bSrc += ldb;
      }
      scalar_noncblas_sgemm_core_rightmostColumns(&prm, &A[row], &C[col], N-col, K-row);
    }
  }
}
