void ref_noncblas_sgemm(
 int M, int N, int K, 
 float alpha, 
 const float *A, int lda, 
 const float *B, int ldb,
 float beta, 
 float *C, int ldc)
{
  if (beta != 0) {
    for (int m = 0; m < M; A += lda, C += ldc, ++m) {
      for (int n = 0; n < N; ++n) {
        const float *Bcol = &B[n];
        double acc = 0;
        for (int k = 0; k < K; Bcol += ldb, ++k)
          acc += double(A[k]) * Bcol[0];
        C[n] = float(C[n]*beta + acc*alpha);
      }
    }
  } else {
    for (int m = 0; m < M; A += lda, C += ldc, ++m) {
      for (int n = 0; n < N; ++n) {
        const float *Bcol = &B[n];
        double acc = 0;
        for (int k = 0; k < K; Bcol += ldb, ++k)
          acc += double(A[k]) * Bcol[0];
        C[n] = float(acc*alpha);
      }
    }
  }
}
