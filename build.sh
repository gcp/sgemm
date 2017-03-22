g++ -std=c++11 -c -O1 tst1.cpp
g++ -std=c++11 -c -O2 -I/opt/intel/mkl/include -L/opt/intel/mkl/lib/intel64 ref_noncblas_sgemm.cpp
g++ -std=c++11 -c -O1 -I/opt/intel/mkl/include -L/opt/intel/mkl/lib/intel64 scalar_noncblas_sgemm.cpp
g++ -std=c++11 -c -O1 -I/opt/intel/mkl/include -mavx -L/opt/intel/mkl/lib/intel64 avx128_noncblas_sgemm.cpp
g++ -std=c++11 -c -O1 -I/opt/intel/mkl/include -mfma -L/opt/intel/mkl/lib/intel64 fma128_noncblas_sgemm.cpp
g++ -std=c++11 -c -O1 -I/opt/intel/mkl/include -mavx -L/opt/intel/mkl/lib/intel64 avx256_noncblas_sgemm.cpp
g++ -std=c++11 -c -O1 -I/opt/intel/mkl/include -mfma -L/opt/intel/mkl/lib/intel64 fma256_noncblas_sgemm.cpp
g++ -std=c++11 -I/opt/intel/mkl/include -L/opt/intel/mkl/lib/intel64 tst1.o ref_noncblas_sgemm.o scalar_noncblas_sgemm.o avx128_noncblas_sgemm.o fma128_noncblas_sgemm.o avx256_noncblas_sgemm.o fma256_noncblas_sgemm.o -o mkl_tst1_gcc -lopenblas
