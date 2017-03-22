# SGEMM tester

The test runs several variants of sgemm.
Provided: project for VS2013 and build script for gcc/MINGW64

VS2003 pessimizes AVX/FMA inner loops with its insistence on using load-op form of instruction.
gcc '-O1' does better job of just doing what I told it to do.
gcc '-O2' pessimizes things in different ways. With right set of additional flag it's probably possible to convince it
to not pessimize at all. But why bother? '-O1' works.

# Usage:

By default it tests a small size interesting for computer go.
You can change any size, leading dimension, alpha or beta from command line.
For example 
mkl_ts1 K=111 M=79 alpha=666.33
The names of parameters are the same as in standard SGEMM (non-case-sensitive).

You can't change Order, TransA and TransB. 

# License

BSD
