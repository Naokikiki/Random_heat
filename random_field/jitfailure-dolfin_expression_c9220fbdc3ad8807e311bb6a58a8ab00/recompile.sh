#!/bin/bash
# Execute this file to recompile locally
x86_64-apple-darwin13.4.0-clang++ -Wall -shared -fPIC -std=c++11 -O3 -fno-math-errno -fno-trapping-math -ffinite-math-only -I/Users/naoki/miniconda3/envs/fenicsproject/include -I/Users/naoki/miniconda3/envs/fenicsproject/include/eigen3 -I/Users/naoki/miniconda3/envs/fenicsproject/.cache/dijitso/include dolfin_expression_c9220fbdc3ad8807e311bb6a58a8ab00.cpp -L/Users/naoki/miniconda3/envs/fenicsproject/lib -L/Applications/Xcode_13.2.1.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.9.sdk/usr/lib -L/Users/naoki/miniconda3/envs/fenicsproject/Users/naoki/miniconda3/envs/fenicsproject/lib -L/Users/naoki/miniconda3/envs/fenicsproject/.cache/dijitso/lib -Wl,-rpath,/Users/naoki/miniconda3/envs/fenicsproject/.cache/dijitso/lib -lpmpi -lmpi -lmpicxx -lpetsc -lslepc -lm -ldl -lz -lsz -lpthread -lcurl -lcrypto -lhdf5 -lboost_timer -ldolfin -Wl,-install_name,/Users/naoki/miniconda3/envs/fenicsproject/.cache/dijitso/lib/libdijitso-dolfin_expression_c9220fbdc3ad8807e311bb6a58a8ab00.so -olibdijitso-dolfin_expression_c9220fbdc3ad8807e311bb6a58a8ab00.so