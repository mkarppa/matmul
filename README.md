# matmul

Computes and benchmarks elementary and Strassen-like binary and
Boolean matrix multiplication on NVIDIA DGX-1 (or compatible) hardware.

## Requirements

* Compute Capability 3.7 or higher
* AVX2
* CUDA SDK 9.0 or higher
* GCC 5.3.0 or higher
* OpenMP 4.0 or higher

## Compilation

Simply run `make`, and a binary called `matmul` is compiled (one
source file, one output file) with `nvcc`.

## Usage

`./matmul <test|evaluate> <transpose|change-of-basis|cubic|boolean|strassen-winograd|absinv|abschain> <n> <m> <l> <a>`

### Parameters
* `test`: run the application in `test` mode
* `evaluate`: run the application in `evaluate` mode
* `transpose`: test or evaluate the host-side submatrix transpose
  functions
* `change-of-basis`: test or evaluate the host-side submatrix
  change-of-basis functions
* `cubic`: use the elementary, cubic binary multiplication algorithm
* `boolean`: use the elementary, cubic Boolean multiplication
  algorithm
* `strassen-winograd`: use the Strassen-Winograd binary multiplication
  algorithm
* `absinv`: use the alternative-basis Strassen-like decomposition for
  binary multiplication with the self-inverse property
* `absinv`: use the alternative-basis Strassen-like decomposition for
  binary multiplication with the chaining property
* `n`: integer, power of 2, that determines the maximum instance size
  (matrix side-length) for testing/evaluation
* `m`: integer, power of 2, that determines the maximum instance size
(matrix side-length) that is to be run on a single accelerator
* `l`: positive integer, determines the number of lanes (how many
  times the pipeline is to be replicated in parallel per GPU)
* `a`: positive integer, the number of auxiliary matrices (how many
  results are to be kept in store per each lane)

Running the application in `test` mode generates random data and
computes the multiplication with the desired algorithm and parameters,
and compares the results to a baseline implementation to check that
the more complex algorithms have been implemented correctly (with high
probability).

In `evaluate` mode, random data is generated and the running time of
the algorithm is evaluated with the data.

To replicate the experiments of the manuscript on a DGX-1, use the following invocations:
```
./matmul evaluate transpose 1048576 131072 1 1
./matmul evaluate change-of-basis 1048576 131072 1 1
./matmul evaluate cubic 1048576 131072 1 1
./matmul evaluate boolean 1048576 131072 1 1
./matmul evaluate strassen-winograd 1048576 65536 2 2
./matmul evaluate absinv 1048576 65536 2 2
./matmul evaluate abschain 1048576 65536 2 2
```

## License

The code is licensed under the MIT license (see the LICENSE file).

Copyright &copy; Matti Karppa 2018-2019
