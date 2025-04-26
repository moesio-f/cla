# Benchmark `pycla`/`cla` vs Pure Python 

> [!NOTE]  
> Matrices operation in GPU have not been included due to the pending investigation of [#31](https://github.com/moesio-f/cla/issues/31).

> [!WARNING]  
> The results should be interpreted as **relative** (i.e., CPU/GPU time is hardware-dependent). Therefore, the percentage change between approaches should be fairly hardware-independent given the same conditions (e.g., launch parameters, number of dimensions, vector/matrix values).

## Host Configuration

- Intel i5-10300H (8 cores) @ 4.500GHz;
- NVIDIA GeForce GTX 1650 Mobile;
- 8GB RAM;
- Arch Linux;
- `cla` v1.2.0;
- `pycla` v1.2.0;
- Python 3.13.2;
- CUDA 12.8.93 (NVCC);
- GCC/G++ 14.2.1;

## Results

The table below shows the average execution time for `100` independet runs of operations with Vector/Matrix with dimensions varying from `800` to `1000` (square matrices with those dimensions, `100` runs for each dimension).
 
| Operation | Kind | Approach | Average Execution Time (s) | Standard Deviation | Min | Max |
| --- | --- | --- | --- | --- | --- | --- |
|  `ADD`  | Vector | `pycla` on CPU | **0.000009** | 4.051654e-07 | 0.000009 | 0.000010 |
|         |        | `python` Lists | 0.000049     | 5.285173e-06 | 0.000044 | 0.000055 |
|         |        | `cla` on GPU   | 0.000211     | 0.000019     | 0.000192 | 0.000229 | 
|  `SUB`  | Vector | `pycla` on CPU | **0.000009** | 1.225041e-07 | 0.000009 | 0.000009 |
|         |        | `python` Lists | 0.000040     | 4.318514e-06 | 0.000035 | 0.000044 |
|         |        | `cla` on GPU   | 0.000211     | 0.000019     | 0.000191 | 0.000229 |
|  `MUL`  | Vector | `pycla` on CPU | **0.000010** | 4.355593e-07 | 0.000009 | 0.000010 |
|         |        | `python` Lists | 0.000039     | 3.870907e-06 | 0.000035 | 0.000043 |
|         |        | `cla` on GPU   | 0.000211     | 0.000019     | 0.000191 | 0.000228 |


| Operation | Kind | Approach | Average Execution Time (s) | Standard Deviation | Min | Max |
| --- | --- | --- | --- | --- | --- | --- |
| `ADD` | Matrix | `pycla` on CPU | 0.002233  | 4.999673e-04 | 0.001746  | 0.002745  |
|       |        | `python` Lists | 0.052682  | 1.175074e-02 | 0.041179  | 0.064666  |
| `SUB` | Matrix | `pycla` on CPU | 0.002233  | 4.953388e-04 | 0.001752  | 0.002741  |
|       |        | `python` Lists | 0.045707  | 1.027301e-02 | 0.035670  | 0.056201  |
| `MUL` | Matrix | `pycla` on CPU | 2.269137  | 7.029568e-01 | 1.651169  | 3.033900  | 
|       |        | `python` Lists | 38.241585 | 12.77309     | 26.059212 | 51.533066 |

The results shows that `pycla` in CPU is considerably faster than Python (especially for Matrix operations), while running in GPU with the chosen dimensionality doesn't provide parallelization enough to beat the CPU.
