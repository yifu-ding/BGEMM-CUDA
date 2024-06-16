# BGEMM

This is a repository for Binary General Matrix Multiply (BGEMM) by customized CUDA kernel. Thank FP6-LLM for the wheels! 

#### Still developing, weak compatibility. Welcome to PR!

## Installation

- Tested on SM80, SM86 architecture. 

```sh
cd lowbit_kernel && make bgemm
cd .. && pip3 install .
```

## Speed and correctness test
```sh
cd tests/python
python3 test_kernel.py  # directly test GEMM loops
# cd tests/cpp
python3 test_model_demo.py --model=[bnn_bgemm, bnn_fp16, fp16]  # tiny training demo of a 3-layer MLP
```


## Performance

- Require M % 128 == 0, K % 128 == 0, N % 32 == 0. (Will optimize dimention compatibility in future release)

| M     | N (batch size) | K     | SplitK | Iteration | Time/ms |       | Performance/TFLOPs |        |
| ----- | -------------- | ----- | ------ | --------- | ------- | ----- | ------------------ | ------ |
|       |                |       |        |           | cuBLAS  | BGEMM | cuBLAS             | BGEMM  |
| 128   | 32             | 128   | 1      | 10000     | 0.003   | 0.003 | 0.32               | 0.34   |
| 256   | 32             | 256   | 1      | 10000     | 0.009   | 0.004 | 0.45               | 1.19   |
| 1024  | 32             | 1024  | 1      | 10000     | 0.009   | 0.006 | 7.83               | 10.69  |
| 2048  | 32             | 2048  | 1      | 10000     | 0.023   | 0.009 | 11.86              | 30.03  |
| 13824 | 128            | 5120  | 1      | 10000     | 0.25    | 0.073 | 72.61              | 248.71 |
| 5120  | 128            | 13824 | 1      | 10000     | 0.237   | 0.087 | 76.38              | 208.87 |
| 22016 | 128            | 8192  | 1      | 10000     | 0.743   | 0.162 | 62.13              | 284.6  |
| 8192  | 128            | 22016 | 1      | 10000     | 0.732   | 0.148 | 63.05              | 312.75 |

## TODO
- [ ] Pytorch extension and real model demo. 
- [ ] More bitwidth support, e.g., $W_1A_{f16}$, $W_1A_{f8}$. 
- [ ] Support arbitrarily $N$ (batch size). 
- [ ] Optimize Share Memory Usage. 
- [ ] Larger bandwidth instruction support (`m16n8k256`) for further speedup. 

## Reference

- FP6-LLM. [Arxiv](https://arxiv.org/abs/2401.14112)

```
@misc{xia2024fp6llm,
      title={FP6-LLM: Efficiently Serving Large Language Models Through FP6-Centric Algorithm-System Co-Design}, 
      author={Haojun Xia and Zhen Zheng and Xiaoxia Wu and Shiyang Chen and Zhewei Yao and Stephen Youn and Arash Bakhtiari and Michael Wyatt and Donglin Zhuang and Zhongzhu Zhou and Olatunji Ruwase and Yuxiong He and Shuaiwen Leon Song},
      year={2024},
      eprint={2401.14112},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

- [The PTX ISA 8.5](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)

- [cuBLAS: Basic Linear Algebra on NVIDIA GPUs](https://developer.nvidia.com/cublas)

