# BGEMM & BWTA (Binary Weight and Ternary Activation)

This is a repository for Binary General Matrix Multiply (BGEMM) and BWTA (Binary Weight and Ternary Activation) GEMM by customized CUDA kernels. Thank FP6-LLM for the wheels! 

# Still developing...

## Installation

- Tested on SM80, SM86 architecture. 

```sh
cd lowbit_kernel && make bgemm
cd .. && pip3 install .
```

## Speed and correctness test
```sh
cd tests/python
# directly test GEMM loops
python3 test_kernel.py  
# tiny training demo of a 3-layer MLP
python3 test_model_demo.py --model=[bnn_bgemm, bnn_fp16, fp16]  
```

## TODO
- [x] Pytorch extension and layers (linear and matmul layers (standard matmul with {-1, 1} $\times$ {-1, 1} and $A \times V$ with {0, 1} $\times$ {-1, 1} ) in [bgemm_linear.py](tests/python/bgemm_linear.py)).
- [x] Simple [MLP](tests/python/test_model_demo.py) demo. 
- [ ] BERT model demo using BGEMM kernel. 
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

