from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


extra_compile_args = {
    "cxx": [
        "-O3",
        "-std=c++17"
    ],
    "nvcc": [
        "-O3", 
        "--use_fast_math",
        "-std=c++17",
        "-maxrregcount=255",
        "--ptxas-options=-v,-warn-lmem-usage,--warn-on-spills",
        "-gencode=arch=compute_80,code=sm_80"
    ],
}

setup(
    name="lowbit_kernel",
    author="Yifu Ding",
    version="0.1",
    author_email="eveedyf@gmail.com",
    description = "A Binary General Matrix Multiply Kernel. ",
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "transformers"
    ],
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="lowbit_kernel",
            sources=[
                "lowbit_kernel/csrc/pybind.cpp", 
                "lowbit_kernel/csrc/fp6_linear.cu", 
                "lowbit_kernel/csrc/bgemm.cu",
            ],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension}
)