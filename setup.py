from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="flash_attn_cuda",
    ext_modules=[
        CUDAExtension(
            name="flash_attn_cuda",
            sources=[
                "flash_attention/cuda/vector_add.cu",
                "flash_attention/cuda/bindings.cpp",
                "flash_attention/cuda/flash_attention_forward.cu",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)