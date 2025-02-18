from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="StreamBind",
    include_dirs=["include"],
    ext_modules=[
        CUDAExtension(
            "StreamBind",
            ["src/add2.cpp", "src/add2_kernel.cu"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)