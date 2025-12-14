from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='my_linear_double_backward',
    ext_modules=[
        CppExtension(
            name='my_linear_double_backward',
            sources=['linear_double_backward.cpp'],
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
